import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import cv2

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


# ============================================================
# FFPROBE
# ============================================================

def ffprobe_metadata(video_path: str) -> Dict[str, Any]:

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream="
            "width,"
            "height,"
            "r_frame_rate,"
            "avg_frame_rate,"
            "nb_frames,"
            "codec_name,"
            "pix_fmt"
        ),
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        video_path,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(result.stdout)

    stream = data["streams"][0]
    fmt = data["format"]

    fps_value = (
        stream.get("avg_frame_rate")
        or stream.get("r_frame_rate")
        or "0/1"
    )

    numerator, denominator = fps_value.split("/")

    denominator = float(denominator)

    fps = (
        float(numerator) / denominator
        if denominator != 0
        else 0.0
    )

    raw_frames = stream.get("nb_frames")
    frame_count = int(raw_frames) if raw_frames and str(raw_frames).isdigit() else 0
    if frame_count <= 0:
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except Exception:
            pass

    duration = float(
        fmt.get("duration", 0)
    )
    if frame_count <= 0 and fps > 0 and duration > 0:
        frame_count = int(round(duration * fps))

    return {
        "duration_seconds": duration,

        "duration": duration,

        "width": int(
            stream.get("width", 0)
        ),

        "height": int(
            stream.get("height", 0)
        ),

        "fps": fps,

        "frame_count": frame_count,

        "codec": stream.get("codec_name"),

        "pixel_format": stream.get("pix_fmt"),
    }


# ============================================================
# PYSCENEDETECT
# ============================================================

def detect_scenes(
    video_path: str,
    threshold: float = 27.0,
    min_scene_len: int = 15,
) -> List[Dict[str, Any]]:

    video = open_video(video_path)

    manager = SceneManager()

    detector = ContentDetector(
        threshold=threshold,
        min_scene_len=min_scene_len,
    )

    manager.add_detector(detector)

    manager.detect_scenes(video)

    scene_list = manager.get_scene_list()

    scenes = []

    for index, (start, end) in enumerate(
        scene_list,
        start=1,
    ):

        start_seconds = start.get_seconds()
        end_seconds = end.get_seconds()

        scenes.append(
            {
                "segment_id": (
                    f"segment_{index:03d}"
                ),

                "start_time": start_seconds,

                "end_time": end_seconds,

                "duration": (
                    end_seconds - start_seconds
                ),
            }
        )

    # No cuts = entire video is one segment.
    if not scenes:

        metadata = ffprobe_metadata(
            video_path
        )

        duration = metadata[
            "duration_seconds"
        ]

        scenes.append(
            {
                "segment_id": "segment_001",
                "start_time": 0.0,
                "end_time": duration,
                "duration": duration,
            }
        )

    return scenes


# ============================================================
# OPENCV REPRESENTATIVE FRAMES
# ============================================================

def extract_representative_frames(
    video_path: str,
    output_dir: str,
    start_time: float,
    end_time: float,
    samples: int = 5,
) -> List[Dict[str, Any]]:

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open {video_path}"
        )

    duration = end_time - start_time

    frames = []

    for i in range(samples):

        if samples == 1:
            relative = duration / 2
        else:
            relative = (
                duration * i /
                (samples - 1)
            )

        timestamp = (
            start_time + relative
        )

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000,
        )

        success, frame = cap.read()

        if not success:
            continue

        output_path = (
            output_dir /
            f"{timestamp:.3f}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            frame,
        )

        frames.append(
            {
                "timestamp": timestamp,
                "path": str(output_path),
            }
        )

    cap.release()

    return frames


# ============================================================
# OPENCV MOTION ANALYSIS
# ============================================================

def analyze_motion(
    video_path: str,
    start_time: float,
    end_time: float,
    samples: int = 20,
) -> Dict[str, Any]:

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open {video_path}"
        )

    duration = end_time - start_time

    previous = None
    scores = []

    for i in range(samples):

        timestamp = (
            start_time +
            duration * i /
            max(samples - 1, 1)
        )

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000,
        )

        success, frame = cap.read()

        if not success:
            continue

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.resize(
            gray,
            (320, 180),
        )

        if previous is not None:

            difference = cv2.absdiff(
                previous,
                gray,
            )

            scores.append(
                float(difference.mean())
            )

        previous = gray

    cap.release()

    if not scores:

        return {
            "motion_score": 0.0,
            "motion_level": "unknown",
        }

    score = sum(scores) / len(scores)

    if score < 3:
        level = "very_low"
    elif score < 8:
        level = "low"
    elif score < 15:
        level = "medium"
    elif score < 25:
        level = "high"
    else:
        level = "very_high"

    return {
        "motion_score": round(score, 3),
        "motion_level": level,
    }


# ============================================================
# FFMPEG SEGMENT EXTRACTION
# ============================================================

def extract_segment(
    video_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
):

    duration = end_time - start_time

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time),
        "-i",
        video_path,
        "-t",
        str(duration),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c",
        "copy",
        output_path,
    ]

    subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )


# ============================================================
# FFMPEG RUN
# ============================================================

def run_ffmpeg(
    command: List[str],
):

    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )