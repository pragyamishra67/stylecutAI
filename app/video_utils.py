import json
import subprocess
from pathlib import Path

import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


# ============================================================
# FFPROBE
# ============================================================

def get_video_metadata(video_path: str) -> dict:
    """
    Extract technical metadata from a video using FFprobe.
    """

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
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

    video_stream = next(
        stream
        for stream in data["streams"]
        if stream["codec_type"] == "video"
    )

    # FPS can look like:
    # 30/1
    # 30000/1001

    fps_string = video_stream.get("r_frame_rate", "0/1")

    numerator, denominator = map(
        int,
        fps_string.split("/")
    )

    fps = numerator / denominator if denominator else 0

    metadata = {
        "duration": float(
            data["format"].get("duration", 0)
        ),

        "fps": fps,

        "width": int(
            video_stream.get("width", 0)
        ),

        "height": int(
            video_stream.get("height", 0)
        ),

        "video_codec": video_stream.get(
            "codec_name"
        ),

        "has_audio": any(
            stream.get("codec_type") == "audio"
            for stream in data["streams"]
        ),
    }

    return metadata


# ============================================================
# PYSCENEDETECT
# ============================================================

def detect_scenes(
    video_path: str,
    threshold: float = 27.0,
) -> list[dict]:
    """
    Detect shot/scene boundaries using PySceneDetect.
    """

    video = open_video(video_path)

    scene_manager = SceneManager()

    scene_manager.add_detector(
        ContentDetector(
            threshold=threshold
        )
    )

    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()

    scenes = []

    for index, (start, end) in enumerate(scene_list):

        scenes.append(
            {
                "id": f"R{index + 1}",

                "start": round(
                    start.get_seconds(),
                    3
                ),

                "end": round(
                    end.get_seconds(),
                    3
                ),

                "start_frame": start.get_frames(),

                "end_frame": end.get_frames(),
            }
        )

    return scenes


# ============================================================
# OPENCV
# ============================================================

def extract_frames(
    video_path: str,
    scenes: list[dict],
    output_dir: str,
    frames_per_scene: int = 5,
) -> list[dict]:
    """
    Extract representative frames from every scene.

    These frames are useful for inspection and can later be
    supplied to Gemini if you want more granular visual evidence.
    """

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    extracted = []

    for scene in scenes:

        scene_id = scene["id"]

        scene_dir = (
            output_path / scene_id
        )

        scene_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        start = scene["start"]
        end = scene["end"]

        duration = end - start

        if frames_per_scene == 1:

            timestamps = [
                start + duration / 2
            ]

        else:

            timestamps = [
                start
                + duration * i / (frames_per_scene - 1)
                for i in range(frames_per_scene)
            ]

        scene_frames = []

        for index, timestamp in enumerate(
            timestamps
        ):

            frame_number = int(
                timestamp * fps
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number
            )

            success, frame = cap.read()

            if not success:
                continue

            frame_path = (
                scene_dir
                / f"frame_{index + 1:02d}.jpg"
            )

            cv2.imwrite(
                str(frame_path),
                frame
            )

            scene_frames.append(
                {
                    "timestamp": round(
                        timestamp,
                        3
                    ),
                    "path": str(
                        frame_path
                    ),
                }
            )

        extracted.append(
            {
                "scene_id": scene_id,
                "frames": scene_frames,
            }
        )

    cap.release()

    return extracted


# ============================================================
# OPENCV MOTION ANALYSIS
# ============================================================

def calculate_motion_metrics(
    video_path: str,
    scenes: list[dict],
) -> list[dict]:
    """
    Calculate a basic frame-difference motion score
    for every detected scene.

    This is NOT trying to understand camera movement.
    It simply gives Gemini additional objective evidence
    about how much visual change occurs within each segment.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    results = []

    for scene in scenes:

        start_frame = int(
            scene["start"] * fps
        )

        end_frame = int(
            scene["end"] * fps
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame
        )

        previous_gray = None

        differences = []

        for _ in range(
            max(1, end_frame - start_frame)
        ):

            success, frame = cap.read()

            if not success:
                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # Resize to make this inexpensive.
            gray = cv2.resize(
                gray,
                (320, 180)
            )

            if previous_gray is not None:

                difference = cv2.absdiff(
                    previous_gray,
                    gray
                )

                score = float(
                    difference.mean()
                )

                differences.append(score)

            previous_gray = gray

        if differences:

            average_motion = (
                sum(differences)
                / len(differences)
            )

            maximum_motion = max(
                differences
            )

        else:

            average_motion = 0.0
            maximum_motion = 0.0

        results.append(
            {
                "scene_id": scene["id"],
                "average_frame_difference": round(
                    average_motion,
                    3
                ),
                "maximum_frame_difference": round(
                    maximum_motion,
                    3
                ),
            }
        )

    cap.release()

    return results