import json
import mimetypes
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError
from scenedetect import ContentDetector, detect

from app.schemas import ReferenceAnalysis


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.7-flash"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# GEMINI CLIENT
# ============================================================

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY was not found in .env"
    )

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# UTILITY
# ============================================================

def ensure_directory(path: Path) -> None:
    """
    Create a directory if it does not exist.
    """

    path.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# FFPROBE
# ============================================================

def get_video_metadata(
    video_path: str
) -> dict[str, Any]:

    """
    Extract technical video metadata using FFprobe.
    """

    print("\n[1/5] Running FFprobe...")

    command = [
        "ffprobe",
        "-v",
        "error",

        "-show_entries",
        "format=duration,format_name,size",

        "-show_entries",
        "stream=codec_name,codec_type,width,height,"
        "r_frame_rate,avg_frame_rate,pix_fmt",

        "-of",
        "json",

        video_path
    ]

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

    except FileNotFoundError:

        raise RuntimeError(
            "FFprobe was not found. "
            "Install FFmpeg and add it to PATH."
        )

    except subprocess.CalledProcessError as e:

        raise RuntimeError(
            f"FFprobe failed:\n{e.stderr}"
        )

    data = json.loads(result.stdout)

    duration = float(
        data.get("format", {}).get(
            "duration",
            0
        )
    )

    streams = data.get(
        "streams",
        []
    )

    video_stream = next(
        (
            stream
            for stream in streams
            if stream.get("codec_type") == "video"
        ),
        {}
    )

    width = video_stream.get(
        "width"
    )

    height = video_stream.get(
        "height"
    )

    fps = parse_frame_rate(
        video_stream.get(
            "avg_frame_rate"
        )
        or video_stream.get(
            "r_frame_rate"
        )
    )

    metadata = {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "codec": video_stream.get(
            "codec_name"
        ),
        "pixel_format": video_stream.get(
            "pix_fmt"
        ),
        "format": data.get(
            "format",
            {}
        ).get(
            "format_name"
        ),
        "size_bytes": data.get(
            "format",
            {}
        ).get(
            "size"
        )
    }

    print(
        f"    Duration: {duration:.3f}s"
    )

    print(
        f"    Resolution: "
        f"{width}x{height}"
    )

    print(
        f"    FPS: {fps:.3f}"
    )

    print(
        f"    Codec: {metadata['codec']}"
    )

    return metadata


# ============================================================
# FPS PARSER
# ============================================================

def parse_frame_rate(
    value: str | None
) -> float:

    """
    Convert FFprobe frame-rate strings such as
    '30000/1001' into a float.
    """

    if not value:
        return 0.0

    try:

        if "/" in value:

            numerator, denominator = value.split(
                "/",
                1
            )

            denominator = float(
                denominator
            )

            if denominator == 0:
                return 0.0

            return (
                float(numerator)
                / denominator
            )

        return float(value)

    except (
        ValueError,
        ZeroDivisionError
    ):

        return 0.0


# ============================================================
# PYSCENEDETECT
# ============================================================

def detect_scenes(
    video_path: str
) -> list[dict[str, Any]]:

    """
    Detect shot/scene boundaries using PySceneDetect.

    These timestamps are authoritative for the pipeline.
    Gemini should NOT modify them.
    """

    print(
        "\n[2/5] Detecting scenes with PySceneDetect..."
    )

    scene_list = detect(
        video_path,
        ContentDetector(
            threshold=27.0,
            min_scene_len=15
        ),
        show_progress=True
    )

    scenes = []

    for index, (
        start_time,
        end_time
    ) in enumerate(
        scene_list,
        start=1
    ):

        start = start_time.get_seconds()
        end = end_time.get_seconds()

        scenes.append(
            {
                "id": f"R{index}",
                "start": round(
                    start,
                    6
                ),
                "end": round(
                    end,
                    6
                )
            }
        )

    print(
        f"    Detected {len(scenes)} segments."
    )

    for scene in scenes:

        print(
            f"    {scene['id']}: "
            f"{scene['start']:.3f}s → "
            f"{scene['end']:.3f}s"
        )

    return scenes


# ============================================================
# OPENCV
# ============================================================

def extract_scene_evidence(
    video_path: str,
    scenes: list[dict[str, Any]],
    output_dir: Path,
    fps: float
) -> list[dict[str, Any]]:

    """
    Use OpenCV to:

    1. Extract representative frames.
    2. Calculate simple motion evidence.

    Gemini receives the actual video separately.
    These frames/metrics provide deterministic evidence
    that can be stored alongside the analysis.
    """

    print(
        "\n[3/5] Extracting OpenCV evidence..."
    )

    frames_dir = output_dir / "frames"

    ensure_directory(
        frames_dir
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "OpenCV could not open the video."
        )

    evidence = []

    for scene in scenes:

        scene_id = scene["id"]

        start = scene["start"]
        end = scene["end"]

        duration = max(
            end - start,
            0.001
        )

        timestamps = [
            start,
            start + duration * 0.5,
            max(
                end - 0.001,
                start
            )
        ]

        scene_frame_dir = (
            frames_dir / scene_id
        )

        ensure_directory(
            scene_frame_dir
        )

        saved_frames = []

        previous_gray = None
        motion_values = []

        for frame_index, timestamp in enumerate(
            timestamps,
            start=1
        ):

            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                timestamp * 1000
            )

            success, frame = cap.read()

            if not success:
                continue

            filename = (
                scene_frame_dir
                / f"frame_{frame_index:02d}.jpg"
            )

            cv2.imwrite(
                str(filename),
                frame
            )

            saved_frames.append(
                str(filename)
            )

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            if previous_gray is not None:

                diff = cv2.absdiff(
                    previous_gray,
                    gray
                )

                motion_score = float(
                    diff.mean()
                )

                motion_values.append(
                    motion_score
                )

            previous_gray = gray

        average_motion = (
            sum(motion_values)
            / len(motion_values)
            if motion_values
            else 0.0
        )

        if average_motion < 5:
            motion_level = "low"

        elif average_motion < 20:
            motion_level = "moderate"

        else:
            motion_level = "high"

        evidence.append(
            {
                "id": scene_id,
                "start": start,
                "end": end,
                "duration": round(
                    duration,
                    6
                ),
                "frames": saved_frames,
                "motion_score": round(
                    average_motion,
                    4
                ),
                "motion_level": motion_level
            }
        )

        print(
            f"    {scene_id}: "
            f"motion={motion_level} "
            f"({average_motion:.2f})"
        )

    cap.release()

    return evidence


# ============================================================
# GEMINI FILE UPLOAD
# ============================================================

def upload_video_to_gemini(
    video_path: str
):

    """
    Upload the reference video using the Gemini Files API.
    """

    print(
        "\n[4/5] Uploading video to Gemini..."
    )

    uploaded_file = client.files.upload(
        file=video_path
    )

    print(
        f"    Uploaded: "
        f"{uploaded_file.name}"
    )

    print(
        "    Waiting for Gemini to process video..."
    )

    while True:

        file_info = client.files.get(
            name=uploaded_file.name
        )

        state = getattr(
            file_info,
            "state",
            None
        )

        state_name = (
            getattr(
                state,
                "name",
                str(state)
            )
            if state
            else None
        )

        if state_name == "ACTIVE":

            print(
                "    Video is ready."
            )

            return file_info

        if state_name in {
            "FAILED",
            "ERROR"
        }:

            raise RuntimeError(
                f"Gemini video processing failed: "
                f"{state_name}"
            )

        print(
            f"    Current state: "
            f"{state_name}"
        )

        time.sleep(5)


# ============================================================
# BUILD GEMINI PROMPT
# ============================================================

def build_gemini_prompt(
    metadata: dict[str, Any],
    scenes: list[dict[str, Any]],
    evidence: list[dict[str, Any]]
) -> str:

    """
    Construct the semantic-analysis prompt.

    IMPORTANT:
    We give Gemini the deterministic scene boundaries.
    Gemini interprets what happens inside them.
    """

    scene_information = []

    for scene, ev in zip(
        scenes,
        evidence
    ):

        scene_information.append(
            {
                "id": scene["id"],
                "start": scene["start"],
                "end": scene["end"],
                "duration": ev["duration"],
                "opencv_motion_score": ev[
                    "motion_score"
                ],
                "opencv_motion_level": ev[
                    "motion_level"
                ]
            }
        )

    return f"""
You are an expert professional video editor and video-analysis system.

Analyze the supplied reference video.

Your task is to reverse-engineer the editing structure of the video.

IMPORTANT:

The shot boundaries have already been detected by PySceneDetect.

You MUST preserve those exact segment IDs and timestamps.

Do NOT create new segments.

Do NOT merge segments.

Do NOT change segment start timestamps.

Do NOT change segment end timestamps.

Your job is to semantically analyze what happens inside each
predefined segment.

For every segment determine:

1. Shot type
2. Camera movement
3. Subject movement
4. Motion intensity
5. Playback speed
6. Visible editing effects
7. Crop/reframing
8. Concise description

Also identify the transition type between consecutive segments.

Possible transition types include:

- hard_cut
- crossfade
- fade_in
- fade_out
- wipe
- dip_to_black
- unknown

For camera movement use descriptions such as:

- static
- pan
- tilt
- zoom_in
- zoom_out
- tracking
- handheld
- orbit
- unknown

For speed use:

- normal
- slow_motion
- fast_motion
- unknown

For editing effects, describe only effects that are visually
supported by the video.

Do not hallucinate effects.

----------------------------------------
VIDEO METADATA
----------------------------------------

{json.dumps(
    metadata,
    indent=2
)}

----------------------------------------
DETECTED SEGMENTS
----------------------------------------

{json.dumps(
    scene_information,
    indent=2
)}

----------------------------------------
ANALYSIS RULE
----------------------------------------

Use the actual supplied video as the primary source of truth
for visual interpretation.

Use the OpenCV evidence as supporting evidence.

Use PySceneDetect timestamps as the authoritative segment
boundaries.

Return the structured analysis only.
"""


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_with_gemini(
    video_file,
    metadata: dict[str, Any],
    scenes: list[dict[str, Any]],
    evidence: list[dict[str, Any]]
) -> ReferenceAnalysis:

    """
    Ask Gemini to semantically analyze the reference video.
    """

    print(
        "\n[5/5] Asking Gemini to analyze "
        "the reference video..."
    )

    prompt = build_gemini_prompt(
        metadata,
        scenes,
        evidence
    )

    response = client.models.generate_content(

        model=MODEL,

        contents=[
            prompt,
            video_file
        ],

        config=types.GenerateContentConfig(

            response_mime_type="application/json",

            response_schema=ReferenceAnalysis,

            temperature=0.1
        )
    )

    if not response.text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    print(
        "    Gemini response received."
    )

    try:

        analysis = ReferenceAnalysis.model_validate_json(
            response.text
        )

    except ValidationError as e:

        raise RuntimeError(
            "Gemini returned JSON that could not "
            "be validated against ReferenceAnalysis.\n\n"
            f"{e}\n\n"
            f"Raw response:\n{response.text}"
        )

    return analysis


# ============================================================
# DETERMINISTIC VALIDATION
# ============================================================

def validate_analysis(
    analysis: ReferenceAnalysis,
    scenes: list[dict[str, Any]],
    video_duration: float
) -> None:

    """
    Validate Gemini's output against deterministic
    FFprobe and PySceneDetect information.
    """

    print(
        "\nValidating Gemini analysis..."
    )

    # --------------------------------------------------------
    # Video duration
    # --------------------------------------------------------

    if analysis.video_duration < 0:

        raise ValueError(
            "Gemini returned a negative video duration."
        )

    if (
        analysis.video_duration
        > video_duration + 0.5
    ):

        raise ValueError(
            "Gemini returned a video duration "
            "greater than the actual video duration."
        )

    # --------------------------------------------------------
    # Segment count
    # --------------------------------------------------------

    if len(analysis.segments) != len(scenes):

        raise ValueError(
            "Gemini returned "
            f"{len(analysis.segments)} segments, "
            f"but PySceneDetect detected "
            f"{len(scenes)}."
        )

    # --------------------------------------------------------
    # Segment validation
    # --------------------------------------------------------

    for expected, actual in zip(
        scenes,
        analysis.segments
    ):

        expected_id = expected["id"]

        # ID

        if actual.id != expected_id:

            raise ValueError(
                f"Segment ID mismatch. "
                f"Expected {expected_id}, "
                f"received {actual.id}."
            )

        # Start

        if actual.start < 0:

            raise ValueError(
                f"{actual.id}: "
                "start timestamp cannot be negative."
            )

        # End

        if actual.end <= actual.start:

            raise ValueError(
                f"{actual.id}: "
                "end timestamp must be greater "
                "than start timestamp."
            )

        # Duration boundary

        if actual.end > video_duration + 0.5:

            raise ValueError(
                f"{actual.id}: "
                "end timestamp exceeds "
                "video duration."
            )

        # Compare with PySceneDetect

        start_difference = abs(
            actual.start
            - expected["start"]
        )

        end_difference = abs(
            actual.end
            - expected["end"]
        )

        if start_difference > 0.01:

            raise ValueError(
                f"{actual.id}: Gemini changed "
                f"the start timestamp.\n"
                f"Expected: {expected['start']}\n"
                f"Received: {actual.start}"
            )

        if end_difference > 0.01:

            raise ValueError(
                f"{actual.id}: Gemini changed "
                f"the end timestamp.\n"
                f"Expected: {expected['end']}\n"
                f"Received: {actual.end}"
            )

    # --------------------------------------------------------
    # Chronological ordering
    # --------------------------------------------------------

    for previous, current in zip(
        analysis.segments,
        analysis.segments[1:]
    ):

        if current.start < previous.end - 0.01:

            raise ValueError(
                f"Segments overlap: "
                f"{previous.id} → {current.id}"
            )

    # --------------------------------------------------------
    # Transition validation
    # --------------------------------------------------------

    for transition in analysis.transitions:

        if transition.timestamp < 0:

            raise ValueError(
                "Transition timestamp cannot be negative."
            )

        if (
            transition.timestamp
            > video_duration
        ):

            raise ValueError(
                "Transition timestamp exceeds "
                "video duration."
            )

    print(
        "    Validation successful."
    )


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    data: Any,
    output_path: Path
) -> None:

    """
    Save JSON with readable formatting.
    """

    ensure_directory(
        output_path.parent
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def analyze_reference_video(
    video_path: str,
    output_dir: str = "output"
) -> ReferenceAnalysis:

    """
    Complete reference-video analysis pipeline.

    Pipeline:

    Video
       ↓
    FFprobe
       ↓
    PySceneDetect
       ↓
    OpenCV
       ↓
    Gemini
       ↓
    Pydantic
       ↓
    Deterministic validation
       ↓
    reference_analysis.json
    """

    video_path = str(
        Path(video_path)
    )

    output_path = Path(
        output_dir
    )

    ensure_directory(
        output_path
    )

    # --------------------------------------------------------
    # Check video
    # --------------------------------------------------------

    if not Path(video_path).exists():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    print(
        "\n=========================================="
    )

    print(
        "   REFERENCE VIDEO ANALYSIS"
    )

    print(
        "=========================================="
    )

    print(
        f"\nVideo: {video_path}"
    )

    # --------------------------------------------------------
    # STEP 1 — FFprobe
    # --------------------------------------------------------

    metadata = get_video_metadata(
        video_path
    )

    save_json(
        metadata,
        output_path / "metadata.json"
    )

    # --------------------------------------------------------
    # STEP 2 — PySceneDetect
    # --------------------------------------------------------

    scenes = detect_scenes(
        video_path
    )

    save_json(
        {
            "segments": scenes
        },
        output_path / "scenes.json"
    )

    # --------------------------------------------------------
    # STEP 3 — OpenCV
    # --------------------------------------------------------

    evidence = extract_scene_evidence(
        video_path=video_path,
        scenes=scenes,
        output_dir=output_path,
        fps=metadata["fps"]
    )

    save_json(
        {
            "segments": evidence
        },
        output_path / "analysis_evidence.json"
    )

    # --------------------------------------------------------
    # STEP 4 — Gemini upload
    # --------------------------------------------------------

    video_file = upload_video_to_gemini(
        video_path
    )

    # --------------------------------------------------------
    # STEP 5 — Gemini semantic analysis
    # --------------------------------------------------------

    analysis = analyze_with_gemini(
        video_file=video_file,
        metadata=metadata,
        scenes=scenes,
        evidence=evidence
    )

    # --------------------------------------------------------
    # STEP 6 — Deterministic validation
    # --------------------------------------------------------

    validate_analysis(
        analysis=analysis,
        scenes=scenes,
        video_duration=metadata["duration"]
    )

    # --------------------------------------------------------
    # STEP 7 — Save final JSON
    # --------------------------------------------------------

    final_path = (
        output_path
        / "reference_analysis.json"
    )

    save_json(
        analysis.model_dump(),
        final_path
    )

    print(
        "\n=========================================="
    )

    print(
        "   ANALYSIS COMPLETE"
    )

    print(
        "=========================================="
    )

    print(
        f"\nSaved:"
        f"\n{final_path}"
    )

    return analysis


# ============================================================
# COMMAND LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    analyze_reference_video(
        video_path="data/reference.mp4",
        output_dir="output"
    )