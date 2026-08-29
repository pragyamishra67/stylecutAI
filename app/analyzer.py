import json
import mimetypes
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import requests

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError
from scenedetect import ContentDetector, detect

from app.schemas import (
    ReferenceAnalysis,
    TargetAnalysis,
    EditSpec,
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

LLAMA_MODEL = os.getenv(
    "LLAMA_MODEL",
    "llama3.1:8b"
)


if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY was not found in .env"
    )


# ============================================================
# CLIENTS
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# UTILITY
# ============================================================

def ensure_directory(path: Path) -> None:

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

    print(
        f"\nRunning FFprobe: {video_path}"
    )

    command = [
        "ffprobe",
        "-v",
        "error",

        "-show_entries",
        (
            "format=duration,"
            "format_name,size"
        ),

        "-show_entries",
        (
            "stream=codec_name,"
            "codec_type,"
            "width,"
            "height,"
            "r_frame_rate,"
            "avg_frame_rate,"
            "pix_fmt"
        ),

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
            "Make sure FFmpeg is installed "
            "and added to PATH."
        )

    except subprocess.CalledProcessError as e:

        raise RuntimeError(
            f"FFprobe failed:\n{e.stderr}"
        )

    data = json.loads(
        result.stdout
    )

    duration = float(
        data
        .get("format", {})
        .get("duration", 0)
    )

    streams = data.get(
        "streams",
        []
    )

    video_stream = next(
        (
            stream
            for stream in streams
            if stream.get("codec_type")
            == "video"
        ),
        {}
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

        "width": video_stream.get(
            "width"
        ),

        "height": video_stream.get(
            "height"
        ),

        "fps": fps,

        "codec": video_stream.get(
            "codec_name"
        ),

        "pixel_format": video_stream.get(
            "pix_fmt"
        ),

        "format": data
        .get("format", {})
        .get("format_name"),

        "size_bytes": data
        .get("format", {})
        .get("size")
    }

    return metadata


def parse_frame_rate(
    value: str | None
) -> float | None:

    if not value:
        return None

    try:

        if "/" in value:

            numerator, denominator = (
                value.split("/")
            )

            denominator = float(
                denominator
            )

            if denominator == 0:
                return None

            return float(
                numerator
            ) / denominator

        return float(value)

    except (
        ValueError,
        ZeroDivisionError
    ):

        return None


# ============================================================
# PYSCENEDETECT
# ============================================================

def detect_scenes(
    video_path: str
) -> list[dict[str, Any]]:

    print(
        "\nRunning PySceneDetect..."
    )

    scene_list = detect(
        video_path,
        ContentDetector(
            threshold=27.0
        )
    )

    scenes = []

    for index, (
        start_time,
        end_time
    ) in enumerate(
        scene_list,
        start=1
    ):

        scenes.append({

            "id": f"S{index:03d}",

            "start": start_time.get_seconds(),

            "end": end_time.get_seconds(),

            "duration": (
                end_time.get_seconds()
                - start_time.get_seconds()
            )
        })

    print(
        f"Detected {len(scenes)} scenes."
    )

    return scenes


# ============================================================
# OPENCV EVIDENCE
# ============================================================

def extract_opencv_evidence(
    video_path: str,
    scenes: list[dict[str, Any]],
    output_dir: str
) -> list[dict[str, Any]]:

    print(
        "\nRunning OpenCV analysis..."
    )

    output_path = Path(
        output_dir
    )

    ensure_directory(
        output_path
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "OpenCV could not open video."
        )

    evidence = []

    for scene in scenes:

        start = scene["start"]
        end = scene["end"]

        duration = end - start

        timestamps = [

            start,

            start + duration * 0.5,

            max(
                start,
                end - 0.01
            )
        ]

        previous_gray = None
        motion_values = []

        saved_frames = []

        scene_dir = (
            output_path
            / scene["id"]
        )

        ensure_directory(
            scene_dir
        )

        for frame_number, timestamp in enumerate(
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

            frame_path = (
                scene_dir
                / f"frame_{frame_number:02d}.jpg"
            )

            cv2.imwrite(
                str(frame_path),
                frame
            )

            saved_frames.append(
                str(frame_path)
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

                score = float(
                    diff.mean()
                )

                motion_values.append(
                    score
                )

            previous_gray = gray

        if motion_values:

            motion_score = sum(
                motion_values
            ) / len(motion_values)

        else:

            motion_score = 0.0

        if motion_score < 5:

            motion_level = "low"

        elif motion_score < 20:

            motion_level = "medium"

        else:

            motion_level = "high"

        evidence.append({

            "id": scene["id"],

            "duration": duration,

            "motion_score": motion_score,

            "motion_level": motion_level,

            "frames": saved_frames
        })

    cap.release()

    return evidence


# ============================================================
# GEMINI VIDEO UPLOAD
# ============================================================

def upload_video_to_gemini(
    video_path: str
):

    print(
        "\nUploading video to Gemini..."
    )

    mime_type = (
        mimetypes.guess_type(
            video_path
        )[0]
        or "video/mp4"
    )

    file_info = (
        gemini_client.files.upload(
            file=video_path,
            config=types.UploadFileConfig(
                mime_type=mime_type
            )
        )
    )

    print(
        f"Uploaded: {file_info.name}"
    )

    while True:

        file_info = (
            gemini_client.files.get(
                name=file_info.name
            )
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

        print(
            f"Gemini video state: "
            f"{state_name}"
        )

        if state_name == "ACTIVE":

            print(
                "Gemini video is ready."
            )

            return file_info

        if state_name in {
            "FAILED",
            "ERROR"
        }:

            raise RuntimeError(
                f"Gemini video processing "
                f"failed: {state_name}"
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

    scene_information = []

    for scene, ev in zip(
        scenes,
        evidence
    ):

        scene_information.append({

            "id": scene["id"],

            "start": scene["start"],

            "end": scene["end"],

            "duration": ev["duration"],

            "visual_change_score":
                ev["motion_score"],

            "visual_change_level":
                ev["motion_level"]
        })

    return f"""
You are a professional video analysis system.

Analyze the supplied REFERENCE VIDEO.

Your job is to understand the visual and editing
characteristics of every predefined segment.

IMPORTANT:

The segments have already been detected by
PySceneDetect.

Do NOT create segments.

Do NOT merge segments.

Do NOT change their timestamps.

Use the supplied video as the primary source
of truth.

For every segment determine:

1. shot type
2. camera movement
3. subject movement
4. motion intensity
5. playback speed
6. visible editing effects
7. crop/reframing
8. concise visual description

Also determine the transition between
consecutive segments.

Possible transitions:

- hard_cut
- crossfade
- fade_in
- fade_out
- wipe
- dip_to_black
- unknown

Possible camera movement:

- static
- pan
- tilt
- zoom_in
- zoom_out
- tracking
- handheld
- orbit
- unknown

Possible speed:

- normal
- slow_motion
- fast_motion
- unknown

Do not hallucinate effects.

Technical metadata:

{json.dumps(
    metadata,
    indent=2
)}

Detected segments:

{json.dumps(
    scene_information,
    indent=2
)}

Return ONLY the structured analysis.
"""


# ============================================================
# GEMINI REFERENCE ANALYSIS
# ============================================================

def analyze_reference_with_gemini(
    video_file,
    metadata,
    scenes,
    evidence
) -> ReferenceAnalysis:

    print(
        "\nAnalyzing reference with Gemini..."
    )

    prompt = build_gemini_prompt(
        metadata,
        scenes,
        evidence
    )

    response = (
        gemini_client
        .models
        .generate_content(

            model=GEMINI_MODEL,

            contents=[
                video_file,
                prompt
            ],

            config=types.GenerateContentConfig(

                response_mime_type=(
                    "application/json"
                ),

                response_schema=(
                    ReferenceAnalysis
                ),

                temperature=0.1
            )
        )
    )

    if not response.text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    try:

        analysis = (
            ReferenceAnalysis
            .model_validate_json(
                response.text
            )
        )

    except ValidationError as e:

        raise RuntimeError(
            "Gemini returned invalid "
            "ReferenceAnalysis JSON.\n\n"
            f"{e}\n\n"
            f"Raw response:\n"
            f"{response.text}"
        )

    return analysis


# ============================================================
# TARGET ANALYSIS
# ============================================================

def analyze_target(
    video_path: str,
    output_dir: str
) -> TargetAnalysis:

    print(
        "\nAnalyzing target video..."
    )

    metadata = get_video_metadata(
        video_path
    )

    scenes = detect_scenes(
        video_path
    )

    evidence = extract_opencv_evidence(
        video_path,
        scenes,
        os.path.join(
            output_dir,
            "target_frames"
        )
    )

    return TargetAnalysis(
        video_duration=metadata[
            "duration"
        ],

        segments=[
            {
                "id": scene["id"],
                "start": scene["start"],
                "end": scene["end"],
                "duration": scene["duration"],
                "motion_score": ev[
                    "motion_score"
                ],
                "motion_level": ev[
                    "motion_level"
                ]
            }

            for scene, ev in zip(
                scenes,
                evidence
            )
        ]
    )


# ============================================================
# LLAMA PROMPT
# ============================================================

def build_llama_prompt(
    reference_analysis: ReferenceAnalysis,
    target_analysis: TargetAnalysis
) -> str:

    return f"""
You are an expert video editing planner.

You are NOT a video-understanding model.

Do not attempt to infer visual information
that is not present in the supplied analyses.

Your job is to transform the REFERENCE VIDEO'S
editing style into a deterministic EditSpec
for the TARGET VIDEO.

REFERENCE ANALYSIS:

{reference_analysis.model_dump_json(
    indent=2
)}

TARGET ANALYSIS:

{target_analysis.model_dump_json(
    indent=2
)}

Determine:

1. Which target segments correspond to
   reference segments.

2. What editing operations should be
   applied to the target.

3. The exact target segment on which
   each operation should operate.

4. Operation parameters.

Possible operations include:

- trim
- speed
- crop
- resize
- zoom
- fade
- crossfade
- transition
- reverse
- volume

IMPORTANT:

Do not generate Python code.

Do not generate MoviePy code.

Do not generate FFmpeg commands.

Return ONLY an EditSpec-compatible JSON object.

Every operation must reference an
existing target segment.

Do not invent target segment IDs.

Keep the EditSpec deterministic.
"""


# ============================================================
# LLAMA EDITSPEC GENERATION
# ============================================================

def generate_editspec_with_llama(
    reference_analysis: ReferenceAnalysis,
    target_analysis: TargetAnalysis
) -> EditSpec:

    print(
        "\nGenerating EditSpec with "
        "Llama-3.1-8B-Instruct..."
    )

    prompt = build_llama_prompt(
        reference_analysis,
        target_analysis
    )

    response = requests.post(

        f"{OLLAMA_URL}/api/generate",

        json={

            "model": LLAMA_MODEL,

            "prompt": prompt,

            "stream": False,

            "format": "json",

            "options": {
                "temperature": 0.1
            }
        },

        timeout=300
    )

    response.raise_for_status()

    data = response.json()

    raw_text = data.get(
        "response"
    )

    if not raw_text:

        raise RuntimeError(
            "Llama returned an empty response."
        )

    try:

        editspec = (
            EditSpec
            .model_validate_json(
                raw_text
            )
        )

    except ValidationError as e:

        raise RuntimeError(
            "Llama returned an invalid "
            "EditSpec.\n\n"
            f"{e}\n\n"
            f"Raw response:\n"
            f"{raw_text}"
        )

    return editspec


# ============================================================
# EDITSPEC VALIDATION
# ============================================================

def validate_editspec(
    editspec: EditSpec,
    target_analysis: TargetAnalysis
) -> None:

    print(
        "\nValidating EditSpec..."
    )

    target_ids = {
        segment["id"]
        for segment
        in target_analysis.segments
    }

    for operation in editspec.operations:

        if (
            operation.target_segment
            not in target_ids
        ):

            raise ValueError(
                "EditSpec references "
                f"unknown target segment: "
                f"{operation.target_segment}"
            )

    print(
        "EditSpec validation passed."
    )


# ============================================================
# COMPLETE ANALYSIS PIPELINE
# ============================================================

def analyze_videos(
    reference_path: str,
    target_path: str,
    output_dir: str
):

    output_dir = Path(
        output_dir
    )

    ensure_directory(
        output_dir
    )

    # --------------------------------------------------------
    # REFERENCE
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "REFERENCE VIDEO"
    )

    print(
        "========================================"
    )

    reference_metadata = (
        get_video_metadata(
            reference_path
        )
    )

    reference_scenes = (
        detect_scenes(
            reference_path
        )
    )

    reference_evidence = (
        extract_opencv_evidence(

            reference_path,

            reference_scenes,

            str(
                output_dir
                / "reference_frames"
            )
        )
    )

    reference_video = (
        upload_video_to_gemini(
            reference_path
        )
    )

    reference_analysis = (
        analyze_reference_with_gemini(

            reference_video,

            reference_metadata,

            reference_scenes,

            reference_evidence
        )
    )

    # --------------------------------------------------------
    # SAVE REFERENCE ANALYSIS
    # --------------------------------------------------------

    reference_json = (
        output_dir
        / "reference_analysis.json"
    )

    reference_json.write_text(

        reference_analysis
        .model_dump_json(indent=2),

        encoding="utf-8"
    )

    print(
        f"\nSaved: {reference_json}"
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "TARGET VIDEO"
    )

    print(
        "========================================"
    )

    target_analysis = analyze_target(

        target_path,

        str(output_dir)
    )

    target_json = (
        output_dir
        / "target_analysis.json"
    )

    target_json.write_text(

        target_analysis
        .model_dump_json(indent=2),

        encoding="utf-8"
    )

    print(
        f"Saved: {target_json}"
    )

    # --------------------------------------------------------
    # LLAMA
    # --------------------------------------------------------

    editspec = (
        generate_editspec_with_llama(

            reference_analysis,

            target_analysis
        )
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    validate_editspec(
        editspec,
        target_analysis
    )

    # --------------------------------------------------------
    # SAVE EDITSPEC
    # --------------------------------------------------------

    editspec_json = (
        output_dir
        / "editspec.json"
    )

    editspec_json.write_text(

        editspec
        .model_dump_json(indent=2),

        encoding="utf-8"
    )

    print(
        f"\nSaved: {editspec_json}"
    )

    return {

        "reference_analysis":
            reference_analysis,

        "target_analysis":
            target_analysis,

        "editspec":
            editspec
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    analyze_videos(

        reference_path=(
            "data/reference.mp4"
        ),

        target_path=(
            "data/target.mp4"
        ),

        output_dir=(
            "output"
        )
    )