import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .schemas import (
    ReferenceAnalysis,
    TargetAnalysis,
)

from .video_utils import (
    ffprobe_metadata,
    detect_scenes,
    extract_representative_frames,
    analyze_motion,
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GEMINI MODEL
# ============================================================

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash",
)


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client(api_key_env: str):
    """
    Create a Gemini client using the API key stored in
    the specified environment variable.

    Reference analysis:
        GEMINI_API_KEY

    Target analysis:
        GEMINI_API_KEY1
    """

    api_key = os.getenv(api_key_env)

    if not api_key:
        raise RuntimeError(
            f"{api_key_env} is missing from environment variables."
        )

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# UPLOAD VIDEO TO GEMINI
# ============================================================

def upload_video(
    client,
    video_path: str,
):
    """
    Upload a video using the Gemini Files API and wait until
    Gemini has finished processing it.
    """

    uploaded = client.files.upload(
        file=video_path
    )

    while True:

        state = (
            uploaded.state.name
            if uploaded.state
            else None
        )

        # ----------------------------------------------------
        # VIDEO READY
        # ----------------------------------------------------

        if state == "ACTIVE":
            return uploaded

        # ----------------------------------------------------
        # PROCESSING FAILED
        # ----------------------------------------------------

        if state == "FAILED":
            raise RuntimeError(
                f"Gemini video processing failed: "
                f"{uploaded.name}"
            )

        # ----------------------------------------------------
        # STILL PROCESSING
        # ----------------------------------------------------

        time.sleep(3)

        uploaded = client.files.get(
            name=uploaded.name
        )


# ============================================================
# GEMINI JSON GENERATION
# ============================================================

def generate_json(
    client,
    contents,
    schema,
):
    """
    Send content to Gemini and request structured JSON
    conforming to the supplied Pydantic schema.
    """

    response = client.models.generate_content(

        model=GEMINI_MODEL,

        contents=contents,

        config=types.GenerateContentConfig(

            response_mime_type="application/json",

            response_schema=schema,

            temperature=0.1,
        ),
    )

    return json.loads(
        response.text
    )


# ============================================================
# REFERENCE VIDEO ANALYSIS
# ============================================================

def analyze_reference_video(
    video_path: str,
    workspace_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Analyze reference.mp4.

    Pipeline:

        FFprobe
            ↓
        PySceneDetect
            ↓
        OpenCV representative frames
            ↓
        OpenCV motion analysis
            ↓
        Gemini semantic video analysis
            ↓
        Pydantic validation
            ↓
        reference_analysis

    IMPORTANT:

    This function uses:

        GEMINI_API_KEY

    It does NOT use GEMINI_API_KEY1.
    """

    # --------------------------------------------------------
    # RESOLVE VIDEO PATH
    # --------------------------------------------------------

    video_path = str(
        Path(video_path).resolve()
    )

    # --------------------------------------------------------
    # CHECK VIDEO
    # --------------------------------------------------------

    if not Path(video_path).exists():

        raise FileNotFoundError(
            f"Reference video does not exist: "
            f"{video_path}"
        )

    # --------------------------------------------------------
    # 1. FFPROBE METADATA
    # --------------------------------------------------------

    print(
        "\n[REFERENCE] Running FFprobe..."
    )

    metadata = ffprobe_metadata(
        video_path
    )

    # --------------------------------------------------------
    # 2. PYSCENEDETECT
    # --------------------------------------------------------

    print(
        "[REFERENCE] Running PySceneDetect..."
    )

    segments = detect_scenes(
        video_path
    )

    print(
        f"[REFERENCE] Detected "
        f"{len(segments)} segments."
    )

    # --------------------------------------------------------
    # 3. OPENCV
    # --------------------------------------------------------

    print(
        "[REFERENCE] Running OpenCV analysis..."
    )

    workspace = (
        Path(workspace_dir) / "reference_frames"
        if workspace_dir is not None
        else Path("workspace/reference_frames")
    )

    for segment in segments:

        # ----------------------------------------------------
        # REPRESENTATIVE FRAMES
        # ----------------------------------------------------

        frames = extract_representative_frames(

            video_path,

            str(workspace),

            segment["start_time"],

            segment["end_time"],

            samples=5,
        )

        # ----------------------------------------------------
        # MOTION ANALYSIS
        # ----------------------------------------------------

        motion = analyze_motion(

            video_path,

            segment["start_time"],

            segment["end_time"],
        )

        segment["frames"] = frames

        segment["motion_score"] = (
            motion["motion_score"]
        )

    # --------------------------------------------------------
    # 4. REFERENCE GEMINI CLIENT
    # --------------------------------------------------------

    print(
        "[REFERENCE] Initializing Gemini using "
        "GEMINI_API_KEY..."
    )

    client = get_gemini_client(
        "GEMINI_API_KEY"
    )

    # --------------------------------------------------------
    # 5. UPLOAD REFERENCE VIDEO
    # --------------------------------------------------------

    print(
        "[REFERENCE] Uploading reference.mp4 to Gemini..."
    )

    uploaded_video = upload_video(

        client,

        video_path,
    )

    # --------------------------------------------------------
    # 6. BUILD DETERMINISTIC ANALYSIS
    # --------------------------------------------------------

    deterministic_data = {

        "metadata": metadata,

        "segments": segments,
    }

    # --------------------------------------------------------
    # 7. GEMINI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a reference-video analysis agent.

Your job is to analyze the supplied reference video
and produce a structured reference_analysis.json.

The video has already been analyzed using:

- FFprobe
- PySceneDetect
- OpenCV

The supplied segment boundaries are authoritative.

DO NOT invent new segment boundaries.

Analyze every segment.

For every segment determine:

- segment ID
- start time
- end time
- duration
- shot type
- camera motion
- subject motion
- editing characteristics
- transitions
- description

SHOT TYPES may include:

- extreme_close_up
- close_up
- medium_close_up
- medium
- medium_wide
- wide
- extreme_wide
- over_the_shoulder
- two_shot
- insert
- establishing_shot
- unknown

CAMERA MOTION may include:

- static
- pan_left
- pan_right
- tilt_up
- tilt_down
- zoom_in
- zoom_out
- dolly_in
- dolly_out
- tracking
- handheld
- orbit
- mixed
- unknown

SUBJECT MOTION should describe actual subject movement,
not camera movement.

EDITING CHARACTERISTICS may include:

- fast_paced
- slow_paced
- rhythmic
- match_cut
- jump_cut
- continuity
- montage
- beat_driven
- minimal_cutting
- rapid_cutting

TRANSITIONS may include:

- hard_cut
- fade_in
- fade_out
- dissolve
- wipe
- match_cut
- jump_cut
- none
- unknown

Be conservative.

If something cannot be reliably determined,
return "unknown".

The original uploaded reference video is the visual
source of truth.

PySceneDetect provides the temporal boundaries.

OpenCV motion scores are supporting evidence.

Here is the deterministic analysis:

{json.dumps(
    deterministic_data,
    indent=2
)}

Return ONLY valid JSON.
"""

    # --------------------------------------------------------
    # 8. GEMINI SEMANTIC ANALYSIS
    # --------------------------------------------------------

    print(
        "[REFERENCE] Sending reference video + "
        "analysis to Gemini..."
    )

    result = generate_json(

        client,

        [
            types.Part.from_uri(

                file_uri=uploaded_video.uri,

                mime_type=uploaded_video.mime_type,
            ),

            prompt,
        ],

        ReferenceAnalysis,
    )

    # --------------------------------------------------------
    # 9. PYDANTIC VALIDATION
    # --------------------------------------------------------

    print(
        "[REFERENCE] Validating Gemini result..."
    )

    validated = ReferenceAnalysis.model_validate(
        result
    )

    print(
        "[REFERENCE] Analysis complete."
    )

    return validated.model_dump()


# ============================================================
# TARGET VIDEO ANALYSIS
# ============================================================

def analyze_target_video(
    video_path: str,
    reference_video_path: str,
    workspace_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Analyze target.mp4.

    Pipeline:

        FFprobe
            ↓
        PySceneDetect
            ↓
        OpenCV representative frames
            ↓
        OpenCV motion analysis
            ↓
        Reference representative frames
            ↓
        Gemini semantic target analysis
            ↓
        Pydantic validation
            ↓
        target_analysis

    IMPORTANT:

    This function uses:

        GEMINI_API_KEY1

    It does NOT use GEMINI_API_KEY.
    """

    # --------------------------------------------------------
    # RESOLVE PATHS
    # --------------------------------------------------------

    video_path = str(
        Path(video_path).resolve()
    )

    reference_video_path = str(
        Path(reference_video_path).resolve()
    )

    # --------------------------------------------------------
    # CHECK TARGET VIDEO
    # --------------------------------------------------------

    if not Path(video_path).exists():

        raise FileNotFoundError(
            f"Target video does not exist: "
            f"{video_path}"
        )

    # --------------------------------------------------------
    # CHECK REFERENCE VIDEO
    # --------------------------------------------------------

    if not Path(reference_video_path).exists():

        raise FileNotFoundError(
            f"Reference video does not exist: "
            f"{reference_video_path}"
        )

    # ========================================================
    # TARGET ANALYSIS
    # ========================================================

    # --------------------------------------------------------
    # 1. TARGET FFPROBE
    # --------------------------------------------------------

    print(
        "\n[TARGET] Running FFprobe..."
    )

    metadata = ffprobe_metadata(
        video_path
    )

    # --------------------------------------------------------
    # 2. TARGET PYSCENEDETECT
    # --------------------------------------------------------

    print(
        "[TARGET] Running PySceneDetect..."
    )

    segments = detect_scenes(
        video_path
    )

    print(
        f"[TARGET] Detected "
        f"{len(segments)} segments."
    )

    # --------------------------------------------------------
    # 3. TARGET OPENCV
    # --------------------------------------------------------

    print(
        "[TARGET] Running OpenCV analysis..."
    )

    target_frame_dir = (
        Path(workspace_dir) / "target_frames"
        if workspace_dir is not None
        else Path("workspace/target_frames")
    )

    for segment in segments:

        # ----------------------------------------------------
        # REPRESENTATIVE FRAMES
        # ----------------------------------------------------

        frames = extract_representative_frames(

            video_path,

            str(target_frame_dir),

            segment["start_time"],

            segment["end_time"],

            samples=5,
        )

        # ----------------------------------------------------
        # MOTION ANALYSIS
        # ----------------------------------------------------

        motion = analyze_motion(

            video_path,

            segment["start_time"],

            segment["end_time"],
        )

        segment["frames"] = frames

        segment["motion_score"] = (
            motion["motion_score"]
        )

    # ========================================================
    # REFERENCE FRAME EXTRACTION
    # ========================================================

    print(
        "[TARGET] Extracting representative "
        "reference frames..."
    )

    reference_segments = detect_scenes(
        reference_video_path
    )

    reference_frame_dir = (
        Path(workspace_dir) / "reference_frames_for_target"
        if workspace_dir is not None
        else Path("workspace/reference_frames_for_target")
    )

    reference_frames = []

    for segment in reference_segments:

        frames = extract_representative_frames(

            reference_video_path,

            str(reference_frame_dir),

            segment["start_time"],

            segment["end_time"],

            samples=3,
        )

        reference_frames.extend(
            frames
        )

    # ========================================================
    # TARGET GEMINI CLIENT
    # ========================================================

    print(
        "[TARGET] Initializing Gemini using "
        "GEMINI_API_KEY1..."
    )

    # IMPORTANT:
    #
    # Target analysis explicitly uses GEMINI_API_KEY1.
    #
    # It cannot accidentally use GEMINI_API_KEY because
    # get_gemini_client() receives the exact environment
    # variable name.

    client = get_gemini_client(
        "GEMINI_API_KEY1"
    )

    # ========================================================
    # BUILD IMAGE PARTS
    # ========================================================

    image_parts = []

    # --------------------------------------------------------
    # TARGET FRAMES
    # --------------------------------------------------------

    print(
        "[TARGET] Preparing target frames..."
    )

    for segment in segments:

        for frame in segment["frames"]:

            frame_path = frame["path"]

            if not Path(frame_path).exists():
                continue

            with open(
                frame_path,
                "rb",
            ) as f:

                image_parts.append(

                    types.Part.from_bytes(

                        data=f.read(),

                        mime_type="image/jpeg",
                    )
                )

    # --------------------------------------------------------
    # REFERENCE FRAMES
    # --------------------------------------------------------

    print(
        "[TARGET] Preparing reference frames..."
    )

    for frame in reference_frames:

        frame_path = frame["path"]

        if not Path(frame_path).exists():
            continue

        with open(
            frame_path,
            "rb",
        ) as f:

            image_parts.append(

                types.Part.from_bytes(

                    data=f.read(),

                    mime_type="image/jpeg",
                )
            )

    # ========================================================
    # TARGET GEMINI PROMPT
    # ========================================================

    prompt = f"""
You are the target-video analysis agent.

Analyze target.mp4 based on its local computer-vision
analysis and representative frames.

The purpose of this analysis is to determine how the
target footage can be edited to reproduce the editing
language of the reference video.

IMPORTANT:

The reference video is STYLE evidence.

The target video is the SOURCE footage.

Do not invent target footage.

Analyze only what actually exists in the target video.

For every target segment determine:

- segment_id
- start_time
- end_time
- duration
- shot_type
- camera_motion
- subject_motion
- description
- motion_score

Use the reference frames as visual STYLE evidence.

Consider:

- shot composition
- shot scale
- visual rhythm
- pacing
- camera language
- subject movement
- visual progression
- cutting style

Do not change the temporal boundaries supplied by
PySceneDetect.

Target deterministic analysis:

{json.dumps(
    segments,
    indent=2
)}

Target metadata:

{json.dumps(
    metadata,
    indent=2
)}

Reference deterministic segments:

{json.dumps(
    reference_segments,
    indent=2
)}

Return ONLY valid JSON.
"""

    # ========================================================
    # GEMINI TARGET ANALYSIS
    # ========================================================

    print(
        "[TARGET] Sending target frames + "
        "reference frames to Gemini using GEMINI_API_KEY1..."
    )

    result = generate_json(

        client,

        image_parts + [
            prompt
        ],

        TargetAnalysis,
    )

    # ========================================================
    # PYDANTIC VALIDATION
    # ========================================================

    print(
        "[TARGET] Validating Gemini result..."
    )

    validated = TargetAnalysis.model_validate(
        result
    )

    print(
        "[TARGET] Analysis complete."
    )

    return validated.model_dump()