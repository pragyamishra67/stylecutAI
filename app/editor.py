import json
from pathlib import Path

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips,
)

from .schemas import EditSpec
from .video_utils import ffprobe_metadata


# ============================================================
# MOVIEPY EDITSPEC EXECUTOR
# ============================================================

def execute_editspec(
    editspec_path: str,
    output_path: str | None = None,
):
    """
    Execute an EditSpec using MoviePy.

    Workflow:

        editspec.json
             |
             v
        target.mp4
             |
             v
        Apply segment operations
             |
             v
        Concatenate edited segments
             |
             v
        Replace target audio with
        reference.mp4 audio
             |
             v
        edited.mp4

    Public API intentionally remains:

        execute_editspec(
            editspec_path,
            output_path=None
        )
    """

    # ========================================================
    # LOAD EDITSPEC
    # ========================================================

    editspec_path = Path(
        editspec_path
    )

    if not editspec_path.exists():
        raise FileNotFoundError(
            f"EditSpec file not found: {editspec_path}"
        )

    with open(
        editspec_path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    # ========================================================
    # VALIDATE EDITSPEC
    # ========================================================

    spec = EditSpec.model_validate(
        data
    )

    # ========================================================
    # PATHS
    # ========================================================

    source_video = Path(
        spec.source_video
    )

    reference_video = Path(
        spec.reference_video
    )

    if output_path is None:

        output_path = spec.output_video

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    if not source_video.exists():

        raise FileNotFoundError(
            f"Target video not found: "
            f"{source_video}"
        )

    if not reference_video.exists():

        raise FileNotFoundError(
            f"Reference video not found: "
            f"{reference_video}"
        )

    # ========================================================
    # WORKSPACE
    # ========================================================

    work_dir = Path(
        "workspace/moviepy"
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # OPEN TARGET VIDEO
    # ========================================================

    print(
        f"[EDITOR] Loading target video: "
        f"{source_video}"
    )

    target_clip = VideoFileClip(
        str(source_video)
    )

    # ========================================================
    # REFERENCE VIDEO MASTER TIMELINE
    # ========================================================

    ref_metadata = ffprobe_metadata(
        str(reference_video)
    )

    reference_fps = ref_metadata.get(
        "fps"
    )

    reference_frame_count = ref_metadata.get(
        "frame_count"
    )

    reference_duration = (
        reference_frame_count / reference_fps
        if reference_fps and reference_frame_count
        else float(
            ref_metadata.get("duration")
            or ref_metadata.get("duration_seconds", 0.0)
        )
    )

    print(
        f"[EDITOR] Reference master timeline: "
        f"FPS={reference_fps}, "
        f"Frames={reference_frame_count}, "
        f"Duration={reference_duration:.6f}s"
    )

    # Reference FPS takes precedence as authoritative output FPS
    output_fps = (
        reference_fps
        if reference_fps
        else (
            target_clip.fps
            if target_clip.fps
            else 30.0
        )
    )

    # ========================================================
    # PROCESS SEGMENTS
    # ========================================================

    edited_segments = []

    try:

        for index, segment in enumerate(
            spec.segments
        ):

            print(
                f"\n[EDITOR] Processing segment "
                f"{index + 1}/{len(spec.segments)}"
            )

            print(
                f"[EDITOR] Source segment: "
                f"{segment.source_segment_id}"
            )

            print(
                f"[EDITOR] Source range: "
                f"{segment.source_start:.3f} -> "
                f"{segment.source_end:.3f}"
            )

            # ------------------------------------------------
            # Validate source range
            # ------------------------------------------------

            if segment.source_start < 0:

                raise ValueError(
                    f"source_start cannot be negative: "
                    f"{segment.source_start}"
                )

            if segment.source_end <= segment.source_start:

                raise ValueError(
                    f"Invalid source range for "
                    f"{segment.source_segment_id}: "
                    f"{segment.source_start} -> "
                    f"{segment.source_end}"
                )

            if segment.source_end > target_clip.duration:

                raise ValueError(
                    f"Source segment "
                    f"{segment.source_segment_id} "
                    f"ends at {segment.source_end:.3f}s, "
                    f"but target video is only "
                    f"{target_clip.duration:.3f}s long."
                )

            # ------------------------------------------------
            # Extract target footage
            # ------------------------------------------------

            clip = target_clip.subclipped(
                segment.source_start,
                segment.source_end,
            )

            # ------------------------------------------------
            # Apply operations
            # ------------------------------------------------

            clip = _apply_operations(
                clip,
                segment.operations,
            )

            # ------------------------------------------------
            # Store processed clip
            # ------------------------------------------------

            edited_segments.append(
                clip
            )

            print(
                f"[EDITOR] Final segment duration: "
                f"{clip.duration:.3f}s"
            )

        # ====================================================
        # CHECK SEGMENTS
        # ====================================================

        if not edited_segments:

            raise ValueError(
                "EditSpec contains no segments."
            )

        # ====================================================
        # CONCATENATE VIDEO
        # ====================================================

        print(
            "\n[EDITOR] Concatenating edited segments..."
        )

        final_video = concatenate_videoclips(
            edited_segments,
            method="compose",
        )

        print(
            f"[EDITOR] Final video duration before timeline enforcement: "
            f"{final_video.duration:.6f}s"
        )

        # ====================================================
        # ENFORCE EXACT REFERENCE MASTER TIMELINE & FRAME COUNT
        # ====================================================

        if reference_frame_count and output_fps:

            exact_duration = (
                reference_frame_count / output_fps
            )

            print(
                f"[EDITOR] Enforcing reference master timeline: "
                f"{reference_frame_count} frames @ {output_fps} fps -> {exact_duration:.6f}s"
            )

            if final_video.duration > exact_duration:

                print(
                    f"[EDITOR] Final video ({final_video.duration:.6f}s) is longer "
                    f"than reference timeline ({exact_duration:.6f}s). "
                    f"Trimming to exact reference timeline."
                )

                final_video = final_video.subclipped(
                    0,
                    exact_duration,
                )

            elif final_video.duration < exact_duration:

                deficit = (
                    exact_duration - final_video.duration
                )

                print(
                    f"[EDITOR] Final video ({final_video.duration:.6f}s) is shorter "
                    f"than reference timeline ({exact_duration:.6f}s) by {deficit:.6f}s."
                )

                # Reach exact reference frame count using existing target footage
                if edited_segments and len(spec.segments) > 0:

                    last_seg = spec.segments[-1]
                    needed_source = deficit

                    for op in last_seg.operations:
                        if op.operation == "speed":
                            factor = float(
                                op.parameters.get("factor", 1.0)
                            )
                            needed_source = deficit * factor

                    if last_seg.source_end + needed_source <= target_clip.duration:

                        print(
                            f"[EDITOR] Extending final segment from target video to reach "
                            f"exact reference timeline..."
                        )

                        extra_subclip = target_clip.subclipped(
                            last_seg.source_end,
                            last_seg.source_end + needed_source,
                        )

                        extra_ops = [
                            op for op in last_seg.operations
                            if op.operation != "trim"
                        ]

                        extra_subclip = _apply_operations(
                            extra_subclip,
                            extra_ops,
                        )

                        edited_segments.append(
                            extra_subclip
                        )

                        final_video = concatenate_videoclips(
                            edited_segments,
                            method="compose",
                        )

                        final_video = final_video.subclipped(
                            0,
                            exact_duration,
                        )

                    else:

                        error_msg = (
                            f"[EDITOR] ERROR: Output frame count does not match reference.\n"
                            f"Generated edit ({final_video.duration:.6f}s) is shorter than reference "
                            f"timeline ({exact_duration:.6f}s) and insufficient target footage remains."
                        )
                        print(error_msg)
                        raise ValueError(error_msg)

                else:

                    error_msg = (
                        f"[EDITOR] ERROR: Output frame count does not match reference.\n"
                        f"Generated edit ({final_video.duration:.6f}s) is shorter than reference "
                        f"timeline ({exact_duration:.6f}s)."
                    )
                    print(error_msg)
                    raise ValueError(error_msg)

            # Prevent float underflow during MoviePy frame iteration
            final_video = final_video.with_duration(
                (reference_frame_count + 1e-5) / output_fps
            )

        # ====================================================
        # LOAD REFERENCE AUDIO
        # ====================================================

        print(
            f"[EDITOR] Loading reference audio from: "
            f"{reference_video}"
        )

        reference_clip = AudioFileClip(
            str(reference_video)
        )

        # ====================================================
        # APPLY REFERENCE AUDIO
        # ====================================================

        final_video = _attach_reference_audio(
            final_video,
            reference_clip,
        )

        # ====================================================
        # WRITE FINAL VIDEO
        # ====================================================

        print(
            f"\n[EDITOR] Rendering final video at {output_fps} FPS:"
            f"\n          {output_path}"
        )

        final_video.write_videofile(

            str(output_path),

            codec="libx264",

            audio_codec="aac",

            preset="medium",

            fps=output_fps,

            threads=4,

            logger="bar",
        )

        # ====================================================
        # FINAL VALIDATION (EXACT FPS & FRAME COUNT)
        # ====================================================

        print(
            "\n[EDITOR] Running final FFprobe validation on rendered output..."
        )

        out_meta = ffprobe_metadata(
            str(output_path)
        )

        out_fps = out_meta.get("fps")
        out_frame_count = out_meta.get("frame_count")
        out_duration = float(
            out_meta.get("duration")
            or out_meta.get("duration_seconds", 0.0)
        )

        print(
            f"[EDITOR] Reference master timeline: "
            f"FPS={reference_fps}, "
            f"Frames={reference_frame_count}, "
            f"Duration={reference_duration:.6f}s"
        )
        print(
            f"[EDITOR] Rendered output video:     "
            f"FPS={out_fps}, "
            f"Frames={out_frame_count}, "
            f"Duration={out_duration:.6f}s"
        )

        # Validate FPS
        if reference_fps is not None and abs(out_fps - reference_fps) > 0.01:

            error_msg = (
                f"[EDITOR] ERROR: Output FPS does not match reference.\n"
                f"Reference: {reference_fps}\n"
                f"Output: {out_fps}"
            )
            print(error_msg)
            raise ValueError(error_msg)

        # Validate Frame Count (Fail loudly if not exact)
        if reference_frame_count is not None and out_frame_count != reference_frame_count:

            error_msg = (
                f"[EDITOR] ERROR: Output frame count does not match reference.\n"
                f"Reference: {reference_frame_count}\n"
                f"Output: {out_frame_count}"
            )
            print(error_msg)
            raise ValueError(error_msg)

        print(
            f"[EDITOR] SUCCESS: Output matches exact reference timeline!\n"
            f"          FPS: {out_fps} == {reference_fps}\n"
            f"          Frames: {out_frame_count} == {reference_frame_count}\n"
            f"          Duration: {out_duration:.6f}s"
        )

        print(
            "\n[EDITOR] Editing completed successfully."
        )

        return str(
            output_path
        )

    finally:

        # ====================================================
        # CLEANUP
        # ====================================================

        print(
            "[EDITOR] Cleaning up MoviePy resources..."
        )

        try:
            target_clip.close()
        except Exception:
            pass

        for clip in edited_segments:

            try:
                clip.close()
            except Exception:
                pass

        try:
            final_video.close()
        except Exception:
            pass

        try:
            reference_clip.close()
        except Exception:
            pass


# ============================================================
# APPLY EDIT OPERATIONS
# ============================================================

def _apply_operations(
    clip,
    operations,
):
    """
    Apply all EditSpec operations to one MoviePy clip.
    """

    for operation in operations:

        op = operation.operation
        params = operation.parameters

        print(
            f"[EDITOR] Applying operation: "
            f"{op}"
        )

        # ====================================================
        # TRIM
        # ====================================================

        if op == "trim":

            start = float(
                params.get(
                    "start",
                    0,
                )
            )

            end = params.get(
                "end"
            )

            if end is not None:

                end = float(end)

                clip = clip.subclipped(
                    start,
                    end,
                )

            else:

                clip = clip.subclipped(
                    start
                )

        # ====================================================
        # SPEED
        # ====================================================

        elif op == "speed":

            factor = float(
                params["factor"]
            )

            if factor <= 0:

                raise ValueError(
                    "Speed factor must be greater "
                    "than 0."
                )

            # MoviePy v2
            clip = clip.with_effects(
                [
                    __import__(
                        "moviepy"
                    ).video.fx.MultiplySpeed(
                        factor=factor
                    )
                ]
            )

        # ====================================================
        # SCALE
        # ====================================================

        elif op == "scale":

            width = params.get(
                "width"
            )

            height = params.get(
                "height"
            )

            if width is None and height is None:

                raise ValueError(
                    "scale requires width or height."
                )

            if width is not None:

                width = int(width)

            if height is not None:

                height = int(height)

            if width is not None and height is not None:

                clip = clip.resized(
                    new_size=(
                        width,
                        height,
                    )
                )

            elif width is not None:

                clip = clip.resized(
                    width=width
                )

            else:

                clip = clip.resized(
                    height=height
                )

        # ====================================================
        # CROP
        # ====================================================

        elif op == "crop":

            width = params.get(
                "width"
            )

            height = params.get(
                "height"
            )

            if width is None or height is None:

                raise ValueError(
                    "crop requires width and height."
                )

            width = int(width)
            height = int(height)

            x = params.get(
                "x"
            )

            y = params.get(
                "y"
            )

            # ------------------------------------------------
            # Center crop if x/y aren't specified
            # ------------------------------------------------

            if x is None:

                x = (
                    clip.w - width
                ) / 2

            if y is None:

                y = (
                    clip.h - height
                ) / 2

            x = int(x)
            y = int(y)

            clip = clip.cropped(
                x1=x,
                y1=y,
                x2=x + width,
                y2=y + height,
            )

        # ====================================================
        # FADE IN
        # ====================================================

        elif op == "fade_in":

            from moviepy.video.fx import FadeIn

            duration = float(
                params.get(
                    "duration",
                    0.3,
                )
            )

            clip = clip.with_effects(
                [
                    FadeIn(
                        duration
                    )
                ]
            )

        # ====================================================
        # FADE OUT
        # ====================================================

        elif op == "fade_out":

            from moviepy.video.fx import FadeOut

            duration = float(
                params.get(
                    "duration",
                    0.3,
                )
            )

            clip = clip.with_effects(
                [
                    FadeOut(
                        duration
                    )
                ]
            )

        # ====================================================
        # VOLUME
        # ====================================================

        elif op == "volume":

            factor = float(
                params.get(
                    "factor",
                    params.get(
                        "volume",
                        1.0,
                    ),
                )
            )

            if clip.audio is not None:

                from moviepy.audio.fx import MultiplyVolume

                clip = clip.with_effects(
                    [
                        MultiplyVolume(
                            factor
                        )
                    ]
                )

        # ====================================================
        # MUTE
        # ====================================================

        elif op == "mute":

            clip = clip.without_audio()

        # ====================================================
        # UNKNOWN OPERATION
        # ====================================================

        else:

            raise ValueError(
                f"Unsupported EditSpec operation: "
                f"{op}"
            )

    return clip


# ============================================================
# ATTACH REFERENCE AUDIO
# ============================================================

def _attach_reference_audio(
    video_clip,
    reference_audio,
):
    """
    Replace the target video's audio with the reference
    video's audio.

    If the reference audio is longer than the edited video,
    it is trimmed to the edited video's duration.

    If the reference audio is shorter than the edited video,
    it is NOT looped. The remaining video portion will have
    no audio.

    This follows the requirement that reference audio should
    not be artificially looped or extended.
    """

    video_duration = (
        video_clip.duration
    )

    audio_duration = (
        reference_audio.duration
    )

    print(
        f"[EDITOR] Video duration: "
        f"{video_duration:.3f}s"
    )

    print(
        f"[EDITOR] Reference audio duration: "
        f"{audio_duration:.3f}s"
    )

    # ========================================================
    # REFERENCE AUDIO LONGER THAN VIDEO
    # ========================================================

    if audio_duration > video_duration:

        print(
            "[EDITOR] Reference audio is longer "
            "than final video."
        )

        print(
            "[EDITOR] Trimming reference audio "
            "to video duration."
        )

        reference_audio = reference_audio.subclipped(
            0,
            video_duration,
        )

    # ========================================================
    # REFERENCE AUDIO SHORTER THAN VIDEO
    # ========================================================

    elif audio_duration < video_duration:

        print(
            "[EDITOR] WARNING: Reference audio is "
            "shorter than final video."
        )

        print(
            "[EDITOR] Audio will NOT be looped "
            "or extended."
        )

    # ========================================================
    # ATTACH AUDIO
    # ========================================================

    return video_clip.with_audio(
        reference_audio
    )


# ============================================================
# COMMAND LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Execute a MoviePy EditSpec."
        )
    )

    parser.add_argument(
        "editspec",
        help=(
            "Path to editspec.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output video path."
        ),
    )

    args = parser.parse_args()

    result = execute_editspec(
        editspec_path=args.editspec,
        output_path=args.output,
    )

    print(
        f"\nOutput video: {result}"
    )