import json
import subprocess
from pathlib import Path

from .schemas import EditSpec


def execute_editspec(
    editspec_path: str,
    output_path: str | None = None,
):

    with open(
        editspec_path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    spec = EditSpec.model_validate(
        data
    )

    source_video = spec.source_video

    if output_path is None:

        output_path = (
            spec.output_video
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    segment_files = []

    work_dir = Path(
        "workspace/rendered_segments"
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # RENDER INDIVIDUAL SEGMENTS
    # --------------------------------------------------------

    for index, segment in enumerate(
        spec.segments
    ):

        segment_path = (
            work_dir /
            f"segment_{index:04d}.mp4"
        )

        duration = (
            segment.source_end -
            segment.source_start
        )

        video_filters = []

        audio_filters = []

        # ----------------------------------------------------
        # OPERATIONS
        # ----------------------------------------------------

        for operation in (
            segment.operations
        ):

            op = operation.operation
            params = operation.parameters

            if op == "scale":

                width = params[
                    "width"
                ]

                height = params[
                    "height"
                ]

                video_filters.append(
                    f"scale={width}:{height}"
                )

            elif op == "crop":

                width = params[
                    "width"
                ]

                height = params[
                    "height"
                ]

                x = params.get(
                    "x",
                    "(iw-ow)/2",
                )

                y = params.get(
                    "y",
                    "(ih-oh)/2",
                )

                video_filters.append(
                    f"crop={width}:{height}:{x}:{y}"
                )

            elif op == "speed":

                speed = float(
                    params["factor"]
                )

                video_filters.append(
                    f"setpts={1/speed}*PTS"
                )

                audio_filters.append(
                    f"atempo={speed}"
                )

            elif op == "fade_in":

                duration_fade = float(
                    params.get(
                        "duration",
                        0.3,
                    )
                )

                video_filters.append(
                    "fade=t=in:"
                    f"st=0:"
                    f"d={duration_fade}"
                )

            elif op == "fade_out":

                duration_fade = float(
                    params.get(
                        "duration",
                        0.3,
                    )
                )

                start = max(
                    duration -
                    duration_fade,
                    0,
                )

                video_filters.append(
                    "fade=t=out:"
                    f"st={start}:"
                    f"d={duration_fade}"
                )

            elif op == "mute":

                audio_filters.append(
                    "volume=0"
                )

        # ----------------------------------------------------
        # BUILD COMMAND
        # ----------------------------------------------------

        command = [
            "ffmpeg",
            "-y",

            "-ss",
            str(segment.source_start),

            "-i",
            source_video,

            "-t",
            str(duration),
        ]

        if video_filters:

            command += [
                "-vf",
                ",".join(
                    video_filters
                ),
            ]

        if audio_filters:

            command += [
                "-af",
                ",".join(
                    audio_filters
                ),
            ]

        command += [
            "-c:v",
            "libx264",

            "-preset",
            "medium",

            "-crf",
            "18",

            "-c:a",
            "aac",

            str(segment_path),
        ]

        subprocess.run(
            command,
            check=True,
        )

        segment_files.append(
            segment_path
        )

    # --------------------------------------------------------
    # CONCAT
    # --------------------------------------------------------

    concat_file = (
        work_dir /
        "concat.txt"
    )

    with open(
        concat_file,
        "w",
        encoding="utf-8",
    ) as f:

        for path in segment_files:

            f.write(
                f"file '{path.resolve()}'\n"
            )

    command = [
        "ffmpeg",
        "-y",

        "-f",
        "concat",

        "-safe",
        "0",

        "-i",
        str(concat_file),

        "-c:v",
        "libx264",

        "-preset",
        "medium",

        "-crf",
        "18",

        "-c:a",
        "aac",

        str(output_path),
    ]

    subprocess.run(
        command,
        check=True,
    )

    return str(
        output_path
    )