import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .analyzer import (
    analyze_reference_video,
    analyze_target_video,
)

from .llama_planner import (
    QwenEditPlanner,
)

from .editor import (
    execute_editspec,
)


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


REFERENCE_VIDEO = (
    DATA_DIR / "reference.mp4"
)

TARGET_VIDEO = (
    DATA_DIR / "target.mp4"
)


def save_json(
    path: Path,
    data: dict,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )


def run_pipeline():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # STEP 1
    # REFERENCE + TARGET ANALYSIS IN PARALLEL
    # ========================================================

    print(
        "\n[1/4] Starting parallel video analysis..."
    )

    with ThreadPoolExecutor(
        max_workers=2
    ) as executor:

        reference_future = (
            executor.submit(
                analyze_reference_video,
                str(REFERENCE_VIDEO),
            )
        )

        target_future = (
            executor.submit(
                analyze_target_video,
                str(TARGET_VIDEO),
                str(REFERENCE_VIDEO),
            )
        )

        reference_analysis = (
            reference_future.result()
        )

        target_analysis = (
            target_future.result()
        )

    # ========================================================
    # STEP 2
    # SAVE ANALYSIS
    # ========================================================

    print(
        "\n[2/4] Saving analysis JSON..."
    )

    reference_path = (
        OUTPUT_DIR /
        "reference_analysis.json"
    )

    target_path = (
        OUTPUT_DIR /
        "target_analysis.json"
    )

    save_json(
        reference_path,
        reference_analysis,
    )

    save_json(
        target_path,
        target_analysis,
    )

    # ========================================================
    # STEP 3
    # QWEN EDIT PLANNING
    # ========================================================

    print(
        "\n[3/4] Generating EditSpec with "
        "Qwen 8B model..."
    )

    planner = QwenEditPlanner()

    editspec = (
        planner.generate_editspec(
            reference_analysis,
            target_analysis,
        )
    )

    editspec_path = (
        OUTPUT_DIR /
        "editspec.json"
    )

    save_json(
        editspec_path,
        editspec,
    )

    # ========================================================
    # STEP 4
    # EXECUTE EDITSPEC
    # ========================================================

    print(
        "\n[4/4] Executing EditSpec..."
    )

    edited_video = (
        OUTPUT_DIR /
        "edited.mp4"
    )

    execute_editspec(
        str(editspec_path),
        str(edited_video),
    )

    print(
        "\n================================"
    )

    print(
        "VIDEO EDITING PIPELINE COMPLETE"
    )

    print(
        "================================"
    )

    print(
        f"Reference: {reference_path}"
    )

    print(
        f"Target:    {target_path}"
    )

    print(
        f"EditSpec:  {editspec_path}"
    )

    print(
        f"Output:    {edited_video}"
    )

    return {
        "reference_analysis": reference_path,
        "target_analysis": target_path,
        "editspec": editspec_path,
        "edited_video": edited_video,
    }