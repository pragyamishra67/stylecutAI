import json
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .analyzer import (
    analyze_reference_video,
    analyze_target_video,
)

from .llama_planner import (
    LlamaEditPlanner,
)

from .editor import (
    execute_editspec,
)

from .supabase_storage import (
    SupabaseStorage,
)


DATA_DIR = Path("data")
ROOT_OUTPUT_DIR = Path("outputs")
ROOT_WORKSPACE_DIR = Path("workspace")

DEFAULT_REFERENCE_VIDEO = DATA_DIR / "reference.mp4"
DEFAULT_TARGET_VIDEO = DATA_DIR / "target.mp4"


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


def generate_project_id() -> str:
    """Generate a unique, collision-free, timestamped project identifier."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:8]
    return f"project_{timestamp}_{unique_suffix}"


def run_pipeline(
    project_id: Optional[str] = None,
    reference_video: Optional[Union[str, Path]] = None,
    target_video: Optional[Union[str, Path]] = None,
    upload_to_supabase: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete video editing pipeline with project isolation and
    persistent Supabase Storage integration.

    Parameters:
        project_id: Unique identifier for the editing job. Generated automatically if None.
        reference_video: Path to reference video. Defaults to data/reference.mp4.
        target_video: Path to target video. Defaults to data/target.mp4.
        upload_to_supabase: Whether to sync project artifacts to Supabase Storage.
    """

    # ========================================================
    # 0. INITIALIZE PROJECT & DIRECTORIES
    # ========================================================

    if not project_id:
        project_id = generate_project_id()

    print("\n" + "=" * 60)
    print(f"STARTING EDITING PIPELINE FOR PROJECT: {project_id}")
    print("=" * 60)

    # Local project-specific workspace and outputs
    project_workspace = ROOT_WORKSPACE_DIR / project_id
    project_workspace.mkdir(parents=True, exist_ok=True)

    project_output_dir = ROOT_OUTPUT_DIR / project_id
    project_output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve input video paths
    ref_video_path = Path(reference_video) if reference_video else DEFAULT_REFERENCE_VIDEO
    tgt_video_path = Path(target_video) if target_video else DEFAULT_TARGET_VIDEO

    if not ref_video_path.exists():
        raise FileNotFoundError(f"Reference video not found: {ref_video_path}")

    if not tgt_video_path.exists():
        raise FileNotFoundError(f"Target video not found: {tgt_video_path}")

    # Initialize Supabase storage layer
    storage = SupabaseStorage()
    if upload_to_supabase:
        if storage.is_configured:
            print(f"[STORAGE] Connected to Supabase bucket '{storage.bucket_name}'.")
            print(f"[STORAGE] Project namespace: projects/{project_id}/")
            storage.ensure_bucket_exists()
        else:
            print("[STORAGE] Supabase credentials not found in environment.")
            print("[STORAGE] Running in local-isolated mode (files saved under workspace/ & outputs/).")

    # ========================================================
    # 1. SYNC INPUT VIDEOS TO SUPABASE (IF CONFIGURED)
    # ========================================================

    remote_ref_path = None
    remote_tgt_path = None

    if upload_to_supabase and storage.is_configured:
        print("\n[STORAGE] Syncing input videos to Supabase Storage...")
        try:
            remote_ref_path = storage.upload_file(
                ref_video_path,
                storage.get_reference_remote_path(project_id, ref_video_path.name),
            )
            remote_tgt_path = storage.upload_file(
                tgt_video_path,
                storage.get_target_remote_path(project_id, tgt_video_path.name),
            )
        except Exception as e:
            print(f"[STORAGE] Warning: Error uploading input videos: {e}")

    # ========================================================
    # STEP 1: REFERENCE + TARGET ANALYSIS IN PARALLEL
    # ========================================================

    print("\n[1/4] Starting parallel video analysis...")

    with ThreadPoolExecutor(max_workers=2) as executor:
        reference_future = executor.submit(
            analyze_reference_video,
            str(ref_video_path),
            project_workspace,
        )

        target_future = executor.submit(
            analyze_target_video,
            str(tgt_video_path),
            str(ref_video_path),
            project_workspace,
        )

        reference_analysis = reference_future.result()
        target_analysis = target_future.result()

    # Upload extracted representative frames to Supabase
    if upload_to_supabase and storage.is_configured:
        print("\n[STORAGE] Syncing extracted frames to Supabase Storage...")
        try:
            ref_frames_dir = project_workspace / "reference_frames"
            if ref_frames_dir.exists():
                storage.upload_directory(
                    ref_frames_dir,
                    storage.get_reference_frames_remote_prefix(project_id),
                )

            tgt_frames_dir = project_workspace / "target_frames"
            if tgt_frames_dir.exists():
                storage.upload_directory(
                    tgt_frames_dir,
                    storage.get_target_frames_remote_prefix(project_id),
                )

            ref_target_frames_dir = project_workspace / "reference_frames_for_target"
            if ref_target_frames_dir.exists():
                storage.upload_directory(
                    ref_target_frames_dir,
                    storage.get_reference_frames_for_target_remote_prefix(project_id),
                )
        except Exception as e:
            print(f"[STORAGE] Warning: Error uploading frames: {e}")

    # ========================================================
    # STEP 2: SAVE ANALYSIS JSON
    # ========================================================

    print("\n[2/4] Saving analysis JSON...")

    reference_path = project_output_dir / "reference_analysis.json"
    target_path = project_output_dir / "target_analysis.json"

    save_json(reference_path, reference_analysis)
    save_json(target_path, target_analysis)

    # Also update legacy/root outputs folder for backward compatibility
    save_json(ROOT_OUTPUT_DIR / "reference_analysis.json", reference_analysis)
    save_json(ROOT_OUTPUT_DIR / "target_analysis.json", target_analysis)

    remote_ref_analysis_path = None
    remote_tgt_analysis_path = None

    if upload_to_supabase and storage.is_configured:
        print("[STORAGE] Syncing analyses to Supabase Storage...")
        try:
            remote_ref_analysis_path = storage.upload_file(
                reference_path,
                storage.get_reference_analysis_remote_path(project_id),
            )
            remote_tgt_analysis_path = storage.upload_file(
                target_path,
                storage.get_target_analysis_remote_path(project_id),
            )
        except Exception as e:
            print(f"[STORAGE] Warning: Error uploading analyses: {e}")

    # ========================================================
    # STEP 3: LLAMA EDIT PLANNING
    # ========================================================

    print("\n[3/4] Generating EditSpec with Llama model...")

    planner = LlamaEditPlanner()

    editspec = planner.generate_editspec(
        reference_analysis,
        target_analysis,
    )

    # Ensure source_video and reference_video point to the current project's inputs
    if isinstance(editspec, dict):
        editspec["source_video"] = str(tgt_video_path)
        editspec["reference_video"] = str(ref_video_path)

    editspec_path = project_output_dir / "editspec.json"
    save_json(editspec_path, editspec)

    # Also update legacy/root outputs folder
    save_json(ROOT_OUTPUT_DIR / "editspec.json", editspec)

    remote_editspec_path = None

    if upload_to_supabase and storage.is_configured:
        print("[STORAGE] Syncing EditSpec to Supabase Storage...")
        try:
            remote_editspec_path = storage.upload_file(
                editspec_path,
                storage.get_editspec_remote_path(project_id),
            )
        except Exception as e:
            print(f"[STORAGE] Warning: Error uploading EditSpec: {e}")

    # ========================================================
    # STEP 4: EXECUTE EDITSPEC
    # ========================================================

    print("\n[4/4] Executing EditSpec...")

    edited_video = project_output_dir / "edited.mp4"

    execute_editspec(
        str(editspec_path),
        str(edited_video),
    )

    # Keep a copy in root outputs for any external tooling checking outputs/edited.mp4
    try:
        shutil.copy2(edited_video, ROOT_OUTPUT_DIR / "edited.mp4")
    except Exception:
        pass

    remote_output_path = None
    signed_video_url = None

    if upload_to_supabase and storage.is_configured:
        print("\n[STORAGE] Syncing final edited video to Supabase Storage...")
        try:
            remote_output_path = storage.upload_file(
                edited_video,
                storage.get_output_remote_path(project_id, edited_video.name),
            )
            signed_video_url = storage.create_signed_url(
                remote_output_path,
                expires_in=86400,  # 24 hour URL
            )
        except Exception as e:
            print(f"[STORAGE] Warning: Error uploading edited video: {e}")

    # ========================================================
    # MANIFEST & PERSISTENCE
    # ========================================================

    manifest = {
        "project_id": project_id,
        "created_at": datetime.now().isoformat(),
        "status": "completed",
        "local": {
            "workspace": str(project_workspace),
            "reference_video": str(ref_video_path),
            "target_video": str(tgt_video_path),
            "reference_analysis": str(reference_path),
            "target_analysis": str(target_path),
            "editspec": str(editspec_path),
            "edited_video": str(edited_video),
        },
        "supabase": {
            "bucket": storage.bucket_name if storage.is_configured else None,
            "reference_path": remote_ref_path,
            "target_path": remote_tgt_path,
            "reference_analysis_path": remote_ref_analysis_path,
            "target_analysis_path": remote_tgt_analysis_path,
            "editspec_path": remote_editspec_path,
            "output_path": remote_output_path,
            "signed_video_url": signed_video_url,
        },
    }

    # Save manifest locally and to Supabase
    save_json(project_output_dir / "manifest.json", manifest)

    if upload_to_supabase and storage.is_configured:
        storage.save_project_metadata(project_id, manifest)

    print("\n" + "=" * 60)
    print("VIDEO EDITING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Project ID: {project_id}")
    print(f"Local Workspace: {project_workspace}")
    print(f"Local Output:    {edited_video}")
    if remote_output_path:
        print(f"Supabase Output: {storage.bucket_name}/{remote_output_path}")
    if signed_video_url:
        print(f"Signed Video URL: {signed_video_url}")

    return manifest