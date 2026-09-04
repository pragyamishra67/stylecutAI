import argparse
from app.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="StyleCut AI Video Editing Pipeline")
    parser.add_argument(
        "--project-id",
        type=str,
        default=None,
        help="Optional unique project ID (default: auto-generated timestamped ID)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to reference video (default: data/reference.mp4)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Path to target video (default: data/target.mp4)",
    )
    parser.add_argument(
        "--no-supabase",
        action="store_true",
        help="Disable syncing to Supabase Storage",
    )

    args = parser.parse_args()

    run_pipeline(
        project_id=args.project_id,
        reference_video=args.reference,
        target_video=args.target,
        upload_to_supabase=not args.no_supabase,
    )


if __name__ == "__main__":
    main()