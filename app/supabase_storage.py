import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


class SupabaseStorage:
    """
    Persistent cloud storage manager for StyleCut AI using Supabase Storage.
    
    Provides isolated namespaces for each video-editing project:
        projects/<project_id>/
            reference/
            target/
            reference_frames/
            reference_frames_for_target/
            target_frames/
            analyses/
            editspec/
            output/
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        bucket: Optional[str] = None,
    ):
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
        self.bucket_name = bucket or os.getenv("SUPABASE_BUCKET", "stylecut")
        self._client = None
        self._bucket_verified = False

    @property
    def is_configured(self) -> bool:
        """Check if Supabase credentials are configured in the environment."""
        return bool(self.url and self.key)

    @property
    def client(self):
        """Lazy-initialize and return the Supabase client."""
        if not self.is_configured:
            return None

        if self._client is None:
            try:
                from supabase import create_client
                self._client = create_client(self.url, self.key)
            except Exception as e:
                print(f"[SUPABASE] Warning: Failed to initialize Supabase client: {e}")
                self._client = None

        return self._client

    def ensure_bucket_exists(self) -> bool:
        """Verify or create the designated storage bucket."""
        if not self.is_configured or self.client is None:
            return False

        if self._bucket_verified:
            return True

        try:
            buckets = self.client.storage.list_buckets()
            bucket_names = [b.name for b in buckets] if buckets else []
            if self.bucket_name not in bucket_names:
                print(f"[SUPABASE] Creating bucket '{self.bucket_name}' (private)...")
                self.client.storage.create_bucket(
                    self.bucket_name,
                    options={"public": False}
                )
            self._bucket_verified = True
            return True
        except Exception as e:
            # If bucket exists or permission error on list_buckets, assume bucket exists
            print(f"[SUPABASE] Bucket check notice: {e}")
            self._bucket_verified = True
            return True

    # ========================================================
    # FILE OPERATIONS
    # ========================================================

    def upload_file(
        self,
        local_path: str | Path,
        remote_path: str,
        content_type: Optional[str] = None,
        upsert: bool = True,
    ) -> Optional[str]:
        """
        Upload a local file to Supabase Storage.
        Returns the remote path upon success, or None if skipped/failed.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        if not self.is_configured or self.client is None:
            print(f"[SUPABASE] Storage not configured. Skipping upload for {remote_path}")
            return None

        self.ensure_bucket_exists()

        if content_type is None:
            content_type, _ = mimetypes.guess_type(str(local_path))
            if content_type is None:
                content_type = "application/octet-stream"

        remote_path = remote_path.replace("\\", "/").lstrip("/")

        print(f"[SUPABASE] Uploading {local_path.name} -> {self.bucket_name}/{remote_path} ({content_type})")

        with open(local_path, "rb") as f:
            file_bytes = f.read()

        file_options = {
            "content-type": content_type,
            "upsert": "true" if upsert else "false",
        }

        try:
            self.client.storage.from_(self.bucket_name).upload(
                path=remote_path,
                file=file_bytes,
                file_options=file_options,
            )
            return remote_path
        except Exception as e:
            error_str = str(e)
            if "Duplicate" in error_str or "already exists" in error_str:
                # Update if already exists and upsert is desired
                try:
                    self.client.storage.from_(self.bucket_name).update(
                        path=remote_path,
                        file=file_bytes,
                        file_options=file_options,
                    )
                    return remote_path
                except Exception as update_err:
                    print(f"[SUPABASE] Error updating existing file {remote_path}: {update_err}")
                    raise update_err
            print(f"[SUPABASE] Upload failed for {remote_path}: {e}")
            raise e

    def download_file(
        self,
        remote_path: str,
        local_path: str | Path,
    ) -> Path:
        """Download a file from Supabase Storage to a local path."""
        if not self.is_configured or self.client is None:
            raise RuntimeError("Supabase credentials not configured.")

        remote_path = remote_path.replace("\\", "/").lstrip("/")
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[SUPABASE] Downloading {self.bucket_name}/{remote_path} -> {local_path}")

        res = self.client.storage.from_(self.bucket_name).download(remote_path)
        with open(local_path, "wb") as f:
            f.write(res)

        return local_path

    def upload_json(
        self,
        data: Dict[str, Any] | List[Any],
        remote_path: str,
    ) -> Optional[str]:
        """Upload a dictionary/list directly as a JSON file."""
        if not self.is_configured or self.client is None:
            return None

        self.ensure_bucket_exists()
        remote_path = remote_path.replace("\\", "/").lstrip("/")
        json_bytes = json.dumps(data, indent=2).encode("utf-8")

        file_options = {
            "content-type": "application/json",
            "upsert": "true",
        }

        try:
            self.client.storage.from_(self.bucket_name).upload(
                path=remote_path,
                file=json_bytes,
                file_options=file_options,
            )
            return remote_path
        except Exception as e:
            if "Duplicate" in str(e) or "already exists" in str(e):
                self.client.storage.from_(self.bucket_name).update(
                    path=remote_path,
                    file=json_bytes,
                    file_options=file_options,
                )
                return remote_path
            raise e

    def download_json(self, remote_path: str) -> Dict[str, Any]:
        """Download and parse a JSON file from Supabase Storage."""
        if not self.is_configured or self.client is None:
            raise RuntimeError("Supabase credentials not configured.")

        remote_path = remote_path.replace("\\", "/").lstrip("/")
        res = self.client.storage.from_(self.bucket_name).download(remote_path)
        return json.loads(res.decode("utf-8"))

    def upload_directory(
        self,
        local_dir: str | Path,
        remote_prefix: str,
    ) -> List[str]:
        """
        Recursively upload all files within a local directory into a remote prefix.
        Useful for uploading extracted frame batches.
        """
        local_dir = Path(local_dir)
        if not local_dir.exists() or not local_dir.is_dir():
            return []

        uploaded = []
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(local_dir).as_posix()
                target_remote = f"{remote_prefix.rstrip('/')}/{rel_path}"
                res = self.upload_file(file_path, target_remote)
                if res:
                    uploaded.append(res)

        return uploaded

    def create_signed_url(
        self,
        remote_path: str,
        expires_in: int = 3600,
    ) -> Optional[str]:
        """Create a time-limited signed URL for private asset access."""
        if not self.is_configured or self.client is None:
            return None

        remote_path = remote_path.replace("\\", "/").lstrip("/")
        try:
            res = self.client.storage.from_(self.bucket_name).create_signed_url(
                path=remote_path,
                expires_in=expires_in,
            )
            if isinstance(res, dict) and "signedURL" in res:
                return res["signedURL"]
            elif isinstance(res, dict) and "signedUrl" in res:
                return res["signedUrl"]
            elif hasattr(res, "signed_url"):
                return res.signed_url
            return str(res)
        except Exception as e:
            print(f"[SUPABASE] Notice: Could not generate signed URL for {remote_path}: {e}")
            return None

    def list_project_files(self, project_id: str) -> List[Dict[str, Any]]:
        """List all files stored under projects/<project_id>/."""
        if not self.is_configured or self.client is None:
            return []

        prefix = f"projects/{project_id}"
        try:
            return self.client.storage.from_(self.bucket_name).list(prefix)
        except Exception as e:
            print(f"[SUPABASE] Failed listing files for project {project_id}: {e}")
            return []

    # ========================================================
    # PROJECT METADATA & PERSISTENCE
    # ========================================================

    def save_project_metadata(
        self,
        project_id: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Persist project manifest to storage as metadata.json and optionally
        to a 'projects' Supabase table if it exists.
        """
        # Always persist metadata.json in project root in Storage
        metadata_remote_path = f"projects/{project_id}/metadata.json"
        try:
            self.upload_json(metadata, metadata_remote_path)
        except Exception as e:
            print(f"[SUPABASE] Warning: Failed saving metadata.json in storage: {e}")

        # If DB is configured, try recording in `projects` table
        if self.is_configured and self.client is not None:
            try:
                db_record = {
                    "id": project_id,
                    "status": metadata.get("status", "completed"),
                    "reference_path": metadata.get("reference_path"),
                    "target_path": metadata.get("target_path"),
                    "reference_analysis_path": metadata.get("reference_analysis_path"),
                    "target_analysis_path": metadata.get("target_analysis_path"),
                    "editspec_path": metadata.get("editspec_path"),
                    "output_path": metadata.get("output_path"),
                    "metadata": metadata,
                }
                self.client.table("projects").upsert(db_record).execute()
                print(f"[SUPABASE] Project {project_id} recorded in 'projects' table.")
            except Exception as db_err:
                # Database table may not exist yet; storage upload was still successful
                pass

        return True

    # ========================================================
    # PATH HELPERS
    # ========================================================

    @staticmethod
    def get_reference_remote_path(project_id: str, filename: str = "reference.mp4") -> str:
        return f"projects/{project_id}/reference/{filename}"

    @staticmethod
    def get_target_remote_path(project_id: str, filename: str = "target.mp4") -> str:
        return f"projects/{project_id}/target/{filename}"

    @staticmethod
    def get_reference_frames_remote_prefix(project_id: str) -> str:
        return f"projects/{project_id}/reference_frames"

    @staticmethod
    def get_reference_frames_for_target_remote_prefix(project_id: str) -> str:
        return f"projects/{project_id}/reference_frames_for_target"

    @staticmethod
    def get_target_frames_remote_prefix(project_id: str) -> str:
        return f"projects/{project_id}/target_frames"

    @staticmethod
    def get_reference_analysis_remote_path(project_id: str) -> str:
        return f"projects/{project_id}/analyses/reference_analysis.json"

    @staticmethod
    def get_target_analysis_remote_path(project_id: str) -> str:
        return f"projects/{project_id}/analyses/target_analysis.json"

    @staticmethod
    def get_editspec_remote_path(project_id: str) -> str:
        return f"projects/{project_id}/editspec/editspec.json"

    @staticmethod
    def get_output_remote_path(project_id: str, filename: str = "edited.mp4") -> str:
        return f"projects/{project_id}/output/{filename}"
