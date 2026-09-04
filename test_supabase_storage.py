import os
import unittest
from pathlib import Path

from app.supabase_storage import SupabaseStorage
from app.pipeline import generate_project_id, run_pipeline
from app.analyzer import analyze_reference_video, analyze_target_video
import inspect


class TestSupabaseStorageIntegration(unittest.TestCase):

    def test_project_id_generation(self):
        id1 = generate_project_id()
        id2 = generate_project_id()
        self.assertNotEqual(id1, id2)
        self.assertTrue(id1.startswith("project_"))
        self.assertTrue(id2.startswith("project_"))
        print(f"[TEST] Generated project IDs: {id1}, {id2}")

    def test_storage_path_helpers(self):
        project_id = "test_project_123"
        storage = SupabaseStorage()

        ref_path = storage.get_reference_remote_path(project_id, "ref.mp4")
        self.assertEqual(ref_path, "projects/test_project_123/reference/ref.mp4")

        tgt_path = storage.get_target_remote_path(project_id, "tgt.mp4")
        self.assertEqual(tgt_path, "projects/test_project_123/target/tgt.mp4")

        ref_frames = storage.get_reference_frames_remote_prefix(project_id)
        self.assertEqual(ref_frames, "projects/test_project_123/reference_frames")

        ref_tgt_frames = storage.get_reference_frames_for_target_remote_prefix(project_id)
        self.assertEqual(ref_tgt_frames, "projects/test_project_123/reference_frames_for_target")

        tgt_frames = storage.get_target_frames_remote_prefix(project_id)
        self.assertEqual(tgt_frames, "projects/test_project_123/target_frames")

        ref_analysis = storage.get_reference_analysis_remote_path(project_id)
        self.assertEqual(ref_analysis, "projects/test_project_123/analyses/reference_analysis.json")

        tgt_analysis = storage.get_target_analysis_remote_path(project_id)
        self.assertEqual(tgt_analysis, "projects/test_project_123/analyses/target_analysis.json")

        editspec = storage.get_editspec_remote_path(project_id)
        self.assertEqual(editspec, "projects/test_project_123/editspec/editspec.json")

        output = storage.get_output_remote_path(project_id, "edited.mp4")
        self.assertEqual(output, "projects/test_project_123/output/edited.mp4")

        print("[TEST] All storage path helpers verified successfully.")

    def test_analyzer_function_signatures(self):
        # Verify backward compatibility: workspace_dir is optional
        ref_sig = inspect.signature(analyze_reference_video)
        self.assertIn("workspace_dir", ref_sig.parameters)
        self.assertIsNone(ref_sig.parameters["workspace_dir"].default)

        tgt_sig = inspect.signature(analyze_target_video)
        self.assertIn("workspace_dir", tgt_sig.parameters)
        self.assertIsNone(tgt_sig.parameters["workspace_dir"].default)

        print("[TEST] Analyzer backward-compatible signatures verified.")

    def test_storage_fallback_without_credentials(self):
        # Storage initialized without env credentials should not crash
        mock_storage = SupabaseStorage(url="", key="")
        self.assertFalse(mock_storage.is_configured)
        self.assertIsNone(mock_storage.client)
        print("[TEST] Graceful storage fallback verified.")


if __name__ == "__main__":
    unittest.main()
