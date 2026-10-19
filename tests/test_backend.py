import os
import sys
import unittest
import io
import zipfile
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from fastapi.testclient import TestClient

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from webapp.main import app, agent_service, FRONTEND_DIST
from webapp.model_loader import download_model_checkpoint
from dqn import DQN


class TestBackend(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_01_api_health(self):
        """Test GET /api/health returns ok status."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("version"), "1.0.0")

    def test_02_model_status_endpoint(self):
        """Test GET /api/model-status reflects real checkpoint status."""
        response = self.client.get("/api/model-status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("model_loaded"))
        self.assertEqual(data.get("state_dim"), 180)
        self.assertEqual(data.get("action_dim"), 2)
        self.assertEqual(data.get("checkpoint_path"), "flappybirdv0.pt")
        self.assertAlmostEqual(data.get("best_recorded_reward"), 62.400000000000716, places=5)

    def test_03_checkpoint_and_model_architecture(self):
        """Verify the checkpoint state_dict directly loads into DQN(180, 2)."""
        model = DQN(state_dim=180, action_dim=2, hidden_dim=256)
        ckpt_path = os.path.join(PROJECT_ROOT, "runs", "flappybirdv0.pt")
        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint missing at {ckpt_path}")
        
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        dummy_obs = torch.randn(1, 180)
        with torch.inference_mode():
            q_values = model(dummy_obs)
        self.assertEqual(q_values.shape, (1, 2))

    def test_04_environment_and_rendering(self):
        """Verify headless rendering produces valid browser-renderable frame."""
        import gymnasium as gym
        import flappy_bird_gymnasium

        env = gym.make("FlappyBird-v0", render_mode="rgb_array")
        obs, info = env.reset()
        self.assertEqual(obs.shape, (180,))
        
        frame = env.render()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (512, 288, 3))
        self.assertEqual(frame.dtype, np.uint8)

        # Test frame encoding to JPEG base64
        encoded = agent_service._encode_frame(frame)
        self.assertTrue(encoded.startswith("data:image/jpeg;base64,"))
        self.assertGreater(len(encoded), 1000)

        # Perform 1 step
        next_obs, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(next_obs.shape, (180,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        env.close()

    def test_05_session_lifecycle(self):
        """Test Start, Pause, Resume, Reset, Stop endpoints."""
        # Start
        start_res = self.client.post("/api/session/start", json={"fps": 30})
        self.assertEqual(start_res.status_code, 200)
        self.assertTrue(start_res.json()["success"])
        self.assertEqual(agent_service._status, "running")

        # Pause
        pause_res = self.client.post("/api/session/pause")
        self.assertEqual(pause_res.status_code, 200)
        self.assertTrue(pause_res.json()["success"])
        self.assertEqual(agent_service._status, "paused")

        # Resume
        resume_res = self.client.post("/api/session/resume")
        self.assertEqual(resume_res.status_code, 200)
        self.assertTrue(resume_res.json()["success"])
        self.assertEqual(agent_service._status, "running")

        # Reset
        reset_res = self.client.post("/api/session/reset")
        self.assertEqual(reset_res.status_code, 200)
        self.assertTrue(reset_res.json()["success"])

        # Stop
        stop_res = self.client.post("/api/session/stop")
        self.assertEqual(stop_res.status_code, 200)
        self.assertTrue(stop_res.json()["success"])
        self.assertEqual(agent_service._status, "stopped")

    def test_06_metrics_history_endpoint(self):
        """Test GET /api/metrics/history endpoint."""
        res = self.client.get("/api/metrics/history")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("episodes", data)
        self.assertIn("best_reward", data)
        self.assertIn("total_completed", data)

    def test_07_websocket_stream(self):
        """Test WebSocket connection and init_state snapshot delivery."""
        with self.client.websocket_connect("/ws/session") as websocket:
            data = websocket.receive_json()
            self.assertEqual(data.get("type"), "init_state")
            self.assertIn("model_status", data)
            self.assertIn("metrics", data)
            self.assertEqual(data["model_status"]["model_loaded"], True)

    def test_08_missing_checkpoint_error_handling(self):
        """Verify graceful error status when model checkpoint is missing."""
        from webapp.agent_service import AgentDemonstrationService
        missing_service = AgentDemonstrationService(model_filename="non_existent_ckpt.pt")
        self.assertFalse(missing_service.model_loaded)
        self.assertEqual(missing_service._status, "error")
        start_result = missing_service.start()
        self.assertFalse(start_result["success"])
        self.assertIn("Model not loaded", start_result["message"])

    def test_09_comprehensive_reset_clears_metrics(self):
        """Verify that Reset stops worker, clears episode history, and resets all session counters."""
        # Simulate populated session history
        from webapp.schemas import EpisodeRecord
        agent_service.episode_history.append(EpisodeRecord(
            episode=1, reward=12.5, score=3, steps=120, timestamp=100.0, is_best=True
        ))
        agent_service.best_session_reward = 12.5
        agent_service.current_episode = 2
        agent_service.current_score = 3
        agent_service.current_step = 40
        agent_service.current_episode_reward = 4.2

        # Call reset endpoint
        res = self.client.post("/api/session/reset")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["success"])
        self.assertEqual(agent_service._status, "ready")
        self.assertEqual(agent_service.current_episode, 1)
        self.assertEqual(agent_service.current_score, 0)
        self.assertEqual(agent_service.current_step, 0)
        self.assertEqual(agent_service.current_episode_reward, 0.0)
        self.assertIsNone(agent_service.best_session_reward)
        self.assertEqual(len(agent_service.episode_history), 0)

        # Verify metrics history endpoint returns empty list after reset
        hist_res = self.client.get("/api/metrics/history")
        hist_data = hist_res.json()
        self.assertEqual(len(hist_data["episodes"]), 0)
        self.assertIsNone(hist_data["best_reward"])

    def test_10_model_download_missing_url(self):
        """Verify download_model_checkpoint fails gracefully when MODEL_URL is missing."""
        test_path = os.path.join(PROJECT_ROOT, "runs", "test_missing_model.pt")
        if os.path.exists(test_path):
            os.remove(test_path)

        with patch.dict(os.environ, {}, clear=True):
            success, msg = download_model_checkpoint(test_path, model_url=None)
            self.assertFalse(success)
            self.assertIn("MODEL_URL environment variable is not configured", msg)
            self.assertFalse(os.path.exists(test_path))

    def test_11_model_download_success(self):
        """Verify safe download writes to temporary file and atomically installs checkpoint."""
        test_path = os.path.join(PROJECT_ROOT, "runs", "test_download_model.pt")
        if os.path.exists(test_path):
            os.remove(test_path)

        fake_checkpoint_bytes = b"FLAPPY_DQN_MOCK_CHECKPOINT_DATA" * 50  # > 1024 bytes
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.side_effect = [fake_checkpoint_bytes, b""]
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False

        with patch("urllib.request.urlopen", return_value=mock_response):
            success, msg = download_model_checkpoint(
                test_path,
                model_url="https://example.com/models/flappybirdv0.pt",
                min_size_bytes=100
            )
            self.assertTrue(success)
            self.assertIn("Successfully downloaded", msg)
            self.assertTrue(os.path.exists(test_path))
            with open(test_path, "rb") as f:
                content = f.read()
            self.assertEqual(content, fake_checkpoint_bytes)

        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)

    def test_12_model_download_empty_or_small_file_rejected(self):
        """Verify empty or undersized downloads are rejected and leave no partial files."""
        test_path = os.path.join(PROJECT_ROOT, "runs", "test_empty_model.pt")
        if os.path.exists(test_path):
            os.remove(test_path)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.side_effect = [b"short", b""]
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False

        with patch("urllib.request.urlopen", return_value=mock_response):
            success, msg = download_model_checkpoint(
                test_path,
                model_url="https://example.com/models/empty.pt",
                min_size_bytes=1024
            )
            self.assertFalse(success)
            self.assertIn("too small", msg)
            self.assertFalse(os.path.exists(test_path))

    def test_13_model_download_network_error_cleans_up(self):
        """Verify network exceptions are caught cleanly and no partial files are left."""
        import urllib.error
        test_path = os.path.join(PROJECT_ROOT, "runs", "test_network_fail.pt")
        if os.path.exists(test_path):
            os.remove(test_path)

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            success, msg = download_model_checkpoint(
                test_path,
                model_url="https://example.com/unreachable/model.pt"
            )
            self.assertFalse(success)
            self.assertIn("Failed to download model checkpoint", msg)
            self.assertFalse(os.path.exists(test_path))

    def test_14_production_spa_fallback_and_static_assets(self):
        """Verify root and SPA routes serve index.html, while static assets return directly."""
        # 1. Root route
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertIn("text/html", res_root.headers.get("content-type", ""))
        self.assertIn("Flappy Bird AI", res_root.text)

        # 2. Client-side SPA navigation route (e.g. /dashboard or /analytics)
        res_spa = self.client.get("/dashboard")
        self.assertEqual(res_spa.status_code, 200)
        self.assertIn("text/html", res_spa.headers.get("content-type", ""))
        self.assertIn("Flappy Bird AI", res_spa.text)

        # 3. Static assets if dist exists
        if os.path.exists(FRONTEND_DIST):
            assets_dir = os.path.join(FRONTEND_DIST, "assets")
            if os.path.exists(assets_dir):
                asset_files = os.listdir(assets_dir)
                if asset_files:
                    asset_res = self.client.get(f"/assets/{asset_files[0]}")
                    self.assertEqual(asset_res.status_code, 200)

    def test_15_api_404_not_intercepted_by_spa(self):
        """Ensure missing API routes return 404 JSON, never falling back to index.html."""
        res = self.client.get("/api/nonexistent-route")
        self.assertEqual(res.status_code, 404)
        data = res.json()
        self.assertIn("detail", data)
        self.assertIn("API route not found", data["detail"])

    def test_16_model_download_zip_archive_success(self):
        """Verify ZIP archive is downloaded, safely extracted, and checkpoint installed."""
        test_path = os.path.join(PROJECT_ROOT, "runs", "test_zip_extracted.pt")
        if os.path.exists(test_path):
            os.remove(test_path)

        fake_model_data = b"ZIP_MODEL_WEIGHTS_CONTENT" * 100  # > 1024 bytes
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("test_zip_extracted.pt", fake_model_data)
        zip_bytes = zip_buf.getvalue()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.side_effect = [zip_bytes, b""]
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False

        with patch("urllib.request.urlopen", return_value=mock_response):
            success, msg = download_model_checkpoint(
                test_path,
                model_url="https://github.com/arpbiswas04-coder/FlappyBird-DQN/releases/download/v1.0.0/flappybird-model.zip",
                min_size_bytes=100
            )
            self.assertTrue(success, f"Expected success but got: {msg}")
            self.assertIn("Successfully downloaded and extracted", msg)
            self.assertTrue(os.path.exists(test_path))
            with open(test_path, "rb") as f:
                self.assertEqual(f.read(), fake_model_data)

        if os.path.exists(test_path):
            os.remove(test_path)

    def test_17_model_download_zip_nested_path_safe_extraction(self):
        """Verify nested paths in ZIP archive extract safely to the target path without traversal."""
        test_path = os.path.join(PROJECT_ROOT, "runs", "test_nested.pt")
        if os.path.exists(test_path):
            os.remove(test_path)

        fake_model_data = b"NESTED_ZIP_MODEL_DATA" * 100
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            # Nested member in archive
            zf.writestr("deeply/nested/folder/test_nested.pt", fake_model_data)
        zip_bytes = zip_buf.getvalue()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.side_effect = [zip_bytes, b""]
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False

        with patch("urllib.request.urlopen", return_value=mock_response):
            success, msg = download_model_checkpoint(
                test_path,
                model_url="https://example.com/models.zip",
                min_size_bytes=100
            )
            self.assertTrue(success)
            self.assertTrue(os.path.exists(test_path))
            with open(test_path, "rb") as f:
                self.assertEqual(f.read(), fake_model_data)

        if os.path.exists(test_path):
            os.remove(test_path)

    def test_18_model_download_zip_missing_checkpoint_file(self):
        """Verify ZIP archive missing the target checkpoint fails with a clear message."""
        test_path = os.path.join(PROJECT_ROOT, "runs", "flappybirdv0.pt_test_missing")
        if os.path.exists(test_path):
            os.remove(test_path)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("wrong_model_name.pt", b"SOME_DATA" * 100)
        zip_bytes = zip_buf.getvalue()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.side_effect = [zip_bytes, b""]
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False

        with patch("urllib.request.urlopen", return_value=mock_response):
            success, msg = download_model_checkpoint(
                test_path,
                model_url="https://example.com/models.zip",
                min_size_bytes=100
            )
            self.assertFalse(success)
            self.assertIn("not found in ZIP archive", msg)
            self.assertFalse(os.path.exists(test_path))

    def test_19_model_download_zip_corrupted(self):
        """Verify corrupted ZIP archive download reports error clearly and leaves no partial files."""
        test_path = os.path.join(PROJECT_ROOT, "runs", "test_corrupt.pt")
        if os.path.exists(test_path):
            os.remove(test_path)

        corrupt_bytes = b"NOT_A_VALID_ZIP_FILE_DATA" * 50

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.side_effect = [corrupt_bytes, b""]
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False

        with patch("urllib.request.urlopen", return_value=mock_response):
            success, msg = download_model_checkpoint(
                test_path,
                model_url="https://example.com/models.zip",
                min_size_bytes=100
            )
            self.assertFalse(success)
            self.assertIn("not a valid ZIP archive", msg)
            self.assertFalse(os.path.exists(test_path))

    def test_20_model_download_zip_empty_checkpoint(self):
        """Verify ZIP archive containing empty checkpoint file is rejected."""
        test_path = os.path.join(PROJECT_ROOT, "runs", "test_empty_in_zip.pt")
        if os.path.exists(test_path):
            os.remove(test_path)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("test_empty_in_zip.pt", b"")
        zip_bytes = zip_buf.getvalue()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.side_effect = [zip_bytes, b""]
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False

        with patch("urllib.request.urlopen", return_value=mock_response):
            success, msg = download_model_checkpoint(
                test_path,
                model_url="https://example.com/models.zip",
                min_size_bytes=100
            )
            self.assertFalse(success)
            self.assertIn("too small", msg)
            self.assertFalse(os.path.exists(test_path))


if __name__ == "__main__":
    unittest.main()
