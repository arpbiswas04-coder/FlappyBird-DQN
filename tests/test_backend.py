import os
import sys
import unittest
import numpy as np
import torch
from fastapi.testclient import TestClient

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from webapp.main import app, agent_service
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


if __name__ == "__main__":
    unittest.main()
