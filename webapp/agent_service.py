import os
import sys
import time
import base64
import io
import re
import threading
import asyncio
from typing import Optional, List, Set, Dict, Any
from PIL import Image
import numpy as np

# Set headless SDL video driver before importing pygame/gymnasium
os.environ["SDL_VIDEODRIVER"] = "dummy"

import torch
import gymnasium as gym
import flappy_bird_gymnasium

# Add project root to sys.path to import dqn
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dqn import DQN
from webapp.schemas import EpisodeRecord, ModelStatusResponse, MetricsHistoryResponse


class AgentDemonstrationService:
    def __init__(self, runs_dir: str = "runs", model_filename: str = "flappybirdv0.pt", log_filename: str = "flappybirdv0.log"):
        self.runs_dir = os.path.join(PROJECT_ROOT, runs_dir)
        self.model_path = os.path.join(self.runs_dir, model_filename)
        self.log_path = os.path.join(self.runs_dir, log_filename)
        
        self.device = self._detect_device()
        self.model: Optional[DQN] = None
        self.model_loaded = False
        self.state_dim = 180
        self.action_dim = 2
        self.best_recorded_reward: Optional[float] = None
        
        # Session state
        self._session_id = 0
        self._status = "ready"  # ready, running, paused, stopped, error
        self._status_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self.target_fps = 30
        
        # Metrics tracking
        self.current_episode = 1
        self.current_episode_reward = 0.0
        self.current_step = 0
        self.current_score = 0
        self.best_session_reward: Optional[float] = None
        self.episode_history: List[EpisodeRecord] = []
        self.last_frame_b64: Optional[str] = None
        self.last_action: int = 0
        
        # Async broadcast setup
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._active_connections: Set[Any] = set()
        self._connections_lock = threading.Lock()
        
        # Parse initial history & load model
        self._load_best_reward_from_log()
        self.load_model()
        
        # Render initial starting frame so game canvas is populated immediately
        self._capture_initial_frame()

    def _detect_device(self) -> str:
        """Select safe execution device (CPU or CUDA if fully functional)."""
        if torch.cuda.is_available():
            try:
                t = torch.zeros(1, device="cuda")
                del t
                return "cuda"
            except Exception:
                return "cpu"
        return "cpu"

    def _load_best_reward_from_log(self):
        """Extract best recorded reward from historical training log if available."""
        if not os.path.exists(self.log_path):
            return
        try:
            best_val = None
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    m = re.search(r"best reward\s*=\s*([-\d.]+)", line)
                    if m:
                        val = float(m.group(1))
                        if best_val is None or val > best_val:
                            best_val = val
            self.best_recorded_reward = best_val
        except Exception as e:
            print(f"Warning: could not parse training log: {e}")

    def _capture_initial_frame(self):
        """Generate a clean initial game frame from the environment."""
        try:
            env = gym.make("FlappyBird-v0", render_mode="rgb_array")
            env.reset()
            frame_arr = env.render()
            if frame_arr is not None:
                self.last_frame_b64 = self._encode_frame(frame_arr)
            env.close()
        except Exception as e:
            print(f"Could not render initial frame: {e}")

    def load_model(self) -> bool:
        """Load the PyTorch DQN model checkpoint safely using weights_only=True."""
        try:
            if not os.path.exists(self.model_path):
                self._status = "error"
                print(f"Model file not found at {self.model_path}")
                return False

            model = DQN(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=256)
            
            # Load with weights_only=True for secure deserialization
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint)
            model.to(self.device)
            model.eval()
            
            self.model = model
            self.model_loaded = True
            self._status = "ready"
            return True
        except Exception as e:
            self.model_loaded = False
            self._status = "error"
            print(f"Failed to load checkpoint: {e}")
            return False

    def get_status(self) -> ModelStatusResponse:
        """Return the current model status and server state."""
        return ModelStatusResponse(
            model_loaded=self.model_loaded,
            checkpoint_path=os.path.basename(self.model_path),
            checkpoint_exists=os.path.exists(self.model_path),
            device=self.device,
            architecture="DQN (Linear 180->256 -> ReLU -> Linear 256->256 -> ReLU -> Linear 256->2)",
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            best_recorded_reward=self.best_recorded_reward,
            session_status=self._status,
            current_fps=self.target_fps,
        )

    def get_metrics_history(self) -> MetricsHistoryResponse:
        """Return the list of completed episodes and session best reward."""
        return MetricsHistoryResponse(
            episodes=list(self.episode_history),
            best_reward=self.best_session_reward,
            total_completed=len(self.episode_history),
        )

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Register the asyncio event loop for threadsafe WebSocket broadcasting."""
        self._loop = loop

    def register_connection(self, websocket: Any):
        """Register an active WebSocket client."""
        with self._connections_lock:
            self._active_connections.add(websocket)

    def unregister_connection(self, websocket: Any):
        """Remove a disconnected WebSocket client."""
        with self._connections_lock:
            self._active_connections.discard(websocket)

    def _broadcast_payload(self, payload: Dict[str, Any]):
        """Schedule threadsafe broadcast of JSON message to all connected clients."""
        if self._loop is None or not self._loop.is_running():
            return
        
        async def _send_all():
            with self._connections_lock:
                clients = list(self._active_connections)
            if not clients:
                return
            
            dead_clients = []
            for client in clients:
                try:
                    await client.send_json(payload)
                except Exception:
                    dead_clients.append(client)
            
            if dead_clients:
                with self._connections_lock:
                    for dc in dead_clients:
                        self._active_connections.discard(dc)

        asyncio.run_coroutine_threadsafe(_send_all(), self._loop)

    def start(self, fps: int = 30) -> Dict[str, Any]:
        """Start or resume the demonstration session."""
        with self._status_lock:
            if not self.model_loaded:
                return {"success": False, "status": self._status, "message": "Model not loaded. Checkpoint missing or incompatible."}
            
            if self._status == "paused":
                self.target_fps = fps
                self._pause_event.set()
                self._status = "running"
                self._broadcast_payload({"type": "status_change", "status": self._status, "message": "Resumed"})
                return {"success": True, "status": self._status, "message": "Demonstration resumed."}

            if self._status == "running":
                self.target_fps = fps
                return {"success": True, "status": self._status, "message": "Demonstration is already running."}

            # If stopped or ready, launch fresh thread for new session
            self.target_fps = fps
            self._stop_event.clear()
            self._pause_event.set()
            self._status = "running"
            
            sid = self._session_id
            self._thread = threading.Thread(
                target=self._demonstration_worker, 
                args=(sid,),
                daemon=True, 
                name=f"FlappyDQNWorker-{sid}"
            )
            self._thread.start()
            
            self._broadcast_payload({"type": "status_change", "status": self._status, "message": "Started"})
            return {"success": True, "status": self._status, "message": "Demonstration session started."}

    def pause(self) -> Dict[str, Any]:
        """Pause demonstration without dropping state."""
        with self._status_lock:
            if self._status != "running":
                return {"success": False, "status": self._status, "message": f"Cannot pause while {self._status}."}
            
            self._pause_event.clear()
            self._status = "paused"
            self._broadcast_payload({"type": "status_change", "status": self._status, "message": "Demonstration paused."})
            return {"success": True, "status": self._status, "message": "Demonstration paused."}

    def resume(self) -> Dict[str, Any]:
        """Resume paused demonstration."""
        with self._status_lock:
            if self._status != "paused":
                return {"success": False, "status": self._status, "message": f"Cannot resume while {self._status}."}
            
            self._pause_event.set()
            self._status = "running"
            self._broadcast_payload({"type": "status_change", "status": self._status, "message": "Demonstration resumed."})
            return {"success": True, "status": self._status, "message": "Demonstration resumed."}

    def reset(self) -> Dict[str, Any]:
        """
        Safely reset the entire demonstration session.
        Stops background worker, resets environment, clears episode history & reward progression.
        """
        # Step 1: Safely terminate running thread and invalidate session ID to prevent race conditions
        with self._status_lock:
            self._stop_event.set()
            self._pause_event.set()
            self._session_id += 1
            old_thread = self._thread
            self._status = "ready"

        if old_thread and old_thread.is_alive():
            old_thread.join(timeout=1.5)

        # Step 2: Clear all session metrics, score, steps, and history
        with self._status_lock:
            self.current_episode = 1
            self.current_episode_reward = 0.0
            self.current_step = 0
            self.current_score = 0
            self.last_action = 0
            self.best_session_reward = None
            self.episode_history = []
            
            # Step 3: Capture fresh initial environment frame
            self._capture_initial_frame()

            # Step 4: Broadcast complete session_reset message to all clients
            reset_payload = {
                "type": "session_reset",
                "session_id": self._session_id,
                "status": "ready",
                "last_frame": self.last_frame_b64,
                "current_episode": 1,
                "current_score": 0,
                "current_step": 0,
                "current_episode_reward": 0.0,
                "best_reward": None,
                "episodes": [],
            }
            self._broadcast_payload(reset_payload)

            return {
                "success": True, 
                "status": "ready", 
                "message": "Demonstration session and metrics reset successfully."
            }

    def stop(self) -> Dict[str, Any]:
        """Safely stop the active demonstration session."""
        with self._status_lock:
            self._stop_event.set()
            self._pause_event.set()  # release wait if paused
            self._status = "stopped"
            
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            
        self._broadcast_payload({"type": "status_change", "status": self._status, "message": "Demonstration stopped."})
        return {"success": True, "status": self._status, "message": "Demonstration stopped cleanly."}

    def _encode_frame(self, frame_array: np.ndarray) -> str:
        """Convert RGB numpy array to base64-encoded JPEG image."""
        img = Image.fromarray(frame_array)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _demonstration_worker(self, session_id: int):
        """Background inference worker stepping the Flappy Bird environment with the trained DQN."""
        env = None
        try:
            # Create headless environment
            env = gym.make("FlappyBird-v0", render_mode="rgb_array")
            obs, info = env.reset()
            self.current_episode_reward = 0.0
            self.current_step = 0
            self.current_score = 0
            
            idle_disconnect_counter = 0

            while not self._stop_event.is_set():
                if self._session_id != session_id:
                    break

                loop_start = time.perf_counter()

                # Wait if paused
                self._pause_event.wait()
                if self._stop_event.is_set() or self._session_id != session_id:
                    break

                # Auto-stop check: if no websocket clients connected for 45s, stop demonstration
                with self._connections_lock:
                    num_clients = len(self._active_connections)
                if num_clients == 0:
                    idle_disconnect_counter += 1
                    if idle_disconnect_counter > 1350:
                        with self._status_lock:
                            if self._session_id == session_id:
                                self._status = "stopped"
                        break
                else:
                    idle_disconnect_counter = 0

                # Render current frame
                frame_arr = env.render()
                if frame_arr is not None:
                    self.last_frame_b64 = self._encode_frame(frame_arr)

                # Select action using trained DQN in evaluation mode
                state_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.inference_mode():
                    q_vals = self.model(state_tensor)
                    action = q_vals.argmax(dim=1).item()
                self.last_action = action

                # Step the actual environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Guard against race condition after reset
                if self._session_id != session_id:
                    break

                self.current_episode_reward += float(reward)
                self.current_step += 1
                self.current_score = int(info.get("score", 0))

                # Broadcast live step update
                frame_msg = {
                    "type": "frame",
                    "session_id": session_id,
                    "frame": self.last_frame_b64,
                    "step": self.current_step,
                    "episode": self.current_episode,
                    "action": action,
                    "action_name": "Flap" if action == 1 else "Glide",
                    "reward": float(reward),
                    "episode_reward": self.current_episode_reward,
                    "score": self.current_score,
                    "status": self._status,
                }
                self._broadcast_payload(frame_msg)

                # Handle episode completion
                if terminated or truncated:
                    if self._session_id != session_id:
                        break

                    is_best = False
                    if self.best_session_reward is None or self.current_episode_reward > self.best_session_reward:
                        self.best_session_reward = self.current_episode_reward
                        is_best = True
                    
                    record = EpisodeRecord(
                        episode=self.current_episode,
                        reward=self.current_episode_reward,
                        score=self.current_score,
                        steps=self.current_step,
                        timestamp=time.time(),
                        is_best=is_best,
                    )
                    self.episode_history.append(record)
                    if len(self.episode_history) > 300:
                        self.episode_history.pop(0)

                    # Send episode complete message
                    complete_msg = {
                        "type": "episode_complete",
                        "session_id": session_id,
                        "episode": self.current_episode,
                        "reward": self.current_episode_reward,
                        "score": self.current_score,
                        "steps": self.current_step,
                        "best_reward": self.best_session_reward,
                        "is_new_best": is_best,
                    }
                    self._broadcast_payload(complete_msg)

                    # Reset for next episode
                    obs, info = env.reset()
                    self.current_episode += 1
                    self.current_episode_reward = 0.0
                    self.current_step = 0
                    self.current_score = 0
                else:
                    obs = next_obs

                # Pacing control
                target_interval = 1.0 / max(5, min(60, self.target_fps))
                elapsed = time.perf_counter() - loop_start
                sleep_time = target_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            print(f"Error in demonstration worker: {e}")
            with self._status_lock:
                if self._session_id == session_id:
                    self._status = "error"
            self._broadcast_payload({"type": "error", "error": str(e), "status": "error", "session_id": session_id})
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            with self._status_lock:
                if self._session_id == session_id and self._status not in ("error", "ready"):
                    self._status = "stopped"
            if self._session_id == session_id:
                self._broadcast_payload({"type": "status_change", "status": self._status, "message": "Worker finished", "session_id": session_id})
