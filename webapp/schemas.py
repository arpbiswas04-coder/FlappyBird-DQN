from typing import Optional, List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


class ModelStatusResponse(BaseModel):
    model_loaded: bool
    checkpoint_path: str
    checkpoint_exists: bool
    device: str
    architecture: str
    state_dim: int
    action_dim: int
    best_recorded_reward: Optional[float] = None
    session_status: str
    current_fps: int = 30
    message: Optional[str] = None


class SessionControlRequest(BaseModel):
    fps: Optional[int] = Field(default=30, ge=5, le=60, description="Inference pacing (FPS)")


class SessionControlResponse(BaseModel):
    success: bool
    status: str
    message: str


class EpisodeRecord(BaseModel):
    episode: int
    reward: float
    score: int
    steps: int
    timestamp: float
    is_best: bool = False


class MetricsHistoryResponse(BaseModel):
    episodes: List[EpisodeRecord]
    best_reward: Optional[float]
    total_completed: int
