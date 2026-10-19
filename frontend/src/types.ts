export interface ModelStatus {
  model_loaded: boolean;
  checkpoint_path: string;
  checkpoint_exists: boolean;
  device: string;
  architecture: string;
  state_dim: number;
  action_dim: number;
  best_recorded_reward: number | null;
  session_status: 'ready' | 'running' | 'paused' | 'stopped' | 'error';
  current_fps: number;
}

export interface EpisodeRecord {
  episode: number;
  reward: number;
  score: number;
  steps: number;
  timestamp: number;
  is_best?: boolean;
}

export interface MetricsHistory {
  episodes: EpisodeRecord[];
  best_reward: number | null;
  total_completed: number;
}

export interface LiveFramePayload {
  type: 'frame';
  frame: string;
  step: number;
  episode: number;
  action: number;
  action_name: string;
  reward: number;
  episode_reward: number;
  score: number;
  status: string;
}

export interface EpisodeCompletePayload {
  type: 'episode_complete';
  episode: number;
  reward: number;
  score: number;
  steps: number;
  best_reward: number;
  is_new_best: boolean;
}

export interface InitStatePayload {
  type: 'init_state';
  model_status: ModelStatus;
  metrics: MetricsHistory;
  current_episode: number;
  current_episode_reward: number;
  current_score: number;
  current_step: number;
  last_frame: string | null;
  last_action: number;
  status: 'ready' | 'running' | 'paused' | 'stopped' | 'error';
}

export interface StatusChangePayload {
  type: 'status_change';
  status: 'ready' | 'running' | 'paused' | 'stopped' | 'error';
  message: string;
}

export interface ErrorPayload {
  type: 'error';
  error: string;
  status: string;
}

export interface SessionResetPayload {
  type: 'session_reset';
  session_id: number;
  status: 'ready';
  last_frame: string | null;
  current_episode: number;
  current_score: number;
  current_step: number;
  current_episode_reward: number;
  best_reward: number | null;
  episodes: EpisodeRecord[];
}
