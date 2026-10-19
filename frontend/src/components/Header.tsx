import React from 'react';
import { Play, Pause, RotateCcw, Square } from 'lucide-react';
import type { ModelStatus } from '../types';

interface HeaderProps {
  modelStatus: ModelStatus | null;
  sessionStatus: 'ready' | 'running' | 'paused' | 'stopped' | 'error';
  isConnected: boolean;
  targetFps: number;
  onStart: () => void;
  onPauseResume: () => void;
  onReset: () => void;
  onStop: () => void;
  onFpsChange: (fps: number) => void;
}

export const Header: React.FC<HeaderProps> = ({
  modelStatus,
  sessionStatus,
  isConnected,
  targetFps,
  onStart,
  onPauseResume,
  onReset,
  onStop,
  onFpsChange,
}) => {
  const isRunning = sessionStatus === 'running';
  const isPaused = sessionStatus === 'paused';

  return (
    <header className="brutal-header">
      {/* Brand Title: Clean & Compact Flappy Bird AI */}
      <div className="brutal-brand">
        <div className="brand-icon-box">
          <img src="/favicon.svg" alt="Flappy Bird" width="30" height="30" />
        </div>
        <div className="brand-title-wrap">
          <h1 className="brand-title">Flappy Bird AI</h1>
        </div>
      </div>

      {/* Top-Right Controls */}
      <div className="brutal-header-controls">
        {/* Playback Speed Selector */}
        <div className="speed-selector-group">
          <span className="speed-label">SPEED:</span>
          <div className="speed-buttons">
            {[15, 30, 45, 60].map((fps) => (
              <button
                key={fps}
                className={`speed-btn ${targetFps === fps ? 'speed-btn-active' : ''}`}
                onClick={() => onFpsChange(fps)}
                title={`Target pacing: ${fps} FPS`}
              >
                {fps} FPS
              </button>
            ))}
          </div>
        </div>

        {/* Action Buttons with Neo-Brutalist Color Codes */}
        <div className="action-button-group">
          {!isRunning && !isPaused ? (
            <button
              id="btn-start"
              className="brutal-btn btn-start"
              onClick={onStart}
              disabled={!modelStatus?.model_loaded}
              title="Start DQN Demonstration"
            >
              <Play size={16} fill="currentColor" />
              <span>Start</span>
            </button>
          ) : (
            <button
              id="btn-pause-resume"
              className={`brutal-btn ${isPaused ? 'btn-resume' : 'btn-pause'}`}
              onClick={onPauseResume}
              title={isPaused ? "Resume demonstration" : "Pause demonstration"}
            >
              {isPaused ? <Play size={16} fill="currentColor" /> : <Pause size={16} />}
              <span>{isPaused ? 'Resume' : 'Pause'}</span>
            </button>
          )}

          <button
            id="btn-reset"
            className="brutal-btn btn-reset"
            onClick={onReset}
            title="Reset demonstration and session metrics"
          >
            <RotateCcw size={16} />
            <span>Reset</span>
          </button>

          <button
            id="btn-stop"
            className="brutal-btn btn-stop"
            onClick={onStop}
            disabled={sessionStatus === 'ready' || sessionStatus === 'stopped'}
            title="Stop demonstration session"
          >
            <Square size={16} />
            <span>Stop</span>
          </button>
        </div>

        {/* Retained: ONLY a green status indicator dot, NO "Live Stream" text or pill */}
        <div
          className={`brutal-status-dot ${isConnected ? 'dot-connected' : 'dot-disconnected'}`}
          title={isConnected ? 'Backend Connected' : 'Reconnecting...'}
        />
      </div>
    </header>
  );
};
