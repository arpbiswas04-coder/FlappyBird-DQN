import React from 'react';
import { Play, Pause, Award, AlertTriangle } from 'lucide-react';

interface GamePanelProps {
  frameSrc: string | null;
  score: number;
  step: number;
  sessionStatus: 'ready' | 'running' | 'paused' | 'stopped' | 'error';
  modelLoaded: boolean;
  onStart: () => void;
  onPauseResume: () => void;
}

export const GamePanel: React.FC<GamePanelProps> = ({
  frameSrc,
  score,
  step,
  sessionStatus,
  modelLoaded,
  onStart,
  onPauseResume,
}) => {
  const isPaused = sessionStatus === 'paused';
  const isStopped = sessionStatus === 'stopped';
  const isReady = sessionStatus === 'ready';
  const isError = sessionStatus === 'error';

  return (
    <div className="brutal-game-card">
      <div className="game-card-header">
        <h2 className="game-card-title">Live Environment</h2>
      </div>

      <div className="game-viewport-frame">
        <div className="game-screen-portrait">
          {/* Real Backend Offscreen Environment Frame */}
          {frameSrc ? (
            <img
              src={frameSrc}
              alt="Live Flappy Bird Environment"
              className="game-canvas-image"
            />
          ) : (
            <div className="game-empty-screen">
              <img src="/favicon.svg" alt="Flappy Bird" className="empty-bird-icon" />
            </div>
          )}

          {/* In-Game Score HUD Badge: Neo-Brutalist score counter reflecting real environment score */}
          {sessionStatus === 'running' && (
            <div className="brutal-hud-score">
              <span className="hud-score-label">SCORE</span>
              <span className="hud-score-number">{score}</span>
            </div>
          )}

          {/* Overlays for Idle / Ready, Paused, Stopped, Error */}
          {isReady && (
            <div className="brutal-overlay overlay-ready">
              <div className="brutal-modal">
                <div className="modal-badge-ready">AI AGENT READY</div>
                <h3 className="modal-title">Watch Agent Play</h3>
                <p className="modal-text">Trained Deep Q-Network autonomously navigating the real Flappy Bird gym world.</p>
                <button
                  id="overlay-start-btn"
                  className="brutal-btn btn-start modal-cta"
                  onClick={onStart}
                  disabled={!modelLoaded}
                >
                  <Play size={18} fill="currentColor" />
                  <span>Start Demonstration</span>
                </button>
              </div>
            </div>
          )}

          {isPaused && (
            <div className="brutal-overlay overlay-paused">
              <div className="brutal-modal">
                <div className="modal-icon-paused">
                  <Pause size={28} />
                </div>
                <h3 className="modal-title">Demonstration Paused</h3>
                <p className="modal-text">Environment paused at step {step}</p>
                <button className="brutal-btn btn-resume modal-cta" onClick={onPauseResume}>
                  <Play size={16} fill="currentColor" />
                  <span>Resume</span>
                </button>
              </div>
            </div>
          )}

          {isStopped && (
            <div className="brutal-overlay overlay-stopped">
              <div className="brutal-modal">
                <Award size={36} className="modal-award-icon" />
                <h3 className="modal-title">Session Stopped</h3>
                <p className="modal-text">Final score: {score} pipes · {step} steps</p>
                <button className="brutal-btn btn-start modal-cta" onClick={onStart}>
                  <Play size={16} fill="currentColor" />
                  <span>Start New Run</span>
                </button>
              </div>
            </div>
          )}

          {isError && (
            <div className="brutal-overlay overlay-error">
              <div className="brutal-modal">
                <AlertTriangle size={36} color="#FF6B5E" />
                <h3 className="modal-title">Model Error</h3>
                <p className="modal-text">Checkpoint missing or could not load model.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
