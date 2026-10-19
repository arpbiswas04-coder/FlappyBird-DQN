import React from 'react';
import { Layers, Activity, Trophy, Clock, Cpu } from 'lucide-react';
import type { ModelStatus } from '../types';

interface MetricsGridProps {
  currentEpisode: number;
  currentEpisodeReward: number | null;
  bestReward: number | null;
  sessionStatus: 'ready' | 'running' | 'paused' | 'stopped' | 'error';
  modelStatus: ModelStatus | null;
}

export const MetricsGrid: React.FC<MetricsGridProps> = ({
  currentEpisode,
  currentEpisodeReward,
  bestReward,
  sessionStatus,
  modelStatus,
}) => {
  const getStatusLabel = () => {
    switch (sessionStatus) {
      case 'running':
        return 'Running';
      case 'paused':
        return 'Paused';
      case 'stopped':
        return 'Stopped';
      case 'error':
        return 'Error';
      default:
        return 'Idle';
    }
  };

  const getStatusClass = () => {
    switch (sessionStatus) {
      case 'running':
        return 'status-badge-running';
      case 'paused':
        return 'status-badge-paused';
      case 'stopped':
        return 'status-badge-stopped';
      case 'error':
        return 'status-badge-error';
      default:
        return 'status-badge-idle';
    }
  };

  return (
    <div className="brutal-metrics-grid">
      {/* 1. Current Episode Card (Cyan Tint) */}
      <div className="brutal-stat-card stat-cyan">
        <div className="stat-card-header">
          <span className="stat-label">CURRENT EPISODE</span>
          <Layers size={18} className="stat-header-icon" />
        </div>
        <div className="stat-value-box">
          <span className="stat-number mono">#{currentEpisode}</span>
        </div>
      </div>

      {/* 2. Episode Reward Card (Green Tint) */}
      <div className="brutal-stat-card stat-green">
        <div className="stat-card-header">
          <span className="stat-label">EPISODE REWARD</span>
          <Activity size={18} className="stat-header-icon" />
        </div>
        <div className="stat-value-box">
          <span className="stat-number mono">
            {currentEpisodeReward !== null
              ? (currentEpisodeReward >= 0 ? `+${currentEpisodeReward.toFixed(2)}` : currentEpisodeReward.toFixed(2))
              : '—'}
          </span>
        </div>
      </div>

      {/* 3. Best Reward Card (Yellow Tint) */}
      <div className="brutal-stat-card stat-yellow">
        <div className="stat-card-header">
          <span className="stat-label">BEST REWARD</span>
          <Trophy size={18} className="stat-header-icon" />
        </div>
        <div className="stat-value-box">
          <span className="stat-number mono">
            {bestReward !== null
              ? (bestReward >= 0 ? `+${bestReward.toFixed(2)}` : bestReward.toFixed(2))
              : '—'}
          </span>
        </div>
      </div>

      {/* 4. Agent Status Card (Purple Tint - No subtitles) */}
      <div className="brutal-stat-card stat-purple">
        <div className="stat-card-header">
          <span className="stat-label">AGENT STATUS</span>
          <Clock size={18} className="stat-header-icon" />
        </div>
        <div className="stat-value-box">
          <span className={`brutal-status-badge ${getStatusClass()}`}>
            {getStatusLabel()}
          </span>
        </div>
      </div>

      {/* 5. Execution Device Card (Coral/Peach Tint) */}
      <div className="brutal-stat-card stat-coral">
        <div className="stat-card-header">
          <span className="stat-label">EXECUTION DEVICE</span>
          <Cpu size={18} className="stat-header-icon" />
        </div>
        <div className="stat-value-box">
          <span className="stat-number mono uppercase">
            {modelStatus?.device || 'CPU'}
          </span>
        </div>
      </div>
    </div>
  );
};
