import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import { Header } from './components/Header';
import { GamePanel } from './components/GamePanel';
import { MetricsGrid } from './components/MetricsGrid';
import { RewardChart } from './components/RewardChart';
import { RecentEpisodes } from './components/RecentEpisodes';
import type { ModelStatus, EpisodeRecord, MetricsHistory } from './types';

export const App: React.FC = () => {
  // Model & Session state
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [sessionStatus, setSessionStatus] = useState<'ready' | 'running' | 'paused' | 'stopped' | 'error'>('ready');
  const [isConnected, setIsConnected] = useState(false);
  const [targetFps, setTargetFps] = useState(30);

  // Live Game & Inference State
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [currentEpisode, setCurrentEpisode] = useState(1);
  const [currentScore, setCurrentScore] = useState(0);
  const [currentEpisodeReward, setCurrentEpisodeReward] = useState<number | null>(null);
  const [bestReward, setBestReward] = useState<number | null>(null);

  // Metrics history
  const [episodes, setEpisodes] = useState<EpisodeRecord[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  // Fetch initial REST status
  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/model-status');
      if (res.ok) {
        const data: ModelStatus = await res.json();
        setModelStatus(data);
        setSessionStatus(data.session_status);
      }
    } catch (err) {
      console.warn('Initial status fetch error:', err);
    }
  }, []);

  const fetchHistory = useCallback(async () => {
    try {
      const res = await fetch('/api/metrics/history');
      if (res.ok) {
        const data: MetricsHistory = await res.json();
        setEpisodes(data.episodes);
        setBestReward(data.best_reward);
      }
    } catch (err) {
      console.warn('History fetch error:', err);
    }
  }, []);

  // Connect WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/session`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'init_state') {
          setModelStatus(data.model_status);
          setSessionStatus(data.status);
          setCurrentEpisode(data.current_episode);
          setCurrentEpisodeReward(data.current_episode_reward);
          setCurrentScore(data.current_score);
          setCurrentStep(data.current_step);
          if (data.last_frame) setCurrentFrame(data.last_frame);
          if (data.metrics?.episodes) {
            setEpisodes(data.metrics.episodes);
          }
          if (data.metrics?.best_reward !== undefined) {
            setBestReward(data.metrics.best_reward);
          }
        } else if (data.type === 'frame') {
          setCurrentFrame(data.frame);
          setCurrentStep(data.step);
          setCurrentEpisode(data.episode);
          setCurrentEpisodeReward(data.episode_reward);
          setCurrentScore(data.score);
          if (data.status) setSessionStatus(data.status);
        } else if (data.type === 'episode_complete') {
          const newEp: EpisodeRecord = {
            episode: data.episode,
            reward: data.reward,
            score: data.score,
            steps: data.steps,
            timestamp: Date.now() / 1000,
            is_best: data.is_new_best,
          };
          setEpisodes((prev) => [...prev, newEp]);
          if (data.best_reward !== null) {
            setBestReward(data.best_reward);
          }
        } else if (data.type === 'session_reset') {
          // Complete synchronous reset of demonstration session
          setEpisodes([]);
          setCurrentEpisode(data.current_episode);
          setCurrentEpisodeReward(null);
          setCurrentScore(data.current_score);
          setCurrentStep(data.current_step);
          setBestReward(null);
          if (data.last_frame) setCurrentFrame(data.last_frame);
          setSessionStatus(data.status);
        } else if (data.type === 'status_change') {
          setSessionStatus(data.status);
        } else if (data.type === 'error') {
          setSessionStatus('error');
        }
      } catch (err) {
        console.error('WebSocket parse error:', err);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      reconnectTimeoutRef.current = window.setTimeout(() => {
        connectWebSocket();
      }, 2000);
    };

    ws.onerror = (err) => {
      console.warn('WebSocket connection error:', err);
      ws.close();
    };
  }, []);

  useEffect(() => {
    fetchStatus();
    fetchHistory();
    connectWebSocket();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [fetchStatus, fetchHistory, connectWebSocket]);

  // Session Control Handlers
  const handleStart = async () => {
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ command: 'start', fps: targetFps }));
      } else {
        await fetch('/api/session/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fps: targetFps }),
        });
      }
      setSessionStatus('running');
    } catch (err) {
      console.error('Error starting session:', err);
    }
  };

  const handlePauseResume = async () => {
    const isPaused = sessionStatus === 'paused';
    const command = isPaused ? 'resume' : 'pause';
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ command }));
      } else {
        await fetch(`/api/session/${command}`, { method: 'POST' });
      }
      setSessionStatus(isPaused ? 'running' : 'paused');
    } catch (err) {
      console.error(`Error on ${command}:`, err);
    }
  };

  const handleReset = async () => {
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ command: 'reset' }));
      } else {
        await fetch('/api/session/reset', { method: 'POST' });
      }
      // Immediately reset local state to ensure snappy UI feedback
      setEpisodes([]);
      setCurrentEpisode(1);
      setCurrentEpisodeReward(null);
      setCurrentScore(0);
      setCurrentStep(0);
      setBestReward(null);
      setSessionStatus('ready');
    } catch (err) {
      console.error('Error resetting episode:', err);
    }
  };

  const handleStop = async () => {
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ command: 'stop' }));
      } else {
        await fetch('/api/session/stop', { method: 'POST' });
      }
      setSessionStatus('stopped');
    } catch (err) {
      console.error('Error stopping session:', err);
    }
  };

  const handleFpsChange = (fps: number) => {
    setTargetFps(fps);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: 'set_fps', fps }));
    }
  };

  return (
    <div className="brutal-app-layout">
      <Header
        modelStatus={modelStatus}
        sessionStatus={sessionStatus}
        isConnected={isConnected}
        targetFps={targetFps}
        onStart={handleStart}
        onPauseResume={handlePauseResume}
        onReset={handleReset}
        onStop={handleStop}
        onFpsChange={handleFpsChange}
      />

      <main className="brutal-dashboard-body">
        <div className="brutal-main-grid">
          {/* Left Column: Portrait Game Display */}
          <section className="column-game" aria-label="Game Screen">
            <GamePanel
              frameSrc={currentFrame}
              score={currentScore}
              step={currentStep}
              sessionStatus={sessionStatus}
              modelLoaded={modelStatus?.model_loaded ?? false}
              onStart={handleStart}
              onPauseResume={handlePauseResume}
            />
          </section>

          {/* Right Column: Metrics, Reward Chart, and Expanded Recent Episodes */}
          <section className="column-metrics" aria-label="Performance Metrics">
            <MetricsGrid
              currentEpisode={currentEpisode}
              currentEpisodeReward={currentEpisodeReward}
              bestReward={bestReward}
              sessionStatus={sessionStatus}
              modelStatus={modelStatus}
            />

            <RewardChart
              episodes={episodes}
              historicalBest={modelStatus?.best_recorded_reward ?? null}
            />

            {/* Expanded Recent Episodes Panel occupying full width below chart */}
            <RecentEpisodes episodes={episodes} />
          </section>
        </div>
      </main>
    </div>
  );
};

export default App;
