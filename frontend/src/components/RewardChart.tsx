import React, { useState, useRef, useEffect } from 'react';
import { TrendingUp } from 'lucide-react';
import type { EpisodeRecord } from '../types';

interface RewardChartProps {
  episodes: EpisodeRecord[];
  historicalBest: number | null;
}

export const RewardChart: React.FC<RewardChartProps> = ({ episodes, historicalBest }) => {
  const [hoveredPoint, setHoveredPoint] = useState<EpisodeRecord | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to the right whenever a new completed episode arrives
  useEffect(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollLeft = scrollContainerRef.current.scrollWidth;
    }
  }, [episodes.length]);

  if (episodes.length === 0) {
    return (
      <div className="brutal-chart-card">
        <div className="chart-card-header">
          <div>
            <h3 className="chart-title">Reward Progression</h3>
            <span className="chart-subtitle">Real completed episode rewards</span>
          </div>
          {historicalBest !== null && (
            <div className="historical-badge" title="Benchmark recorded from historical training log">
              <span>Log Benchmark: </span>
              <strong>{historicalBest.toFixed(1)}</strong>
            </div>
          )}
        </div>
        <div className="chart-empty-box">
          <TrendingUp size={36} className="empty-chart-icon" />
          <p className="empty-chart-title">Awaiting Completed Episodes</p>
          <p className="empty-chart-desc">
            As the DQN agent finishes episodes, real total episode rewards will plot here in real-time.
          </p>
        </div>
      </div>
    );
  }

  // Calculate Y-scale metrics
  const rewards = episodes.map((e) => e.reward);
  const minRaw = Math.min(...rewards, 0);
  const maxRaw = Math.max(...rewards, 10);
  
  // Add some padding to bounds
  const minReward = Math.floor(minRaw - 1);
  const maxReward = Math.ceil(maxRaw + 1);
  const rewardRange = maxReward - minReward || 1;

  const chartHeight = 255;
  const paddingTop = 20;
  const paddingBottom = 46;
  const plotInnerHeight = chartHeight - paddingTop - paddingBottom;

  const getY = (val: number) => {
    return paddingTop + plotInnerHeight - ((val - minReward) / rewardRange) * plotInnerHeight;
  };

  // Fixed horizontal spacing: 52px per episode ensures comfortable, uncompressed viewing
  const pointSpacing = 52;
  const svgPaddingX = 35;
  const plotWidth = Math.max(560, (episodes.length - 1) * pointSpacing + svgPaddingX * 2);

  const getX = (index: number) => {
    if (episodes.length === 1) return plotWidth / 2;
    return svgPaddingX + index * pointSpacing;
  };

  // Build SVG path
  const points = episodes.map((ep, idx) => `${getX(idx)},${getY(ep.reward)}`);
  const linePath = `M ${points.join(' L ')}`;
  const areaPath = `${linePath} L ${getX(episodes.length - 1)},${getY(minReward)} L ${getX(0)},${getY(minReward)} Z`;

  // Y-axis ticks (4 ticks)
  const yTicks = [
    maxReward,
    minReward + rewardRange * 0.66,
    minReward + rewardRange * 0.33,
    minReward,
  ];

  return (
    <div className="brutal-chart-card">
      <div className="chart-card-header">
        <div>
          <h3 className="chart-title">Reward Progression</h3>
          <span className="chart-subtitle">
            {episodes.length} {episodes.length === 1 ? 'episode' : 'episodes'} completed (scroll horizontally for history)
          </span>
        </div>
        {historicalBest !== null && (
          <div className="historical-badge" title="Benchmark recorded from historical training log">
            <span>Log Benchmark: </span>
            <strong>{historicalBest.toFixed(1)}</strong>
          </div>
        )}
      </div>

      {/* Main Chart Area with Fixed Sticky Y-Axis and Horizontally Scrollable Plot Area */}
      <div className="chart-viewport-wrapper">
        {/* Fixed Y-Axis Column */}
        <div className="chart-fixed-yaxis" style={{ height: `${chartHeight}px` }}>
          <svg width="60" height={chartHeight} className="yaxis-svg">
            {yTicks.map((val, idx) => (
              <text
                key={idx}
                x="52"
                y={getY(val) + 4}
                textAnchor="end"
                className="axis-tick-text"
              >
                {val.toFixed(1)}
              </text>
            ))}
          </svg>
        </div>

        {/* Horizontally Scrollable Plot Area */}
        <div
          ref={scrollContainerRef}
          className="chart-scrollable-plot"
          style={{ height: `${chartHeight}px` }}
        >
          <svg
            width={plotWidth}
            height={chartHeight}
            className="plot-svg"
          >
            {/* Horizontal Grid lines */}
            {yTicks.map((val, idx) => (
              <line
                key={idx}
                x1="0"
                y1={getY(val)}
                x2={plotWidth}
                y2={getY(val)}
                stroke="#171717"
                strokeWidth="1"
                strokeDasharray="4 4"
                strokeOpacity="0.15"
              />
            ))}

            {/* Zero Baseline (solid near-black line) */}
            {minReward <= 0 && maxReward >= 0 && (
              <line
                x1="0"
                y1={getY(0)}
                x2={plotWidth}
                y2={getY(0)}
                stroke="#171717"
                strokeWidth="2"
                strokeOpacity="0.4"
              />
            )}

            {/* Area Fill */}
            <path d={areaPath} fill="#FFD84D" fillOpacity="0.25" />

            {/* Main High-Contrast Reward Line */}
            <path
              d={linePath}
              fill="none"
              stroke="#171717"
              strokeWidth="3.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Interactive Episode Data Points */}
            {episodes.map((ep, idx) => {
              const cx = getX(idx);
              const cy = getY(ep.reward);
              const isHovered = hoveredPoint?.episode === ep.episode;

              return (
                <g key={ep.episode}>
                  {/* Vertical guide line on hover */}
                  {isHovered && (
                    <line
                      x1={cx}
                      y1={paddingTop}
                      x2={cx}
                      y2={paddingTop + plotInnerHeight}
                      stroke="#171717"
                      strokeWidth="1.5"
                      strokeDasharray="2 2"
                    />
                  )}

                  {/* Circular Point with Thick Brutalist Outline */}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isHovered ? 7 : ep.is_best ? 6 : 5}
                    fill={ep.is_best ? '#FFD84D' : '#48D597'}
                    stroke="#171717"
                    strokeWidth="2.5"
                    className="chart-data-circle"
                    onMouseEnter={() => setHoveredPoint(ep)}
                    onMouseLeave={() => setHoveredPoint(null)}
                  />

                  {/* X-axis Episode Label placed with clear gutter above horizontal scrollbar */}
                  <text
                    x={cx}
                    y={chartHeight - 18}
                    textAnchor="middle"
                    className="axis-tick-text"
                  >
                    #{ep.episode}
                  </text>
                </g>
              );
            })}
          </svg>

          {/* Interactive Tooltip */}
          {hoveredPoint && (
            <div
              className="brutal-chart-tooltip"
              style={{
                left: `${getX(episodes.findIndex((e) => e.episode === hoveredPoint.episode)) + 10}px`,
                top: `${Math.max(10, getY(hoveredPoint.reward) - 65)}px`,
              }}
            >
              <div className="tooltip-title">Episode #{hoveredPoint.episode}</div>
              <div className="tooltip-row">
                <span>Reward:</span>
                <strong className="mono">
                  {hoveredPoint.reward >= 0 ? `+${hoveredPoint.reward.toFixed(2)}` : hoveredPoint.reward.toFixed(2)}
                </strong>
              </div>
              <div className="tooltip-row">
                <span>Pipes Passed:</span>
                <strong>{hoveredPoint.score}</strong>
              </div>
              <div className="tooltip-row">
                <span>Steps:</span>
                <strong>{hoveredPoint.steps}</strong>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
