import React from 'react';
import { History, Award, CheckCircle } from 'lucide-react';
import type { EpisodeRecord } from '../types';

interface RecentEpisodesProps {
  episodes: EpisodeRecord[];
}

export const RecentEpisodes: React.FC<RecentEpisodesProps> = ({ episodes }) => {
  // Newest episodes appear first; preserve all recorded rows
  const reversedList = [...episodes].reverse();

  return (
    <div className="brutal-episodes-card">
      <div className="episodes-card-header">
        <div className="header-title-group">
          <History size={18} />
          <h3 className="episodes-title">Recent Episodes</h3>
        </div>
        <span className="episodes-count-badge">{episodes.length} recorded</span>
      </div>

      {reversedList.length === 0 ? (
        <div className="episodes-empty-box">
          <p>No episodes recorded in this session. Click Start to begin running the agent.</p>
        </div>
      ) : (
        <div className="episodes-scroll-container">
          <table className="brutal-table">
            <thead>
              <tr>
                <th>Episode</th>
                <th>Score</th>
                <th>Reward</th>
                <th>Steps</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {reversedList.map((ep) => (
                <tr key={ep.episode} className={ep.is_best ? 'table-row-best' : ''}>
                  <td className="mono font-bold">#{ep.episode}</td>
                  <td>
                    <span className="brutal-score-pill">{ep.score}</span>
                  </td>
                  <td className="mono font-semibold">
                    <span className={ep.reward >= 0 ? 'text-positive-reward' : 'text-negative-reward'}>
                      {ep.reward >= 0 ? `+${ep.reward.toFixed(2)}` : ep.reward.toFixed(2)}
                    </span>
                  </td>
                  <td className="mono text-steps">{ep.steps}</td>
                  <td>
                    {ep.is_best ? (
                      <span className="brutal-badge-best">
                        <Award size={13} />
                        <span>Best</span>
                      </span>
                    ) : (
                      <span className="brutal-badge-completed">
                        <CheckCircle size={13} />
                        <span>Completed</span>
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
