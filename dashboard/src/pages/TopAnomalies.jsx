import { useState, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import KPICard from '../components/KPICard.jsx';
import { LSTM } from '../data/results.js';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-xl text-sm">
      {label && <p className="text-gray-300 font-medium mb-2">{label}</p>}
      {payload.map((e, i) => (
        <p key={i} style={{ color: e.color || e.fill }}>
          {e.name}: <span className="font-mono">{e.value}</span>
        </p>
      ))}
    </div>
  );
};

const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6', '#10b981', '#ec4899', '#14b8a6', '#f97316'];

export default function TopAnomalies() {
  const [sortBy, setSortBy] = useState('score');
  const [userFilter, setUserFilter] = useState('');

  const anomalies = LSTM.topAnomalies;

  // Per-user aggregation
  const perUser = useMemo(() => {
    const map = {};
    anomalies.forEach((a) => {
      if (!map[a.user]) map[a.user] = { user: a.user, days: 0, maxScore: 0, dates: [] };
      map[a.user].days++;
      map[a.user].dates.push(a.date);
      map[a.user].maxScore = Math.max(map[a.user].maxScore, a.score);
    });
    return Object.values(map).sort((a, b) => b.days - a.days);
  }, [anomalies]);

  const filtered = useMemo(() => {
    if (!userFilter) return anomalies;
    return anomalies.filter((a) => a.user.toLowerCase().includes(userFilter.toLowerCase()));
  }, [anomalies, userFilter]);

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) =>
      sortBy === 'score' ? b.score - a.score : a.date.localeCompare(b.date)
    );
  }, [filtered, sortBy]);

  return (
    <div className="space-y-6">
      {/* KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <KPICard label="Top Anomaly Score" value="1.000"  sub="LSTM reconstruction error"  glow="red"   mono />
        <KPICard label="Unique Users (Top 20)" value={perUser.length} sub="With score = 1.0"   glow="red" />
        <KPICard label="TP @ Top-50"       value="16 / 70" sub="Confirmed insiders caught" glow="green" mono />
        <KPICard label="Precision @ K=20"  value="50.0 %"  sub="Best precision cutoff"     glow="green" />
      </div>

      {/* Per-user bar */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">High-Risk Days per User (Score = 1.0)</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={perUser} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="user" tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="#374151" />
            <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="#374151" allowDecimals={false} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="days" name="High-risk days" radius={[4, 4, 0, 0]}>
              {perUser.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Anomaly timeline */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">
          Anomaly Timeline (Test Split: Mar–Apr 2011)
        </h3>
        <div className="space-y-2">
          {perUser.map((u, ui) => (
            <div key={u.user} className="flex items-center gap-3">
              <span className="font-mono text-xs text-gray-400 w-16 flex-shrink-0">{u.user}</span>
              <div className="flex-1 h-8 bg-gray-800 rounded relative overflow-hidden">
                {u.dates.map((d, di) => {
                  // Map date to horizontal position within Mar 1 - Apr 30 2011
                  const start = new Date('2011-03-01');
                  const end   = new Date('2011-04-30');
                  const date  = new Date(d);
                  const pct   = ((date - start) / (end - start)) * 100;
                  return (
                    <div
                      key={di}
                      className="absolute top-1 bottom-1 w-2 rounded-sm"
                      style={{ left: `${Math.max(0, Math.min(98, pct))}%`, background: COLORS[ui % COLORS.length] }}
                      title={`${u.user} · ${d}`}
                    />
                  );
                })}
              </div>
              <span className="text-xs text-gray-500 w-8 text-right">{u.days}d</span>
            </div>
          ))}
          <div className="flex items-center gap-3 mt-1">
            <span className="w-16" />
            <div className="flex-1 flex justify-between text-xs text-gray-600">
              <span>Mar 1</span><span>Apr 1</span><span>Apr 30</span>
            </div>
          </div>
        </div>
      </div>

      {/* Filterable table */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <div className="flex items-center justify-between gap-4 mb-4 flex-wrap">
          <h3 className="text-sm font-semibold text-gray-300">Anomaly Log</h3>
          <div className="flex gap-3 flex-wrap">
            <input
              type="text"
              placeholder="Filter by user ID…"
              value={userFilter}
              onChange={(e) => setUserFilter(e.target.value)}
              className="bg-gray-800 border border-gray-700 text-gray-300 text-xs rounded-lg px-3 py-1.5 w-40"
            />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-gray-800 border border-gray-700 text-gray-300 text-xs rounded-lg px-2 py-1.5"
            >
              <option value="score">Sort by score</option>
              <option value="date">Sort by date</option>
            </select>
          </div>
        </div>
        <div className="overflow-x-auto max-h-64 overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-gray-900">
              <tr className="border-b border-gray-700">
                {['User ID', 'Date', 'LSTM Score', 'Severity'].map((h) => (
                  <th key={h} className="text-left text-gray-400 font-medium py-2 px-3">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((a, i) => (
                <tr key={i} className="border-b border-gray-800 hover:bg-gray-800/50 transition-colors">
                  <td className="py-2 px-3 font-mono text-blue-400">{a.user}</td>
                  <td className="py-2 px-3 font-mono text-gray-300">{a.date}</td>
                  <td className="py-2 px-3 font-mono text-red-400 font-bold">{a.score.toFixed(3)}</td>
                  <td className="py-2 px-3">
                    <span className="bg-red-900/40 text-red-400 border border-red-700/40 text-xs px-2 py-0.5 rounded-full font-medium">
                      {a.severity}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-gray-600 mt-2">{sorted.length} records shown</p>
      </div>

      {/* Conclusions */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-base font-semibold text-gray-200 mb-4">Conclusions</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            { icon: '❌', color: 'red',   title: 'IF Alone is Insufficient',
              body: 'Isolation Forest day-level AUC = 0.38 — below chance. It systematically scores insiders lower than normal users. Not suitable as a primary detector on this dataset.' },
            { icon: '✅', color: 'green', title: 'LSTM Lifts Detection',
              body: 'LSTM Autoencoder reaches day-level AUC = 0.60 and user-level AUC = 0.56 by modelling temporal behavioural patterns. Captures gradual drift over 7-day windows.' },
            { icon: '🧬', color: 'blue',  title: 'GA Provides Measurable Gain',
              body: 'GA raised F1@50 from 0.250 to 0.267 (+1.67%) and precision from 30% to 32% — catching one additional insider — without retraining any model. 100% flag coverage.' },
            { icon: '📦', color: 'amber', title: 'File Exfil is the Dominant Signal',
              body: 'GA redistributed weight toward file_exfil (30%) and usb_activity (28%), consistent with CERT r4.2 insider behaviour: data hoarding via removable media before departure.' },
            { icon: '⚠️', color: 'gray',  title: 'Limitations',
              body: 'CERT r4.2 uses synthetic randomised content, weakening the DLP signal. 7% base rate makes absolute TP counts small. Future work: attention-based encoders, contextual DLP, recall-focused fitness.' },
            { icon: '🎯', color: 'purple', title: 'Operational Recommendation',
              body: 'Use K=20 for highest precision investigations (50%); expand to K=50 when recall is prioritised. Always combine risk score with XAI explanation before initiating an investigation.' },
          ].map(({ icon, color, title, body }) => (
            <div key={title} className={`rounded-xl border p-4 border-${color}-800/40 bg-${color}-950/20`}>
              <p className={`text-sm font-semibold text-${color}-400 mb-2`}>{icon} {title}</p>
              <p className="text-xs text-gray-400 leading-relaxed">{body}</p>
            </div>
          ))}
        </div>
      </div>

      {/* References */}
      <div className="border-t border-gray-800 pt-4">
        <p className="text-xs text-gray-600 font-semibold mb-2">References</p>
        <div className="space-y-1 text-xs text-gray-600">
          <p>Glasser & Lindauer (2013). <em>Bridging the gap: Generating insider threat data.</em> IEEE S&P Workshops.</p>
          <p>Liu et al. (2008). <em>Isolation Forest.</em> ICDM 2008.</p>
          <p>Hochreiter & Schmidhuber (1997). <em>Long Short-Term Memory.</em> Neural Computation.</p>
          <p>Malhotra et al. (2016). <em>LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection.</em> ICML.</p>
        </div>
      </div>
    </div>
  );
}
