import { useState } from 'react';
import {
  LineChart, Line, BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Cell,
} from 'recharts';
import KPICard from '../components/KPICard.jsx';
import { GA, RISK_SIGNALS, EVALUATION_RESULTS, K_SWEEP } from '../data/results.js';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-xl text-sm">
      {label !== undefined && <p className="text-gray-300 font-medium mb-2">{label}</p>}
      {payload.map((e, i) => (
        <p key={i} style={{ color: e.color || e.stroke }}>
          {e.name}: <span className="font-mono">{typeof e.value === 'number' ? e.value.toFixed(4) : e.value}</span>
        </p>
      ))}
    </div>
  );
};

const WEIGHT_DATA = RISK_SIGNALS.map((s) => ({
  signal: s.label,
  Baseline: s.defaultWeight,
  'GA-Optimised': s.gaWeight,
  color: s.color,
}));

const RADAR_DATA = RISK_SIGNALS.map((s) => ({
  signal: s.label,
  Baseline: +(s.defaultWeight * 100).toFixed(1),
  GA: +(s.gaWeight * 100).toFixed(1),
}));

const THRESHOLD_DATA = RISK_SIGNALS.filter((s) => s.key !== 'content_sensitivity').map((s) => ({
  signal: s.label,
  delta: +(s.gaThreshold - s.defaultThreshold).toFixed(4),
  ga: s.gaThreshold,
  baseline: s.defaultThreshold,
  color: s.gaThreshold > s.defaultThreshold ? '#ef4444' : '#10b981',
}));

const LIFT_DATA = [
  { metric: 'F1',         Baseline: GA.baselineMetrics.f1,        GA: GA.metrics.f1 },
  { metric: 'Precision',  Baseline: GA.baselineMetrics.precision,  GA: GA.metrics.precision },
  { metric: 'Recall',     Baseline: GA.baselineMetrics.recall,     GA: GA.metrics.recall },
];

export default function RiskScorerGA() {
  const [showBaseline, setShowBaseline] = useState(true);
  const [view, setView] = useState('weights');

  return (
    <div className="space-y-6">
      {/* Intro */}
      <div className="bg-purple-950/30 border border-purple-700/40 rounded-xl p-4">
        <p className="text-sm font-semibold text-purple-400 mb-1">🧬 Genetic Algorithm Weight Optimisation</p>
        <p className="text-xs text-gray-300">
          The GA evolves signal weights and flag thresholds over up to 100 generations without retraining any model.
          Fitness function: <code className="bg-gray-800 px-1 rounded">0.8 × F1@50 + 0.2 × flag_coverage@50</code>.
          Seeded from domain-knowledge defaults — refines rather than searches from scratch.
        </p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <KPICard label="Population × Gens"  value="60 × 100" sub="Max search budget"       mono />
        <KPICard label="Converged At Gen"    value="26"       sub="Early-stop patience = 25" glow="green" />
        <KPICard label="Best Fitness"        value="0.41336"  sub="0.8·F1 + 0.2·Coverage"   glow="green" mono />
        <KPICard label="Flag Coverage @50"   value="100 %"    sub="Every alert explained"    glow="green" />
      </div>

      {/* Convergence chart */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-gray-300">GA Convergence (26 Generations)</h3>
          <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
            <input type="checkbox" checked={showBaseline} onChange={(e) => setShowBaseline(e.target.checked)}
              className="accent-blue-500" />
            Show mean fitness
          </label>
        </div>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={GA.convergenceHistory} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="gen" tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="#374151" label={{ value: 'Generation', fill: '#6b7280', fontSize: 11, position: 'insideBottom', offset: -2 }} />
            <YAxis domain={[0.30, 0.43]} tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="#374151" />
            <Tooltip content={<CustomTooltip />} />
            <Legend formatter={(v) => <span className="text-gray-300 text-xs">{v}</span>} />
            <ReferenceLine x={26} stroke="#f59e0b" strokeDasharray="4 4"
              label={{ value: 'Early stop', fill: '#f59e0b', fontSize: 10, position: 'top' }} />
            <Line type="monotone" dataKey="best" name="Best fitness"  stroke="#10b981" strokeWidth={2.5} dot={false} />
            {showBaseline && (
              <Line type="monotone" dataKey="mean" name="Mean fitness" stroke="#3b82f6" strokeWidth={1.5} strokeDasharray="5 5" dot={false} />
            )}
          </LineChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-500 mt-2">
          Best fitness locked at 0.41336 from generation 1 — the warm-start seed already near-optimal;
          GA confirms and slightly refines the domain-knowledge baseline.
        </p>
      </div>

      {/* Weights / Radar / Thresholds */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-gray-300">Signal Weight Analysis</h3>
          <div className="flex rounded-lg overflow-hidden border border-gray-700">
            {['weights', 'radar', 'thresholds'].map((v) => (
              <button key={v} onClick={() => setView(v)}
                className={`px-3 py-1.5 text-xs capitalize transition-colors ${view === v ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'}`}>
                {v}
              </button>
            ))}
          </div>
        </div>

        {view === 'weights' && (
          <>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={WEIGHT_DATA} margin={{ top: 5, right: 20, left: 0, bottom: 35 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="signal" tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="#374151"
                  angle={-30} textAnchor="end" interval={0} />
                <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="#374151" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip content={<CustomTooltip />} formatter={(v) => (v * 100).toFixed(1) + '%'} />
                <Legend formatter={(v) => <span className="text-gray-300 text-xs">{v}</span>} />
                <Bar dataKey="Baseline"     fill="#6b7280" radius={[4, 4, 0, 0]} />
                <Bar dataKey="GA-Optimised" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <p className="text-xs text-gray-500 mt-2">
              GA shifted weight from <code className="bg-gray-800 px-1 rounded">lstm_p95</code> (45%→30%)
              toward <code className="bg-gray-800 px-1 rounded">file_exfil</code> (9%→30%) and{' '}
              <code className="bg-gray-800 px-1 rounded">usb_activity</code> (9%→28%).
            </p>
          </>
        )}

        {view === 'radar' && (
          <ResponsiveContainer width="100%" height={320}>
            <RadarChart data={RADAR_DATA} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
              <PolarGrid stroke="#374151" />
              <PolarAngleAxis dataKey="signal" tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <PolarRadiusAxis angle={30} domain={[0, 50]} tick={{ fill: '#6b7280', fontSize: 9 }} tickFormatter={(v) => `${v}%`} />
              <Tooltip content={<CustomTooltip />} formatter={(v) => `${v}%`} />
              <Legend formatter={(v) => <span className="text-gray-300 text-xs">{v}</span>} />
              <Radar name="Baseline"     dataKey="Baseline" stroke="#6b7280" fill="#6b7280" fillOpacity={0.3} />
              <Radar name="GA-Optimised" dataKey="GA"       stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
            </RadarChart>
          </ResponsiveContainer>
        )}

        {view === 'thresholds' && (
          <>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={THRESHOLD_DATA} layout="vertical" margin={{ left: 100, right: 50, top: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
                <XAxis type="number" tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="#374151"
                  label={{ value: 'GA threshold − Baseline threshold', fill: '#6b7280', fontSize: 10, position: 'insideBottom', offset: -2 }} />
                <YAxis dataKey="signal" type="category" tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="#374151" width={100} />
                <Tooltip content={<CustomTooltip />} formatter={(v, n, p) => [`${p.payload.ga.toFixed(3)} (was ${p.payload.baseline.toFixed(3)})`, 'GA threshold']} />
                <ReferenceLine x={0} stroke="#6b7280" strokeWidth={1} />
                <Bar dataKey="delta" name="Threshold Δ" radius={[0, 4, 4, 0]}>
                  {THRESHOLD_DATA.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="text-xs text-gray-500 mt-2">
              Positive = GA set a stricter threshold. <code className="bg-gray-800 px-1 rounded">after_hours</code> threshold raised to 0.95 — de-emphasised almost entirely.
            </p>
          </>
        )}
      </div>

      {/* Performance lift */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-gray-300 mb-4">Performance Lift @ K=50</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={LIFT_DATA} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="metric" tick={{ fill: '#9ca3af', fontSize: 12 }} stroke="#374151" />
              <YAxis domain={[0, 0.6]} tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="#374151" />
              <Tooltip content={<CustomTooltip />} />
              <Legend formatter={(v) => <span className="text-gray-300 text-xs">{v}</span>} />
              <Bar dataKey="Baseline"     fill="#6b7280" radius={[4, 4, 0, 0]} />
              <Bar dataKey="GA"           fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-3 gap-2 mt-3">
            {[
              { label: 'ΔF1', value: '+0.0167', col: 'text-green-400' },
              { label: 'ΔPrec', value: '+0.0200', col: 'text-green-400' },
              { label: 'ΔRecall', value: '+0.0143', col: 'text-green-400' },
            ].map(({ label, value, col }) => (
              <div key={label} className="bg-gray-800 rounded-lg p-2 text-center">
                <p className={`font-mono font-bold text-sm ${col}`}>{value}</p>
                <p className="text-xs text-gray-500">{label}</p>
              </div>
            ))}
          </div>
        </div>

        {/* K-sweep */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-gray-300 mb-4">P / R / F1 vs K (GA Weights)</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={K_SWEEP} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="k" tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="#374151" />
              <YAxis domain={[0, 0.6]} tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="#374151" />
              <Tooltip content={<CustomTooltip />} />
              <Legend formatter={(v) => <span className="text-gray-300 text-xs">{v}</span>} />
              <Line type="monotone" dataKey="precision" name="Precision" stroke="#ef4444" strokeWidth={2} dot={{ fill: '#ef4444', r: 4 }} />
              <Line type="monotone" dataKey="recall"    name="Recall"    stroke="#3b82f6" strokeWidth={2} dot={{ fill: '#3b82f6', r: 4 }} />
              <Line type="monotone" dataKey="f1"        name="F1"        stroke="#10b981" strokeWidth={2.5} dot={{ fill: '#10b981', r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 mt-2">
            K=20 gives 50% precision (highest). K=50 gives best F1 (0.267). Trade-off: precision vs recall.
          </p>
        </div>
      </div>

      {/* Full evaluation table */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">Full Evaluation Results</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-700">
                {['Configuration', 'Aggregation', 'K', 'Precision', 'Recall', 'F1', 'TP / 70'].map((h) => (
                  <th key={h} className="text-left text-gray-400 font-medium py-2 px-3">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {EVALUATION_RESULTS.map((r, i) => (
                <tr key={i} className={`border-b border-gray-800 ${r.best ? 'bg-green-950/20' : ''}`}>
                  <td className={`py-2 px-3 font-medium ${r.best ? 'text-green-400' : 'text-gray-300'}`}>
                    {r.best && '★ '}{r.config}
                  </td>
                  <td className="py-2 px-3 text-gray-400 font-mono">{r.agg}</td>
                  <td className="py-2 px-3 text-gray-300 font-mono">{r.k}</td>
                  <td className="py-2 px-3 text-gray-300 font-mono">{r.precision != null ? r.precision.toFixed(3) : '—'}</td>
                  <td className="py-2 px-3 text-gray-300 font-mono">{r.recall  != null ? r.recall.toFixed(4)   : '—'}</td>
                  <td className={`py-2 px-3 font-mono font-bold ${r.best ? 'text-green-400' : r.f1 != null ? 'text-gray-200' : 'text-red-400'}`}>
                    {r.f1 != null ? r.f1.toFixed(4) : '< baseline'}
                  </td>
                  <td className="py-2 px-3 text-gray-300 font-mono">{r.tp != null ? r.tp : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* GA hyperparameters */}
      <details className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <summary className="px-5 py-4 text-sm font-semibold text-gray-300 cursor-pointer hover:text-white">
          GA Hyperparameters ▾
        </summary>
        <div className="px-5 pb-5">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {Object.entries(GA.params).map(([k, v]) => (
              <div key={k} className="bg-gray-800 rounded-lg p-3">
                <p className="text-xs text-gray-500 font-mono">{k}</p>
                <p className="text-sm font-bold text-white font-mono">{v}</p>
              </div>
            ))}
          </div>
        </div>
      </details>
    </div>
  );
}
