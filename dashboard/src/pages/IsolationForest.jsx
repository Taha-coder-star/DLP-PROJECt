import { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import KPICard from '../components/KPICard.jsx';
import ConfusionMatrix from '../components/ConfusionMatrix.jsx';
import { ISOLATION_FOREST } from '../data/results.js';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-xl text-sm">
      {label && <p className="text-gray-300 font-medium mb-2">{label}</p>}
      {payload.map((e, i) => (
        <p key={i} style={{ color: e.color || e.fill }}>
          {e.name}: <span className="font-mono">{typeof e.value === 'number' ? e.value.toFixed(3) : e.value}</span>
        </p>
      ))}
    </div>
  );
};

const IF = ISOLATION_FOREST;

export default function IsolationForest() {
  const [granularity, setGranularity] = useState('day');
  const cm = granularity === 'day' ? IF.day : IF.user;

  const topData = IF.topAnomalies.map((a) => ({
    label: `${a.user} · ${a.date}`,
    score: a.score,
  }));

  return (
    <div className="space-y-6">
      {/* Context */}
      <div className="bg-red-950/30 border border-red-700/40 rounded-xl p-4">
        <p className="text-sm font-semibold text-red-400 mb-1">⚠️ Baseline Detector — Not Recommended</p>
        <p className="text-xs text-gray-300">
          The Isolation Forest assigns <em>lower</em> anomaly scores to confirmed insiders than to normal users
          (day-level ROC AUC = 0.38, below chance). This is a known failure mode on datasets where behavioural
          anomalies are interleaved with normal activity. It is retained here as a comparative baseline only.
        </p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <KPICard label="n_estimators"      value="300"    sub="Number of trees"         />
        <KPICard label="Contamination"     value="0.03"   sub="3% outlier rate assumed" />
        <KPICard label="Day-level AUC"     value="0.3819" sub="Below chance (0.5)"       glow="red" delta="↓ Inverted detector" deltaGood={false} />
        <KPICard label="User-level AUC"    value="0.4624" sub="Marginally below chance"  glow="red" delta="↓ Not usable"        deltaGood={false} />
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <KPICard label="Input Features"    value="32"     sub="All feature columns"     />
        <KPICard label="Suspicious (p95)"  value="0.496"  sub="Day-level score"         mono />
        <KPICard label="High-risk (p99)"   value="0.642"  sub="Day-level score"         mono />
        <KPICard label="Suspicious Rows"   value="12,903" sub="Days flagged suspicious"  />
      </div>

      {/* Confusion matrix with toggle */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-5">
          <h3 className="text-sm font-semibold text-gray-300">Confusion Matrix</h3>
          <div className="flex rounded-lg overflow-hidden border border-gray-700">
            {['day', 'user'].map((g) => (
              <button
                key={g}
                onClick={() => setGranularity(g)}
                className={`px-3 py-1.5 text-xs font-medium transition-colors ${
                  granularity === g ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'
                }`}
              >
                {g === 'day' ? 'Day-level (Test)' : 'User-level (All)'}
              </button>
            ))}
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <ConfusionMatrix
            tp={cm.tp} fp={cm.fp} fn={cm.fn} tn={cm.tn}
            title={granularity === 'day'
              ? `Day-level · n=${IF.day.nTotal.toLocaleString()} · ${IF.day.nInsiders} insider-days`
              : `User-level · n=${IF.user.nTotal} · ${IF.user.nInsiders} insiders`}
          />
          <div className="space-y-3">
            <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Metrics</h4>
            {[
              { label: 'ROC AUC',       value: cm.rocAuc.toFixed(4),  bad: cm.rocAuc < 0.5 },
              { label: 'Avg Precision', value: cm.avgPrecision.toFixed(4), bad: true },
              { label: 'Precision',     value: cm.precision.toFixed(4),  bad: true },
              { label: 'Recall',        value: cm.recall.toFixed(4),  bad: false },
              { label: 'F1 Score',      value: cm.f1.toFixed(4),      bad: true },
            ].map(({ label, value, bad }) => (
              <div key={label} className="flex items-center justify-between">
                <span className="text-xs text-gray-400">{label}</span>
                <span className={`font-mono text-sm font-bold ${bad ? 'text-red-400' : 'text-gray-200'}`}>
                  {value}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Top anomalies bar */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-1">Top 10 IF Anomalies (Highest Score Days)</h3>
        <p className="text-xs text-gray-500 mb-4">
          KBP0008 dominates — not a confirmed insider. Illustrates IF's false-positive bias on this dataset.
        </p>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={topData} layout="vertical" margin={{ left: 140, right: 20, top: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
            <XAxis type="number" domain={[0.9, 1.01]} tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="#374151" />
            <YAxis dataKey="label" type="category" tick={{ fill: '#9ca3af', fontSize: 10 }} stroke="#374151" width={140} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="score" name="IF Score" radius={[0, 4, 4, 0]}>
              {topData.map((_, i) => (
                <Cell key={i} fill="#6b7280" />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Why IF fails */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-3">Why Isolation Forest Fails Here</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-gray-400">
          <div className="bg-gray-800 rounded-lg p-3">
            <p className="text-gray-200 font-medium mb-1">No temporal context</p>
            <p>IF treats each day independently. Insider behaviour is a multi-day pattern — a single suspicious day looks like noise.</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-3">
            <p className="text-gray-200 font-medium mb-1">Outlier clustering</p>
            <p>Insiders often cluster together in behaviour space (they do similar things), so IF doesn't isolate them as extreme outliers.</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-3">
            <p className="text-gray-200 font-medium mb-1">Score inversion</p>
            <p>AUC &lt; 0.5 means the model is systematically wrong — insiders score <em>lower</em> than normals. Reversing the scores would give AUC = 0.62, but this trick cannot be justified on real deployments.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
