import { useState } from 'react';
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import KPICard from '../components/KPICard.jsx';
import ConfusionMatrix from '../components/ConfusionMatrix.jsx';
import { LSTM, ISOLATION_FOREST } from '../data/results.js';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-xl text-sm">
      {label && <p className="text-gray-300 font-medium mb-2">{label}</p>}
      {payload.map((e, i) => (
        <p key={i} style={{ color: e.color || e.fill }}>
          {e.name}: <span className="font-mono">{typeof e.value === 'number' ? e.value.toFixed(4) : e.value}</span>
        </p>
      ))}
    </div>
  );
};

const AUC_DATA = [
  { level: 'Day-level',  IF: ISOLATION_FOREST.day.rocAuc,  LSTM: LSTM.day.rocAuc },
  { level: 'User-level', IF: ISOLATION_FOREST.user.rocAuc, LSTM: LSTM.user.rocAuc },
];

const SEVERITY_DATA = [
  { name: 'High Risk',   value: LSTM.highRows,                                   color: '#ef4444' },
  { name: 'Suspicious',  value: LSTM.suspiciousRows - LSTM.highRows,              color: '#f59e0b' },
  { name: 'Normal',      value: 326985 - LSTM.suspiciousRows - LSTM.undeterminedRows, color: '#374151' },
];

export default function LSTMAutoencoder() {
  const [threshold, setThreshold] = useState(0.7228);

  const estimatedFlagged = Math.round(
    LSTM.suspiciousRows * Math.max(0, 1 - (threshold - LSTM.params.suspiciousThreshold) / (1 - LSTM.params.suspiciousThreshold))
  );

  return (
    <div className="space-y-6">
      {/* Why LSTM */}
      <div className="bg-green-950/30 border border-green-700/40 rounded-xl p-4">
        <p className="text-sm font-semibold text-green-400 mb-1">🧠 Primary Detector</p>
        <p className="text-xs text-gray-300">
          An LSTM Autoencoder trained <em>only on normal behaviour</em> learns each user's expected temporal structure
          over 7-day windows. Days that deviate produce high reconstruction error — making the model sensitive to
          gradual behavioural drift characteristic of insider planning and execution phases.
        </p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
        <KPICard label="Window Size"    value="7 days"  sub="Sliding window per user"  />
        <KPICard label="Hidden / Latent" value="32 / 16" sub="Encoder dimensions"      mono />
        <KPICard label="Epochs"         value="20"      sub="Adam, lr = 1e-3"          />
        <KPICard label="Day AUC"        value="0.6032"  sub="+0.22 vs IF"              glow="green" delta="↑ vs IF 0.38" />
        <KPICard label="User AUC"       value="0.5618"  sub="+0.10 vs IF"              glow="green" delta="↑ vs IF 0.46" />
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <KPICard label="Suspicious (p95)" value="0.7228" sub="Day-level threshold"    mono />
        <KPICard label="High-risk (p99)"  value="0.9288" sub="Day-level threshold"    mono />
        <KPICard label="Suspicious Rows"  value="12,936" sub="User-days flagged"      />
        <KPICard label="High-risk Rows"   value="4,945"  sub="Max anomaly days"        glow="red" />
      </div>

      {/* AUC comparison */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">ROC AUC: Isolation Forest vs LSTM</h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={AUC_DATA} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="level" tick={{ fill: '#9ca3af', fontSize: 12 }} stroke="#374151" />
            <YAxis domain={[0, 1]} tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="#374151" />
            <Tooltip content={<CustomTooltip />} />
            <Legend formatter={(v) => <span className="text-gray-300 text-xs">{v}</span>} />
            {/* Reference line at 0.5 */}
            <Bar dataKey="IF"   name="Isolation Forest" fill="#6b7280" radius={[4, 4, 0, 0]} />
            <Bar dataKey="LSTM" name="LSTM Autoencoder"  fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-500 mt-2">
          Reference line: AUC = 0.5 (random chance). LSTM consistently above chance at both granularities.
        </p>
      </div>

      {/* Confusion matrix + metrics */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">Day-level Confusion Matrix (Test Split)</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <ConfusionMatrix
            tp={LSTM.day.tp} fp={LSTM.day.fp} fn={LSTM.day.fn} tn={LSTM.day.tn}
            title={`Day-level · ${LSTM.day.nTotal.toLocaleString()} test-period days · ${LSTM.day.nInsiders.toLocaleString()} insider-days`}
          />
          <div className="space-y-3">
            <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Day-level Metrics</h4>
            {[
              { label: 'ROC AUC',        value: LSTM.day.rocAuc,        good: true },
              { label: 'Avg Precision',  value: LSTM.day.avgPrecision,  good: false },
              { label: 'Precision',      value: LSTM.day.precision,     good: false },
              { label: 'Recall',         value: LSTM.day.recall,        good: true },
              { label: 'F1 Score',       value: LSTM.day.f1,            good: false },
            ].map(({ label, value, good }) => (
              <div key={label} className="flex items-center justify-between">
                <span className="text-xs text-gray-400">{label}</span>
                <span className={`font-mono text-sm font-bold ${good ? 'text-green-400' : 'text-gray-200'}`}>
                  {value.toFixed(4)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Interactive threshold */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-gray-300">Threshold Sensitivity</h3>
          <span className="font-mono text-blue-400 text-sm">{threshold.toFixed(4)}</span>
        </div>
        <input
          type="range" min={0.1} max={1.0} step={0.001}
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          className="w-full mb-4"
        />
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="bg-gray-800 rounded-lg p-3">
            <p className="text-xl font-bold font-mono text-amber-400">{threshold.toFixed(4)}</p>
            <p className="text-xs text-gray-500">Selected threshold</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-3">
            <p className="text-xl font-bold font-mono text-blue-400">{estimatedFlagged.toLocaleString()}</p>
            <p className="text-xs text-gray-500">Est. rows above threshold</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-3">
            <p className="text-xs text-gray-400 mb-1 font-medium">Reference points</p>
            <p className="text-xs text-gray-500">p95 = 0.7228 (suspicious)</p>
            <p className="text-xs text-gray-500">p99 = 0.9288 (high-risk)</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Severity donut */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-gray-300 mb-4">Anomaly Severity Distribution</h3>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={SEVERITY_DATA} cx="50%" cy="50%" innerRadius={55} outerRadius={85} dataKey="value" paddingAngle={3}>
                {SEVERITY_DATA.map((d, i) => <Cell key={i} fill={d.color} />)}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend formatter={(v) => <span className="text-gray-300 text-xs">{v}</span>} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Architecture */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-gray-300 mb-4">Model Architecture</h3>
          <div className="flex flex-col gap-2">
            {[
              { label: 'Input',         dim: '7 × 32',   color: '#3b82f6', desc: '7-day window, 32 features' },
              { label: 'LSTM Encoder',  dim: 'hidden=32', color: '#8b5cf6', desc: 'Learns temporal patterns' },
              { label: 'Latent Space',  dim: '16-dim',    color: '#10b981', desc: 'Compressed representation' },
              { label: 'LSTM Decoder',  dim: 'hidden=32', color: '#8b5cf6', desc: 'Reconstructs sequence' },
              { label: 'Output',        dim: '7 × 32',   color: '#3b82f6', desc: 'Reconstructed window' },
              { label: 'Recon. Error',  dim: 'MSE → [0,1]', color: '#ef4444', desc: 'Anomaly score' },
            ].map((layer, i) => (
              <div key={i} className="flex items-center gap-3">
                <div
                  className="flex-1 rounded-lg px-3 py-2 flex items-center justify-between border"
                  style={{ borderColor: layer.color + '44', background: layer.color + '15' }}
                >
                  <span className="text-xs font-semibold text-white">{layer.label}</span>
                  <span className="font-mono text-xs text-gray-400">{layer.dim}</span>
                </div>
                {i < 5 && <span className="text-gray-600 text-sm">↓</span>}
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-3">
            Trained on normal-only data; anomalies produce high reconstruction error.
          </p>
        </div>
      </div>
    </div>
  );
}
