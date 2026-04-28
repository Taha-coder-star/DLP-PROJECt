import { useState } from 'react';
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import KPICard from '../components/KPICard.jsx';
import { DATASET, FEATURE_CATEGORIES } from '../data/results.js';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg p-3 shadow-xl text-sm" style={{ background: '#1e293b', border: '1px solid hsl(220 20% 22%)' }}>
      {label && <p className="text-gray-300 font-medium mb-2">{label}</p>}
      {payload.map((e, i) => (
        <p key={i} style={{ color: e.color || e.fill }}>
          {e.name}: <span className="font-mono">{e.value?.toLocaleString()}</span>
        </p>
      ))}
    </div>
  );
};

const SPLIT_DATA = [
  { name: 'Train Split', value: DATASET.trainRows, color: '#6366f1' },
  { name: 'Test Split',  value: DATASET.testRows,  color: '#8b5cf6' },
];

const CLASS_DATA = [
  { name: 'Normal Users', value: DATASET.users - DATASET.insiders, color: '#6366f1' },
  { name: 'Insider Users', value: DATASET.insiders, color: '#ef4444' },
];

export default function Dataset() {
  const [highlighted, setHighlighted] = useState('all');

  const featureData = FEATURE_CATEGORIES.map((c) => ({
    ...c,
    opacity: highlighted === 'all' || highlighted === c.category ? 1 : 0.25,
  }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
        <KPICard label="Total Rows" value="326,985" sub="User-day records" glow="blue" />
        <KPICard label="Train Rows" value="261,220" sub="Through 2011-02-05" />
        <KPICard label="Test Rows" value="65,765" sub="From 2011-02-06" />
        <KPICard label="Users" value="1,000" sub="Unique employees" />
        <KPICard label="Insiders" value="70" sub="Confirmed (7%)" glow="red" />
      </div>

      <div className="card p-4">
        <h3 className="section-header text-sm font-semibold text-gray-300">Dataset Timeline</h3>
        <div className="relative h-12 rounded-lg overflow-hidden flex" style={{ background: 'hsl(220 20% 10%)' }}>
          <div className="flex items-center justify-center text-xs font-medium text-white"
            style={{ width: `${(261220 / 326985) * 100}%`, background: 'rgba(99,102,241,0.6)' }}>
            Train (2010-01-02 to 2011-02-05)
          </div>
          <div className="flex items-center justify-center text-xs font-medium text-white"
            style={{ width: `${(65765 / 326985) * 100}%`, background: 'rgba(139,92,246,0.6)' }}>
            Test (2011-02-06 to 2011-05-16)
          </div>
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1.5">
          <span>261,220 rows (79.9%)</span><span>65,765 rows (20.1%)</span>
        </div>
        <p className="text-xs text-gray-500 mt-2">Chronological split — no temporal leakage.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card p-4">
          <h3 className="section-header text-sm font-semibold text-gray-300">Train / Test Split</h3>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={SPLIT_DATA} cx="50%" cy="50%" innerRadius={60} outerRadius={90} dataKey="value" paddingAngle={3}>
                {SPLIT_DATA.map((d, i) => <Cell key={i} fill={d.color} />)}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend formatter={(v) => <span className="text-gray-300 text-xs">{v}</span>} />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="card p-4">
          <h3 className="section-header text-sm font-semibold text-gray-300">Class Imbalance</h3>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={CLASS_DATA} cx="50%" cy="50%" innerRadius={60} outerRadius={90} dataKey="value" paddingAngle={3}>
                {CLASS_DATA.map((d, i) => <Cell key={i} fill={d.color} />)}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend formatter={(v) => <span className="text-gray-300 text-xs">{v}</span>} />
            </PieChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 text-center">70 insiders out of 1,000 users (7% positive rate)</p>
        </div>
      </div>

      <div className="card p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="section-header text-sm font-semibold text-gray-300 mb-0">32 Features by Category</h3>
          <select value={highlighted} onChange={(e) => setHighlighted(e.target.value)}
            className="text-gray-300 text-xs rounded-lg px-2 py-1"
            style={{ background: 'hsl(220 20% 12%)', border: '1px solid hsl(220 20% 18%)' }}>
            <option value="all">All categories</option>
            {FEATURE_CATEGORIES.map((c) => <option key={c.category} value={c.category}>{c.category}</option>)}
          </select>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={featureData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="category" tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="#1e293b" />
            <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="#1e293b" />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="count" name="Feature count" radius={[4, 4, 0, 0]}>
              {featureData.map((d, i) => <Cell key={i} fill={d.color} fillOpacity={d.opacity} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        {highlighted !== 'all' && (
          <div className="mt-3 p-3 rounded-lg" style={{ background: 'hsl(220 20% 10%)' }}>
            <p className="text-xs text-gray-400 mb-2 font-medium">{highlighted} features:</p>
            <div className="flex flex-wrap gap-1.5">
              {FEATURE_CATEGORIES.find((c) => c.category === highlighted)?.features.map((f) => (
                <span key={f} className="text-gray-300 text-xs px-2 py-0.5 rounded font-mono"
                  style={{ background: 'hsl(220 20% 14%)' }}>{f}</span>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="card p-4" style={{ borderLeft: '3px solid #6366f1' }}>
        <p className="text-xs text-indigo-300 leading-relaxed">
          <strong>Why chronological split?</strong> Splitting by date prevents the model from "seeing the future" —
          any threshold calibrated on test dates would leak information and inflate reported metrics.
        </p>
      </div>
    </div>
  );
}
