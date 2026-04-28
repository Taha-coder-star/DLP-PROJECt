import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { DLP_TIERS } from '../data/results.js';

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

const WORKED_EXAMPLE = [
  { event: 'Email w/ large attach.',  tier: 'INTERNAL',   score: 1 },
  { event: 'File: salary_Q4.xlsx',    tier: 'SENSITIVE',  score: 2 },
  { event: 'USB write: resume.pdf',   tier: 'SENSITIVE',  score: 2 },
  { event: 'USB write: passwords.txt', tier: 'RESTRICTED', score: 3 },
];

export default function DLPSensitivity() {
  const [selected, setSelected] = useState(null);

  const selectedTier = DLP_TIERS.find((t) => t.label === selected);
  const avgScore = selected
    ? (selectedTier?.score ?? 0)
    : (DLP_TIERS.reduce((s, t) => s + t.score, 0) / DLP_TIERS.length).toFixed(2);

  return (
    <div className="space-y-6">
      {/* Intro */}
      <div className="bg-teal-950/30 border border-teal-700/40 rounded-xl p-4">
        <p className="text-sm font-semibold text-teal-400 mb-1">🛡️ Rule-based DLP Content Sensitivity Classifier</p>
        <p className="text-xs text-gray-300">
          A lightweight, fully vectorised keyword and structural classifier applied to email and file events.
          No language model required — uses keyword matching, file extension tables, magic-byte detection, and
          activity type + destination logic. Output feeds into the risk scorer as the{' '}
          <code className="bg-gray-800 px-1 rounded">content_sensitivity</code> signal (weight = 10%).
        </p>
      </div>

      {/* Tier cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {DLP_TIERS.map((tier) => (
          <button
            key={tier.label}
            onClick={() => setSelected(selected === tier.label ? null : tier.label)}
            className={`text-left rounded-xl border-2 p-4 transition-all ${
              selected === tier.label ? 'opacity-100 scale-[1.02]' : 'opacity-80 hover:opacity-100'
            }`}
            style={{
              borderColor: tier.color,
              background: tier.color + (selected === tier.label ? '30' : '15'),
            }}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold" style={{ color: tier.color }}>{tier.label}</span>
              <span className="font-mono text-2xl font-bold text-white">{tier.score}</span>
            </div>
            <p className="text-xs text-gray-400 leading-relaxed">{tier.signals}</p>
          </button>
        ))}
      </div>

      {/* Tier scoring bar */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">Tier Score Weighting</h3>
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={DLP_TIERS} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="label" tick={{ fill: '#9ca3af', fontSize: 12 }} stroke="#374151" />
            <YAxis domain={[0, 3.5]} tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="#374151" />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="score" name="Sensitivity Score" radius={[4, 4, 0, 0]}>
              {DLP_TIERS.map((d, i) => <Cell key={i} fill={d.color} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Worked example */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-2">Worked Example — Single User-Day</h3>
        <p className="text-xs text-gray-500 mb-4">
          Events on a single day are classified individually; the maximum daily sensitivity score feeds into the risk scorer.
        </p>
        <div className="space-y-2">
          {WORKED_EXAMPLE.map((e, i) => {
            const tier = DLP_TIERS.find((t) => t.label === e.tier);
            return (
              <div key={i} className="flex items-center gap-3 bg-gray-800 rounded-lg px-4 py-2.5">
                <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: tier.color }} />
                <span className="text-xs text-gray-300 flex-1 font-mono">{e.event}</span>
                <span
                  className="text-xs font-bold px-2 py-0.5 rounded"
                  style={{ color: tier.color, background: tier.color + '20' }}
                >
                  {e.tier}
                </span>
                <span className="font-mono text-sm font-bold text-white w-4 text-right">{e.score}</span>
              </div>
            );
          })}
        </div>
        <div className="flex items-center justify-between mt-3 pt-3 border-t border-gray-700">
          <span className="text-xs text-gray-400">Daily max sensitivity score (→ risk scorer):</span>
          <span className="font-mono font-bold text-red-400 text-sm">3 (RESTRICTED)</span>
        </div>
      </div>

      {/* How it works */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          {
            title: 'Email Events',
            color: '#3b82f6',
            points: [
              'Keyword matching on subject + content',
              'Large attachment bonus (+1 tier)',
              'BCC to external domain (+sensitivity)',
              'After-hours timing considered',
            ],
          },
          {
            title: 'File Events',
            color: '#f59e0b',
            points: [
              'File extension tier table (exe, pdf, zip…)',
              'Magic-byte detection from hex content prefix',
              'Write to removable media → upgrade tier',
              'filename keyword matching',
            ],
          },
          {
            title: 'Limitations (Synthetic Data)',
            color: '#6b7280',
            points: [
              'CERT r4.2 email content is randomised',
              'Filenames are auto-generated — no keywords',
              'DLP signal is weak on this dataset',
              'Would be far more informative on real enterprise data',
            ],
          },
        ].map(({ title, color, points }) => (
          <div key={title} className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <h4 className="text-xs font-bold mb-3" style={{ color }}>{title}</h4>
            <ul className="space-y-1.5">
              {points.map((p) => (
                <li key={p} className="text-xs text-gray-400 flex items-start gap-1.5">
                  <span className="mt-0.5 flex-shrink-0" style={{ color }}>•</span>
                  {p}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}
