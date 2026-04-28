import {
  FolderOpen, Wrench, Settings, TreePine, Brain, Shield,
  Scale, Dna, BarChart3, ChevronRight,
} from 'lucide-react';
import KPICard from '../components/KPICard.jsx';
import { DATASET, GA, LSTM } from '../data/results.js';

const PIPELINE_ICONS = {
  'Raw CERT Logs': FolderOpen,
  'Data Cleaning': Wrench,
  'Feature Engineering': Settings,
  'Isolation Forest': TreePine,
  'LSTM Autoencoder': Brain,
  'DLP Classifier': Shield,
  'Risk Scorer': Scale,
  'GA Optimiser': Dna,
  'Ranked Alert Queue': BarChart3,
};

const PIPELINE_STEPS = [
  { label: 'Raw CERT Logs', sub: 'email, logon, device, file, psychometric', color: '#6b7280' },
  { label: 'Data Cleaning', sub: 'Chunked merge — 326,985 user-day rows', color: '#8b5cf6' },
  { label: 'Feature Engineering', sub: '32 features per user-day', color: '#6366f1' },
  { label: 'Isolation Forest', sub: 'Baseline (AUC 0.38 — inverted)', color: '#6b7280' },
  { label: 'LSTM Autoencoder', sub: 'Temporal anomaly (AUC 0.60)', color: '#10b981' },
  { label: 'DLP Classifier', sub: 'PUBLIC to RESTRICTED tiers', color: '#14b8a6' },
  { label: 'Risk Scorer', sub: '7-signal weighted aggregation', color: '#f59e0b' },
  { label: 'GA Optimiser', sub: 'Evolves weights, F1@50 = 0.267', color: '#ef4444' },
  { label: 'Ranked Alert Queue', sub: '16 / 70 insiders flagged @ Top-50', color: '#22c55e' },
];

export default function Overview() {
  return (
    <div className="space-y-8">
      {/* Hero */}
      <div
        className="rounded-2xl p-6 border"
        style={{
          background: 'radial-gradient(ellipse at 20% 50%, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 50%, transparent 80%)',
          borderColor: 'rgba(99,102,241,0.2)',
        }}
      >
        <div className="flex flex-wrap gap-2.5 mb-3">
          <span className="text-xs px-2.5 py-0.5 rounded-full font-medium"
            style={{ background: 'rgba(99,102,241,0.15)', color: '#818cf8', border: '1px solid rgba(99,102,241,0.25)' }}>
            University Project
          </span>
          <span className="text-xs px-2.5 py-0.5 rounded-full font-medium"
            style={{ background: 'rgba(139,92,246,0.15)', color: '#a78bfa', border: '1px solid rgba(139,92,246,0.25)' }}>
            CERT r4.2
          </span>
          <span className="text-xs px-2.5 py-0.5 rounded-full font-medium"
            style={{ background: 'rgba(16,185,129,0.15)', color: '#34d399', border: '1px solid rgba(16,185,129,0.25)' }}>
            GA-Optimised
          </span>
        </div>
        <h1 className="text-2xl font-bold text-white mb-1.5 tracking-tight">
          AI-Driven UEBA + DLP Risk Scoring System
        </h1>
        <p className="text-gray-400 text-sm max-w-3xl leading-relaxed">
          Detects insider threats by fusing an <strong className="text-indigo-300">LSTM Autoencoder</strong>, rule-based{' '}
          <strong className="text-teal-300">DLP content sensitivity</strong>, five behavioural signals, and a{' '}
          <strong className="text-purple-300">Genetic Algorithm</strong> that evolves signal weights to maximise F1
          — without retraining any model.
        </p>
      </div>

      {/* KPI strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <KPICard label="Users Analysed"      value="1,000"   sub="CERT r4.2 employees"      glow="blue" />
        <KPICard label="User-day Records"    value="326,985" sub="32 features each"          />
        <KPICard label="Confirmed Insiders"  value="70"      sub="Ground-truth labels"        glow="red" deltaGood={false} />
        <KPICard label="Best F1 @ Top-50"    value="0.267"   sub="GA-optimised LSTM weights"  glow="green"
                 delta="+1.67% vs domain baseline" />
      </div>

      {/* Additional KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <KPICard label="Insiders Flagged"  value="16 / 70" sub="Top-50 investigation queue" mono />
        <KPICard label="Precision @ K=50" value="32.0 %"   sub="1 in 3 flagged is real"     />
        <KPICard label="Precision @ K=20" value="50.0 %"   sub="1 in 2 flagged is real"     glow="green" />
        <KPICard label="Flag Coverage"    value="100 %"    sub="All top-50 have explanation"  />
      </div>

      {/* Pipeline diagram */}
      <div>
        <h2 className="section-header text-sm font-semibold text-gray-200">System Pipeline</h2>
        <div className="card p-6">
          <div className="flex flex-wrap items-center gap-2 justify-center">
            {PIPELINE_STEPS.map((step, i) => {
              const Icon = PIPELINE_ICONS[step.label];
              return (
                <div key={i} className="flex items-center gap-2">
                  <div
                    className="flex flex-col items-center text-center rounded-xl border px-3 py-2.5 min-w-[110px] transition-all duration-150 hover:scale-[1.03]"
                    style={{ borderColor: step.color + '44', background: step.color + '10' }}
                  >
                    {Icon && <Icon size={18} style={{ color: step.color }} className="mb-1" />}
                    <span className="text-xs font-semibold text-white leading-tight">{step.label}</span>
                    <span className="text-xs text-gray-500 mt-0.5 leading-tight">{step.sub}</span>
                  </div>
                  {i < PIPELINE_STEPS.length - 1 && (
                    <ChevronRight size={16} className="text-gray-600 flex-shrink-0" />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Problem / Approach / Outcome */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card p-4" style={{ borderLeft: '3px solid #ef4444' }}>
          <h3 className="text-sm font-semibold text-red-400 mb-3">The Problem</h3>
          <ul className="space-y-2 text-xs text-gray-300">
            <li className="flex gap-2"><span className="text-red-500 mt-0.5">—</span>Insider threats are among the hardest to detect — attackers already have legitimate access</li>
            <li className="flex gap-2"><span className="text-red-500 mt-0.5">—</span>Malicious activity is interleaved with normal day-to-day work patterns</li>
            <li className="flex gap-2"><span className="text-red-500 mt-0.5">—</span>Static rules miss gradual behavioural drift over days/weeks</li>
            <li className="flex gap-2"><span className="text-red-500 mt-0.5">—</span>Class imbalance: only 7% of 1,000 users are true insiders</li>
          </ul>
        </div>
        <div className="card p-4" style={{ borderLeft: '3px solid #6366f1' }}>
          <h3 className="text-sm font-semibold text-indigo-400 mb-3">The Approach</h3>
          <ul className="space-y-2 text-xs text-gray-300">
            <li className="flex gap-2"><span className="text-indigo-500 mt-0.5">—</span>LSTM Autoencoder learns <em>normal</em> temporal behaviour patterns per user</li>
            <li className="flex gap-2"><span className="text-indigo-500 mt-0.5">—</span>High reconstruction error signals behavioural drift</li>
            <li className="flex gap-2"><span className="text-indigo-500 mt-0.5">—</span>DLP classifier scores email/file events for data sensitivity</li>
            <li className="flex gap-2"><span className="text-indigo-500 mt-0.5">—</span>7 signals fused by GA-evolved weights for a single risk score</li>
          </ul>
        </div>
        <div className="card p-4" style={{ borderLeft: '3px solid #10b981' }}>
          <h3 className="text-sm font-semibold text-emerald-400 mb-3">The Outcome</h3>
          <ul className="space-y-2 text-xs text-gray-300">
            <li className="flex gap-2"><span className="text-emerald-500 mt-0.5">—</span>LSTM day-level AUC 0.60 vs Isolation Forest's 0.38</li>
            <li className="flex gap-2"><span className="text-emerald-500 mt-0.5">—</span>GA raised F1@50 from 0.250 to 0.267 without retraining</li>
            <li className="flex gap-2"><span className="text-emerald-500 mt-0.5">—</span>Top-20 flag: 50% precision — half of all flagged are real insiders</li>
            <li className="flex gap-2"><span className="text-emerald-500 mt-0.5">—</span>100% flag coverage: every alert has a plain-English XAI explanation</li>
          </ul>
        </div>
      </div>

      {/* Key insight */}
      <div className="card p-4" style={{ borderLeft: '3px solid #f59e0b' }}>
        <p className="text-sm text-amber-300 leading-relaxed">
          <strong>Key finding:</strong> The GA redistributed weight away from{' '}
          <code className="bg-gray-800 px-1.5 py-0.5 rounded text-xs font-mono">lstm_p95</code> (0.50 → 0.30) toward{' '}
          <code className="bg-gray-800 px-1.5 py-0.5 rounded text-xs font-mono">file_exfil</code> (0.09 → 0.30) and{' '}
          <code className="bg-gray-800 px-1.5 py-0.5 rounded text-xs font-mono">usb_activity</code> (0.09 → 0.28) — consistent with
          the CERT r4.2 insider population, which is dominated by data exfiltration via removable media.
        </p>
      </div>
    </div>
  );
}
