import { useState } from 'react';
import {
  Home, Database, TreePine, Brain, Shield, Dna, AlertTriangle,
  ShieldCheck, ChevronLeft, ChevronRight,
} from 'lucide-react';
import Overview        from './pages/Overview.jsx';
import Dataset         from './pages/Dataset.jsx';
import IsolationForest from './pages/IsolationForest.jsx';
import LSTMAutoencoder from './pages/LSTMAutoencoder.jsx';
import DLPSensitivity  from './pages/DLPSensitivity.jsx';
import RiskScorerGA    from './pages/RiskScorerGA.jsx';
import TopAnomalies    from './pages/TopAnomalies.jsx';

const TABS = [
  { id: 'overview',  label: 'Overview',          icon: Home,            short: 'Overview' },
  { id: 'dataset',   label: 'Dataset & Pipeline', icon: Database,        short: 'Dataset' },
  { id: 'iforest',   label: 'Isolation Forest',   icon: TreePine,        short: 'IF' },
  { id: 'lstm',      label: 'LSTM Autoencoder',   icon: Brain,           short: 'LSTM' },
  { id: 'dlp',       label: 'DLP Sensitivity',    icon: Shield,          short: 'DLP' },
  { id: 'ga',        label: 'Risk Scorer & GA',   icon: Dna,             short: 'GA' },
  { id: 'anomalies', label: 'Top Anomalies',      icon: AlertTriangle,   short: 'Results' },
];

const PAGE = {
  overview:  Overview,
  dataset:   Dataset,
  iforest:   IsolationForest,
  lstm:      LSTMAutoencoder,
  dlp:       DLPSensitivity,
  ga:        RiskScorerGA,
  anomalies: TopAnomalies,
};

export default function App() {
  const [active, setActive] = useState('overview');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const ActivePage = PAGE[active];

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: 'hsl(222 47% 6%)' }}>
      {/* Sidebar */}
      <aside
        className={`flex flex-col border-r transition-all duration-200 flex-shrink-0 ${
          sidebarOpen ? 'w-56' : 'w-14'
        }`}
        style={{ background: 'hsl(223 40% 7%)', borderColor: 'hsl(220 20% 14%)' }}
      >
        {/* Logo */}
        <div
          className="flex items-center gap-2.5 px-3 py-4 border-b"
          style={{ borderColor: 'hsl(220 20% 14%)' }}
        >
          <ShieldCheck size={22} className="text-accent-light flex-shrink-0" />
          {sidebarOpen && (
            <div>
              <p className="text-sm font-semibold text-white leading-tight tracking-tight">UEBA · DLP</p>
              <p className="text-xs text-gray-500">Insider Threat</p>
            </div>
          )}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="ml-auto text-gray-500 hover:text-gray-300 transition-colors"
          >
            {sidebarOpen
              ? <ChevronLeft size={16} />
              : <ChevronRight size={16} />
            }
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 py-3 overflow-y-auto">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActive(tab.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 text-sm transition-all duration-150 relative ${
                  active === tab.id
                    ? 'text-accent-light font-medium tab-active'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-white/[0.03]'
                }`}
                style={active === tab.id ? { background: 'rgba(99, 102, 241, 0.08)' } : undefined}
                title={!sidebarOpen ? tab.label : undefined}
              >
                <Icon size={18} className="flex-shrink-0" />
                {sidebarOpen && <span className="truncate">{tab.label}</span>}
              </button>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="px-3 py-3 border-t" style={{ borderColor: 'hsl(220 20% 14%)' }}>
          {sidebarOpen ? (
            <div>
              <p className="text-xs text-gray-600">CERT r4.2 · 1,000 users</p>
              <p className="text-xs text-gray-600">70 confirmed insiders</p>
            </div>
          ) : (
            <span className="text-xs text-gray-600">r4.2</span>
          )}
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        {/* Top bar */}
        <div
          className="sticky top-0 z-10 backdrop-blur border-b px-6 py-3 flex items-center gap-4"
          style={{ background: 'hsla(222, 47%, 6%, 0.88)', borderColor: 'hsl(220 20% 14%)' }}
        >
          <div>
            <h1 className="text-sm font-semibold text-white tracking-tight">
              {TABS.find((t) => t.id === active)?.label}
            </h1>
            <p className="text-xs text-gray-500">
              AI-Driven UEBA + DLP · CERT Insider Threat Dataset r4.2
            </p>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <span className="inline-flex items-center gap-1.5 bg-emerald-950/40 text-emerald-400 border border-emerald-700/40 rounded-full px-3 py-1 text-xs font-medium">
              <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
              GA-Optimised Active
            </span>
            <span className="text-xs text-gray-500 font-mono">F1@50 = 0.267</span>
          </div>
        </div>

        {/* Page content */}
        <div className="p-6 page-enter" key={active}>
          <ActivePage />
        </div>
      </main>
    </div>
  );
}
