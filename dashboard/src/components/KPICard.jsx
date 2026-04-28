export default function KPICard({ label, value, sub, delta, deltaGood = true, glow, mono = false }) {
  const glowClass = glow === 'blue'  ? 'metric-glow'
                  : glow === 'green' ? 'metric-glow-green'
                  : glow === 'red'   ? 'metric-glow-red'
                  : '';

  const accentBar = glow === 'blue'  ? 'accent-bar-blue'
                  : glow === 'green' ? 'accent-bar-green'
                  : glow === 'red'   ? 'accent-bar-red'
                  : '';

  return (
    <div className={`card p-4 hover:scale-[1.02] transition-transform duration-150 ${glowClass} ${accentBar}`}>
      <p className="text-xs text-gray-500 uppercase tracking-wider mb-1.5 font-medium">{label}</p>
      <p className={`text-2xl font-bold text-white ${mono ? 'font-mono' : ''}`}>{value}</p>
      {sub && <p className="text-xs text-gray-500 mt-1.5">{sub}</p>}
      {delta !== undefined && (
        <p className={`text-xs mt-1.5 font-medium ${deltaGood ? 'text-emerald-400' : 'text-red-400'}`}>
          {delta}
        </p>
      )}
    </div>
  );
}
