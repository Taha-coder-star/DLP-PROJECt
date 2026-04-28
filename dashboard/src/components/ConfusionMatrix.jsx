export default function ConfusionMatrix({ tp, fp, fn, tn, title }) {
  const total = tp + fp + fn + tn;
  const fmt = (n) => n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
  const pct = (n) => ((n / total) * 100).toFixed(1) + '%';

  const MatrixCell = ({ value, label, bg }) => (
    <div
      className="flex flex-col items-center justify-center rounded-lg p-3"
      style={{ background: bg, border: '1px solid hsl(220 20% 14%)' }}
    >
      <span className="text-xl font-bold font-mono text-white">{fmt(value)}</span>
      <span className="text-xs text-gray-300 mt-0.5">{pct(value)}</span>
      <span className="text-xs text-gray-400 mt-1">{label}</span>
    </div>
  );

  return (
    <div>
      {title && <p className="text-sm font-semibold text-gray-300 mb-3">{title}</p>}
      <div className="text-xs text-gray-500 text-center mb-2">Predicted</div>
      <div className="flex gap-2">
        <div className="flex items-center">
          <span className="text-xs text-gray-500 -rotate-90 whitespace-nowrap origin-center" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>
            Actual
          </span>
        </div>
        <div className="flex-1">
          <div className="grid grid-cols-2 gap-1 mb-1">
            <span className="text-xs text-gray-500 text-center">Pos (Insider)</span>
            <span className="text-xs text-gray-500 text-center">Neg (Normal)</span>
          </div>
          <div className="grid grid-cols-2 gap-1">
            <MatrixCell value={tp} label="TP — Correct catch" bg="rgba(16,185,129,0.2)" />
            <MatrixCell value={fp} label="FP — False alarm"   bg="rgba(239,68,68,0.12)" />
            <MatrixCell value={fn} label="FN — Missed"        bg="rgba(239,68,68,0.16)" />
            <MatrixCell value={tn} label="TN — Correct clear" bg="rgba(16,185,129,0.08)" />
          </div>
        </div>
      </div>
    </div>
  );
}
