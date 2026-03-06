import { useMemo } from "react";

type GaugeProps = {
  value: number;         // Sharpe, 0..5
  min?: number;          // default 0
  max?: number;          // default 5
  strokeWidth?: number;  // default 16
  size?: number;         // default 260 (square)
  title?: string;        // "Sharpe Ratio"
};

function clamp(x: number, a: number, b: number) {
  return Math.min(b, Math.max(a, x));
}

function normalizeDeg(a: number) {
  return ((a % 360) + 360) % 360;
}

// returns clockwise delta from start to end in [0, 360)
function cwDelta(start: number, end: number) {
  const s = normalizeDeg(start);
  const e = normalizeDeg(end);
  return (e - s + 360) % 360;
}

function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number) {
  const toRad = (deg: number) => (deg * Math.PI) / 180;

  const s = normalizeDeg(startAngle);
  const e = normalizeDeg(endAngle);
  const delta = cwDelta(s, e);

  const start = { x: cx + r * Math.cos(toRad(s)), y: cy + r * Math.sin(toRad(s)) };
  const end = { x: cx + r * Math.cos(toRad(e)), y: cy + r * Math.sin(toRad(e)) };

  const largeArcFlag = delta > 180 ? "1" : "0";
  const sweepFlag = "1"; // clockwise in screen coords with your sin/cos usage

  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} ${sweepFlag} ${end.x} ${end.y}`;
}


export default function SharpeGauge({
  value,
  min = 0,
  max = 5,
  strokeWidth = 9.5,
  size = 110,
  title = "Sharpe Ratio",
}: GaugeProps) {
  const v = value > 0 ? clamp(value, min, max) : clamp(value, -max, -min);
  const pct = (v - min) / (max - min); // 0..1

  // Geometry
  const pad = 0;
  const cx = size / 2;
  const cy = size / 2; // push arc down a bit
  const r = size / 2 - pad - strokeWidth / 2;

  // Arc: slightly more than a semicircle like your screenshot
  const startAngle = 165;
  const endAngle = 15;

  const pathD = useMemo(() => describeArc(cx, cy, r, startAngle, endAngle), [cx, cy, r, endAngle, startAngle]);

  // Path length for dash math
  const pathLen = useMemo(() => {
    // approximate with SVG path length once mounted would be ideal,
    // but for gauges, this is fine: use a constant via <path pathLength="100"> trick.
    return 100;
  }, []);

  const dashArray = pathLen;
  const dashOffset = pathLen * (1 - pct);

  const activeColor = value > 0 ? "#1ABD8C" : "#FF4500";



  return (
    <div className="rounded-3xl p-8 flex flex-col items-center justify-center mt-4 2xl:mt-8">
      <div className="text-white 2xl:text-[20px] text-sm font-semibold tracking-tight mb-4 2xl:mb-6">{title}</div>

    <div className="relative w-[100px] h-[80px] 2xl:w-[160px] 2xl:h-[140px]">
        <svg viewBox={`0 0 ${size} ${size}`} className="w-full h-full">
          {/* Track */}
          <path
            d={pathD}
            fill="none"
            stroke="rgba(40,39,53,0.6)"
            strokeWidth={strokeWidth}
            strokeLinecap="butt"
            pathLength={pathLen}
          />

          {/* Active arc */}
          <path
            d={pathD}
            fill="none"
            stroke={activeColor}
            strokeWidth={strokeWidth}
            strokeLinecap="butt"
            pathLength={pathLen}
            strokeDasharray={dashArray}
            strokeDashoffset={dashOffset}
            style={{
              filter: `drop-shadow(0 0 10px ${activeColor}55)`,
              transition: "stroke-dashoffset 400ms ease, stroke 300ms ease",
            }}
          />
        </svg>

        {/* Center value */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-white 2xl:text-3xl text-xl font-semibold">{v < 0 ? (-1 * v).toFixed(1) : v.toFixed(1)}</div>
        </div>
      </div>
    </div>
  );
}
