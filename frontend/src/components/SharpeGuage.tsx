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

// function hexToRgb(hex: string) {
//   const h = hex.replace("#", "");
//   const n = parseInt(h.length === 3 ? h.split("").map(c => c + c).join("") : h, 16);
//   return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
// }

// function rgbToHex(r: number, g: number, b: number) {
//   const to = (v: number) => v.toString(16).padStart(2, "0");
//   return `#${to(r)}${to(g)}${to(b)}`;
// }

// function lerp(a: number, b: number, t: number) {
//   return a + (b - a) * t;
// }

// function lerpColor(c1: string, c2: string, t: number) {
//   const A = hexToRgb(c1);
//   const B = hexToRgb(c2);
//   return rgbToHex(
//     Math.round(lerp(A.r, B.r, t)),
//     Math.round(lerp(A.g, B.g, t)),
//     Math.round(lerp(A.b, B.b, t))
//   );
// }

// function sharpeColor(value: number) {
//   const v = clamp(value, 0, 5);

//   const red = "#ef4444";     // Tailwind red-500
//   const yellow = "#f59e0b";  // amber-500
//   const teal = "#17c9a3";    // close to your green-blue

//   if (v <= 3) {
//     return lerpColor(red, yellow, v / 3);
//   }
//   return lerpColor(yellow, teal, (v - 3) / 2);
// }

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
  strokeWidth = 10,
  size = 110,
  title = "Sharpe Ratio",
}: GaugeProps) {
  const v = clamp(value, min, max);
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

  const activeColor = "#1ABD8C"; // keep 0..5 semantics

  return (
    <div className="w-full h-full rounded-3xl p-8 flex flex-col items-center justify-center">
      <div className="text-white text-[16px] font-semibold tracking-tight mb-4">{title}</div>

      <div className="relative" style={{ width: size, height: size - 20 }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="block">
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
        <div className="absolute inset-0 flex items-center justify-center top-2">
          <div className="text-white text-3xl font-semibold">{v.toFixed(1)}</div>
        </div>
      </div>
    </div>
  );
}
