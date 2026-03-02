import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
} from "recharts";

const signalData = [
  { name: "Mon", value: 2.6 },
  { name: "Tue", value: 1.8 },
  { name: "Wed", value: 2.1 },
  { name: "Thu", value: 0.4 },
  { name: "Fri", value: 1.9 },
];

const SignalGraph = () => {
  return (
    <div>
      <div className="text-[8px] text-white/85 mb-2">Signal Graph</div>

      <div className="h-20 w-full min-w-0">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={signalData} margin={{ top: 6, right: 0, left: 0, bottom: 0 }}>
            <XAxis
              dataKey="name"
              axisLine={false}
              tickLine={false}
              tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 7 }}
            />
            <YAxis
              width={18}
              axisLine={false}
              tickLine={false}
              tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 7 }}
            />
            <Bar
              dataKey="value"
              fill="rgba(16,185,129,0.9)"
              radius={[2, 2, 0, 0]}
              barSize={14}
              isAnimationActive
              animationDuration={800}
              animationEasing="ease-out"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

type ProgressProps = {
  label: string;
  value: number;
  gradient: string;
};

const ProgressBar = ({ label, value, gradient }: ProgressProps) => {
  const [shown, setShown] = useState(0);

  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(value));
    return () => cancelAnimationFrame(id);
  }, [value]);

  return (
    <div className="mb-4">
      <div className="text-[8px] text-white/85 mb-3">{label}</div>

      <div className="relative h-5 rounded-sm bg-black/60 border border-white/10 overflow-hidden">
      <div
        className={`h-full bg-linear-to-r ${gradient} rounded-sm transition-[width] duration-700 ease-out`}
        style={{ width: `${shown}%` }}
      />
      <div className="absolute inset-0 flex items-center pl-2 font-light text-black text-xs">
        {value}%
      </div>
    </div>
    </div>
  );
};

const RiskExposure = () => {
  const portfolioExposure = 65;
  const avgAllocation = 50;

  return (
    <div className="w-40 h-75 rounded-lg purple-shadow p-4">
      <div className="text-md font-semibold mb-2 tracking-tight text-white">
        Risk & Exposure
      </div>

      {/* Portfolio Exposure */}
      <ProgressBar
        label="Portfolio Exposure"
        value={portfolioExposure}
        gradient="from-violet-500 to-indigo-500"
      />

      {/* Average Monthly Allocation */}
      <ProgressBar
        label="Average Monthly Allocation"
        value={avgAllocation}
        gradient="from-emerald-400 to-teal-500"
      />

      {/* Signal Graph */}
      <SignalGraph />
    </div>
  )
}

export default RiskExposure