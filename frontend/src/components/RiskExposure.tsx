import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
} from "recharts";
import { fetchExposure, get_average_allocation, get_last_five_trade_signals } from "../lib/fetchData";

// const signalData = [
//   { name: "Mon", value: 2.6 },
//   { name: "Tue", value: 1.8 },
//   { name: "Wed", value: 2.1 },
//   { name: "Thu", value: 0.4 },
//   { name: "Fri", value: 1.9 },
// ];

const SignalGraph = () => {
  const [signalData, setSignalData] = useState<{ name: string; value: number }[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      // Simulate fetching data with a delay
      const signalRaw = await get_last_five_trade_signals();
      console.log('signalRaw', signalRaw)
      if (signalRaw !== null && signalRaw !== undefined) {
        const signalFormatted = signalRaw.map((item) => ({
          name: item.trade_day,
          value: parseFloat(item.signal),
        }));
        console.log('signalFormatted', signalFormatted)
        setSignalData(signalFormatted)
      }
    };

    fetchData();
  }, []);


  return (
    <div>
      <div className="text-[8px] text-white/85 mb-2 mt-8 2xl:mt-10">Signal Graph</div>

      <div className="2xl:h-32 h-20 w-full min-w-0">
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
    <div className=" 2xl:mb-8 mb-4">
      <div className="text-[8px] text-white/85 mb-3">{label}</div>

      <div className="relative 2xl:h-7 h-5 rounded-sm bg-black/60 border border-white/10 overflow-hidden">
      <div
        className={`h-full bg-linear-to-r ${gradient} rounded-sm transition-[width] duration-700 ease-out`}
        style={{ width: `${shown}%` }}
      />
      <div className="absolute inset-0 flex items-center pl-2 font-light text-black text-xs">
        {value.toString().slice(0, 5)}%
      </div>
    </div>
    </div>
  );
};

const RiskExposure = () => {
  const [portfolioExposure, setPortfolioExposure] = useState(0);
  const [avgAllocation, setAvgAllocation] = useState(0);

  useEffect(() => {
    const fetchData = async () => {
      // Simulate fetching data with a delay
      const rawExposure = await fetchExposure();
      if (rawExposure !== null && rawExposure !== undefined) {
        setPortfolioExposure(parseFloat(rawExposure.toPrecision(4)) * 100);
      }

      const avg = await get_average_allocation();
      if (avg !== null && avg !== undefined) {
        setAvgAllocation(parseFloat(avg.toPrecision(4)) * 100);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="2xl:w-56 2xl:h-100 w-40 h-75 rounded-lg purple-shadow p-4">
      <div className="text-md font-semibold mb-4 tracking-tight text-white">
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