import {useState } from 'react';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';


function tickFormatter(ms: number, mode: number) {
  const d = new Date(ms);

  if (mode === 0) {
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
  }

  if (mode === 1 || mode === 2) {
    // show day of month (or switch to "Jan 12" if you want)
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
  }

  return d.toLocaleString("en-US", { month: "short", timeZone: "UTC" });
}

function dataFormatter(data: { date: string; equity: number }[]) {
  if (!data.length) return [];

  const sorted = [...data].sort((a, b) => a.date.localeCompare(b.date));

  // If only 1 point, add a synthetic previous day point at same equity
  if (sorted.length === 1) {
    const only = sorted[0];
    const onlyMs = new Date(only.date + "T00:00:00Z").getTime();
    return [
      { t: onlyMs - 86400000, equity: only.equity },
      { t: onlyMs, equity: only.equity },
    ];
  }

  const start = sorted[0].date;
  const end = sorted[sorted.length - 1].date;

  const byDate = new Map<string, number>();
  for (const p of sorted) byDate.set(p.date, p.equity);

  const startMs = new Date(start + "T00:00:00Z").getTime();
  const endMs = new Date(end + "T00:00:00Z").getTime();

  const newData: { t: number; equity: number }[] = [];
  let currEquity = byDate.get(start) ?? sorted[0].equity;

  for (let t = startMs; t <= endMs; t += 86400000) {
    const dateStr = new Date(t).toISOString().slice(0, 10);
    const found = byDate.get(dateStr);
    if (found !== undefined) currEquity = found;
    newData.push({ t, equity: currEquity });
  }

  return newData;
}

function monthStartTicks(startMs: number, endMs: number) {
  const ticks: number[] = [];

  const d = new Date(startMs);
  d.setUTCDate(1);
  d.setUTCHours(0, 0, 0, 0);

  while (d.getTime() <= endMs) {
    ticks.push(d.getTime());
    d.setUTCMonth(d.getUTCMonth() + 1);
  }

  return ticks;
}

function dataFilter(data: { t: number; equity: number }[], mode: number) {
  if (!data.length) return [];
  const end = data[data.length - 1].t;
  let start = end;

  if (mode === 1) start = end - 7 * 86400000;
  else if (mode === 2) start = end - 30 * 86400000;
  else if (mode === 3) start = end - 90 * 86400000;
  else if (mode === 4) start = end - 365 * 86400000;
  else if (mode === 5) start = data[0].t;

  return data.filter((p) => p.t >= start && p.t <= end);
}

  const DAY = 86400000;

function domainForMode(mode: number, dataMin: number, dataMax: number): [number, number] {
  const end = dataMax;
  let start = dataMin;

  if (mode === 0) start = end - DAY;
  else if (mode === 1) start = end - 7 * DAY;
  else if (mode === 2) start = end - 30 * DAY;
  else if (mode === 3) start = end - 90 * DAY;
  else if (mode === 4) start = end - 365 * DAY;
  else if (mode === 5) start = dataMin;

  // ensure the domain isn't "after" the earliest data point
  if (start > dataMin) start = dataMin;

  return [start, end];
}

function ticksForMode(mode: number, domain: [number, number]) {
  const [start, end] = domain;

  if (mode === 0) {
    return [end];
  }

  if (mode === 1) {
    return Array.from({ length: 8 }, (_, i) => start + i * DAY);
  }

  if (mode === 2) {
    return Array.from({ length: 7 }, (_, i) => start + i * 5 * DAY);
  }

  if (mode === 3) {
    return Array.from({ length: 7 }, (_, i) => start + i * 15 * DAY);
  }

  if (mode === 4) {
    return Array.from({ length: 13 }, (_, i) => start + i * 30 * DAY);
  }

  if (mode === 5) {
    return monthStartTicks(start, end);
  }

  return [];
}

const Graph = ({ mode, data } : { mode: number, data: {date: string, equity: number}[] }) => {
  const [NOW] = useState(() => Date.now());

  const formattedData = dataFormatter(data);
  const filteredData = dataFilter(formattedData, mode);
  const dataMin = formattedData[0]?.t ?? NOW;
  const dataMax = formattedData[formattedData.length - 1]?.t ?? NOW;

  const domain = domainForMode(mode, dataMin, dataMax);
  const ticks = ticksForMode(mode, domain);

  return (
    <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={filteredData} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
          <defs>
            {/* Fill gradient under the line */}
            <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(16, 185, 129, 0.35)" />
              <stop offset="100%" stopColor="rgba(16, 185, 129, 0.0)" />
            </linearGradient>

            {/* Soft glow on the line */}
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="rgba(16,185,129,0.55)" />
            </filter>
          </defs>

          <CartesianGrid stroke="rgba(255,255,255,0.10)" vertical={false} />

          

          <XAxis
          dataKey="t"
          type="number"
          scale="time"
          domain={domain}
          ticks={ticks.length ? ticks : undefined}
          tickFormatter={(v) => tickFormatter(v as number, mode)}
          tick={{ fill: "rgba(255,255,255,0.75)", fontSize: 10, fontWeight: 300 }}
          interval={0}
          minTickGap={0}
          tickMargin={12}
          axisLine={false}
          tickLine={false}
        />


          <YAxis
            tickFormatter={(v) => `${v}$`}
            tick={{ fill: "rgba(255,255,255,0.75)", fontSize: 8 }}
            axisLine={false}
            tickLine={false}
            width={48}
          />

          <Tooltip
            contentStyle={{
              background: "rgba(10,10,16,0.95)",
              border: "1px solid rgba(255,255,255,0.10)",
              borderRadius: 10,
            }}
            labelStyle={{ color: "rgba(255,255,255,0.8)" }}
            labelFormatter={(label) => {
            const ms = Number(label);
            const d = new Date(ms);

            if (mode === 0) {
              return d.toLocaleString("en-US", { hour: "2-digit", minute: "2-digit",   timeZone: "UTC", });
            }
            if (mode === 1 || mode === 2) {
              return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
            }
            return d.toLocaleString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
          }}
            formatter={(v) => [`${v}$`, "Equity"]}
          />

          <Area
            type="monotone"
            dataKey="equity"
            stroke="#1ABD8C"
            strokeWidth={2.5}
            fill="url(#equityFill)"
            filter="url(#glow)"
            dot={false}
            // activeDot={false}
            // // Render a dot only on the last point
            isAnimationActive={false}
          />

          {/* Custom last-point dot layer */}
          {/* <Area
            type="monotone"
            dataKey="value"
            stroke="transparent"
            fill="transparent"
            dot={(props) => {
              const { cx, cy, index } = props;
              if (index !== data.length - 1) return null;
              return (
                <circle
                  cx={cx}
                  cy={cy}
                  r={6}
                  fill="rgba(16,185,129,1)"
                  stroke="rgba(16,185,129,0.25)"
                  strokeWidth={6}
                />
              );
            }}
          /> */}
        </AreaChart>
      </ResponsiveContainer>
  )
}

export default Graph