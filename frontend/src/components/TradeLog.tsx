import { useEffect, useMemo, useState } from "react";
import { 
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";

import { fetchTrades, type TradeRow } from "../lib/fetchData";

// type TradesResponse = {
//   rows: TradeRow[];
//   page: number;      // 1-based
//   pageSize: number;
//   total: number;
// };

// const FAKE_TRADES: TradeRow[] = [
//   {
//     traded_at: "2026-01-20",
//     symbol: "SPY",
//     side: "BUY",
//     qty: 1.5,
//     price: 5425,
//     notional: 8137.5,
//     sig: 1.2483,
//   },
//   {
//     traded_at: "2026-01-19",
//     symbol: "SPY",
//     side: "SELL",
//     qty: 2.0,
//     price: 5402,
//     notional: 10804,
//     sig: -0.8421,
//   },
//   {
//     traded_at: "2026-01-18",
//     symbol: "QQQ",
//     side: "BUY",
//     qty: 3.0,
//     price: 412,
//     notional: 1236,
//     sig: 0.9325,
//   },
//   {
//     traded_at: "2026-01-17",
//     symbol: "SPY",
//     side: "BUY",
//     qty: 1.0,
//     price: 5388,
//     notional: 5388,
//     sig: 0.5561,
//   },
//   {
//     traded_at: "2026-01-16",
//     symbol: "IWM",
//     side: "SELL",
//     qty: 4.0,
//     price: 210,
//     notional: 840,
//     sig: -1.1032,
//   },
//   {
//     traded_at: "2026-01-15",
//     symbol: "SPY",
//     side: "BUY",
//     qty: 2.5,
//     price: 5400,
//     notional: 13500,
//     sig: 1.2959,
//   },
//   {
//     traded_at: "2026-01-14",
//     symbol: "QQQ",
//     side: "SELL",
//     qty: 1.2,
//     price: 405,
//     notional: 486,
//     sig: -0.6543,
//   },
//   {
//     traded_at: "2026-01-13",
//     symbol: "SPY",
//     side: "BUY",
//     qty: 1.8,
//     price: 5375,
//     notional: 9675,
//     sig: 0.8877,
//   },
//   {
//     traded_at: "2026-01-12",
//     symbol: "IWM",
//     side: "BUY",
//     qty: 3.2,
//     price: 208,
//     notional: 665.6,
//     sig: 0.4412,
//   },
//   {
//     traded_at: "2026-01-11",
//     symbol: "SPY",
//     side: "SELL",
//     qty: 2.1,
//     price: 5350,
//     notional: 11235,
//     sig: -0.9724,
//   },
// ];

const TradeLog = () => {

  const [data, setData] = useState<TradeRow[]>([]);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(6);
  const [total, setTotal] = useState(0);
  const pageCount = Math.max(1, Math.ceil(total / pageSize));

  const col = createColumnHelper<TradeRow>();

  const columns = [
  col.accessor("traded_at", {
    header: "Date Traded",
    size: 180,
  }),
  col.accessor("ticker", {
    header: "Symbol",
    size: 90,
  }),
  col.accessor("side", {
    header: "Side",
    size: 90,
  }),
  col.accessor("qty", {
    header: "Quantity",
    size: 120,
  }),
  col.accessor("price", {
    header: "Price",
    size: 140,
  }),
  col.accessor("notional", {
    header: "Notional",
    size: 160,
  }),
  col.accessor("signal", {
  header: "Signal",
  cell: (info) => {
    const v = info.getValue();

    const color =
      v > 0
        ? "text-emerald-400"
        : v < 0
        ? "text-red-400"
        : "text-gray-400";

    return (
      <span className={`${color} font-semibold`}>
        {v > 0 ? "+" : ""}
        {v.toFixed(4)}
      </span>
    );
  },
}),
];

  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    manualPagination: true,
    pageCount,
  });

  useEffect(() => {
    let cancelled = false;

    async function fetchData() {
      // Replace with your API
      // const res = await fetch(`/api/trades?page=${page}&pageSize=${pageSize}`);
      // const json = (await res.json()) as TradesResponse;

      const rawData = await fetchTrades(page, pageSize);

      if (cancelled) return;
      setData(rawData.rows);
      setTotal(rawData.total);
    }

    fetchData();
    return () => {
      cancelled = true;
    };
  }, [page, pageSize]);

  const canPrev = page > 1;
  const canNext = page < pageCount;

  // Pagination numbers like: 1 2 3 ... 100
  const pages = useMemo(() => {
    const out: (number | "...")[] = [];
    if (pageCount <= 6) {
      for (let i = 1; i <= pageCount; i++) out.push(i);
      return out;
    }
    out.push(1);
    if (page > 3) out.push("...");
    const start = Math.max(2, page - 1);
    const end = Math.min(pageCount - 1, page + 1);
    for (let i = start; i <= end; i++) out.push(i);
    if (page < pageCount - 2) out.push("...");
    out.push(pageCount);
    return out;
  }, [page, pageCount]);

  
  return (
    <div className="w-135 h-75 rounded-lg purple-shadow p-6 relative">
      <div className="w-full">
      <div className="text-white text-md font-semibold tracking-tight flex items-center mb-4">Previous Trades</div>

      <div className="w-full text-white">
        {/* Header */}
       {table.getHeaderGroups().map(hg => (
        <div key={hg.id} className="grid grid-cols-7 gap-6 text-white text-[10px]">
          {hg.headers.map(h => (
            <div key={h.id} style={{ width: h.column.getSize() }}>
              {flexRender(h.column.columnDef.header, h.getContext())}
            </div>
          ))}
        </div>
      ))}

        {/* Rows */}
        {table.getRowModel().rows.map(row => (
        <div key={row.id} className="grid grid-cols-7 gap-6 text-[10px] py-1 my-1 border-t border-white/10">
          {row.getVisibleCells().map(cell => (
            <div key={cell.id} style={{ width: cell.column.getSize()}}>
              {flexRender(cell.column.columnDef.cell, cell.getContext())}
            </div>
          ))}
        </div>
      ))}

        {/* Pagination */}
        <div className="flex items-center justify-end gap-1 text-white/85 text-[10px] absolute bottom-4 right-6">
          <button
            className={`px-2 ${canPrev ? "hover:text-white" : "opacity-40 cursor-not-allowed"}`}
            onClick={() => canPrev && setPage((p) => p - 1)}
            disabled={!canPrev}
          >
            ‹
          </button>

          {pages.map((p, idx) =>
            p === "..." ? (
              <span key={`dots-${idx}`} className="px-1 text-white/60">…</span>
            ) : (
              <button
                key={p}
                onClick={() => setPage(p)}
                className={`px-1 ${p === page ? "text-white" : "text-white/70 hover:text-white"}`}
              >
                {p}
              </button>
            )
          )}

          <button
            className={`px-2 ${canNext ? "hover:text-white" : "opacity-40 cursor-not-allowed"}`}
            onClick={() => canNext && setPage((p) => p + 1)}
            disabled={!canNext}
          >
            ›
          </button>
        </div>
      </div>
    </div>

    </div>
  )
}

export default TradeLog