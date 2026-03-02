import { useEffect, useMemo, useState } from "react";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";

type HeadlineRow = {
  title: string;
  source: string;
  published: string; // YYYY-MM-DD
  score: number;
  url?: string;
};

const FAKE_HEADLINES: HeadlineRow[] = [
  {
    title: "Why Smart Money Is Looking Overseas for Bank Stocks",
    source: "Investing.com",
    published: "2026-01-15",
    score: 0.0959475,
    url: "https://example.com/a",
  },
  {
    title: "Markets Dip as Traders Reprice Rate-Cut Expectations",
    source: "Reuters",
    published: "2026-01-15",
    score: -0.043218,
    url: "https://example.com/b",
  },
  {
    title: "Tech Leads Broad Rally as Volatility Cools",
    source: "Bloomberg",
    published: "2026-01-14",
    score: 0.072113,
    url: "https://example.com/c",
  },
  {
    title: "Energy Names Slide on Soft Demand Signals",
    source: "WSJ",
    published: "2026-01-14",
    score: -0.01842,
    url: "https://example.com/d",
  },
  {
    title: "Earnings Season Preview: What to Watch This Week",
    source: "CNBC",
    published: "2026-01-13",
    score: 0.03155,
    url: "https://example.com/e",
  },
  {
    title: "Dollar Strength Pressures Multinationals",
    source: "FT",
    published: "2026-01-13",
    score: -0.05602,
    url: "https://example.com/f",
  },
  {
    title: "Investors Rotate Into Defensives Ahead of CPI",
    source: "MarketWatch",
    published: "2026-01-12",
    score: -0.00944,
    url: "https://example.com/g",
  },
  {
    title: "Small Caps Catch a Bid as Breadth Improves",
    source: "Barron's",
    published: "2026-01-12",
    score: 0.04491,
    url: "https://example.com/h",
  },
  {
    title: "Analysts Raise SPY Targets After Strong Macro Prints",
    source: "Seeking Alpha",
    published: "2026-01-11",
    score: 0.0832,
    url: "https://example.com/i",
  },
  {
    title: "Risk-Off Move as Credit Spreads Widen",
    source: "Reuters",
    published: "2026-01-11",
    score: -0.0679,
    url: "https://example.com/j",
  },
  // Add more if you want more pages
];

const col = createColumnHelper<HeadlineRow>();

const columns = [
  col.accessor("title", {
    header: "Title ↕",
    size: 240,
    cell: (info) => {
      const row = info.row.original;
      const title = info.getValue();
      const inner = (
    <span className="block w-full min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">
      {title}
    </span>
  );

  return row.url ? (
    <a
      href={row.url}
      target="_blank"
      rel="noreferrer"
      className="block min-w-0 text-white/90 hover:text-white transition-colors hover:underline"
      title={title}
    >
      {inner}
    </a>
  ) : (
    <span className="block min-w-0 text-white/90" title={title}>
      {inner}
    </span>
  );
    },
  }),
  col.accessor("source", {
    header: "Source ↕",
    size: 80,
    cell: (info) => <span className="text-white/80">{info.getValue()}</span>,
  }),
  col.accessor("published", {
    header: "Published ↕",
    size: 80,
    cell: (info) => <span className="text-white/80">{info.getValue()}</span>,
  }),
  col.accessor("score", {
    header: "Score ↕",
    size: 40,
    cell: (info) => {
      const v = info.getValue();
      const color =
        v > 0 ? "text-emerald-400" : v < 0 ? "text-red-400" : "text-gray-400";
      return (
        <span className={`${color} font-semibold`}>
          {v > 0 ? "+" : ""}
          {v.toFixed(7)}
        </span>
      );
    },
  }),
];

const HeadlineLog = () => {
  const [data, setData] = useState<HeadlineRow[]>([]);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(6);
  const [total, setTotal] = useState(0);
  const pageCount = Math.max(1, Math.ceil(total / pageSize));

  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    manualPagination: true,
    pageCount,
  });

  const GRID = "grid grid-cols-[1fr_80px_80px_80px] gap-6";

  useEffect(() => {
    let cancelled = false;

    async function fetchHeadlines() {
      if (cancelled) return;

      setData(FAKE_HEADLINES.slice((page - 1) * pageSize, page * pageSize));
      setTotal(FAKE_HEADLINES.length);
    }

    fetchHeadlines();
    return () => {
      cancelled = true;
    };
  }, [page, pageSize]);

  const canPrev = page > 1;
  const canNext = page < pageCount;

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
    <div className="w-135 h-75 rounded-lg purple-shadow p-6">
      <div className="w-full">
        <div className="text-white text-md font-semibold tracking-tight flex items-center mb-4">
          News Headlines
        </div>

        <div className="w-full text-white">
          {/* Header */}
          {table.getHeaderGroups().map((hg) => (
            <div key={hg.id} className={`${GRID} text-white text-[10px]`}>
              {hg.headers.map((h) => (
                <div key={h.id} className="min-w-0">
                  {flexRender(h.column.columnDef.header, h.getContext())}
                </div>
              ))}
            </div>
          ))}

          {/* Rows */}
          {table.getRowModel().rows.map((row) => (
            <div key={row.id} className={`${GRID} text-[10px] py-1 my-1 border-t border-white/10`}>
              {row.getVisibleCells().map((cell) => (
                <div key={cell.id} className="min-w-0">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </div>
              ))}
            </div>
          ))}

          {/* Pagination */}
          <div className="flex items-center justify-end gap-1 text-white/85 text-[10px] mt-4">
            <button
              className={`px-2 ${
                canPrev ? "hover:text-white" : "opacity-40 cursor-not-allowed"
              }`}
              onClick={() => canPrev && setPage((p) => p - 1)}
              disabled={!canPrev}
            >
              ‹
            </button>

            {pages.map((p, idx) =>
              p === "..." ? (
                <span key={`dots-${idx}`} className="px-1 text-white/60">
                  …
                </span>
              ) : (
                <button
                  key={p}
                  onClick={() => setPage(p)}
                  className={`px-1 ${
                    p === page
                      ? "text-white"
                      : "text-white/70 hover:text-white"
                  }`}
                >
                  {p}
                </button>
              )
            )}

            <button
              className={`px-2 ${
                canNext ? "hover:text-white" : "opacity-40 cursor-not-allowed"
              }`}
              onClick={() => canNext && setPage((p) => p + 1)}
              disabled={!canNext}
            >
              ›
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HeadlineLog;