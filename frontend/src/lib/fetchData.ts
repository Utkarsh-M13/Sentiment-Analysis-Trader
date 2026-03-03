import { supabase } from "../lib/supabase";

export type TradeRow = {
  traded_at: string;
  ticker: string;
  side: "BUY" | "SELL";
  qty: number;
  price: number;
  notional: number;
  signal: number;
  pred_use: string;
  as_of_date: string;
};

export async function fetchTrades(page: number, pageSize: number) {
  const from = (page - 1) * pageSize;
  const to = from + pageSize - 1;

  const { data, error, count } = await supabase
    .from("trade_logs")
    .select(
      "traded_at,ticker,side,qty,price,signal,pred_use,as_of_date,notional",
      { count: "exact" }
    )
    .order("traded_at", { ascending: false })
    .range(from, to);

  if (error) throw error;

  return { rows: data ?? [], total: count ?? 0 };
}

export type HeadlineRow = {
  scored_at: string;
  provider: string;
  source_name: string;
  title: string;
  url: string | undefined;
  score: number;
  model_name: string;
  published_utc: string;
};

export async function fetchHeadlines(
  page: number,
  pageSize: number,
  modelName = "finbert-regression-v1"
) {
  const from = (page - 1) * pageSize;
  const to = from + pageSize - 1;

  const { data, error, count } = await supabase
    .from("headline_scores_with_headline")
    .select(
      "scored_at,provider,source_name,title,url,score,model_name,published_utc",
      { count: "exact" }
    )
    .eq("model_name", modelName)
    .order("scored_at", { ascending: false })
    .range(from, to);

  if (error) throw error;
  console.log('data', data)

  return { rows: (data as HeadlineRow[]) ?? [], total: count ?? 0 };
}