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

  return { rows: (data as HeadlineRow[]) ?? [], total: count ?? 0 };
}

export async function fetchEquityData() {
  const { data, error } = await supabase
    .from("equity_history")
    .select("as_of_date, equity")
    .eq("ticker", "SPY")
    .order("as_of_date", { ascending: true });
  

  if (error) throw error;


  return (data as { as_of_date: string; equity: number }[]).map((row) => ({
    date: formatDateOnly(row.as_of_date).toString(),
    equity: row.equity,
  })); 
}

export async function fetchBnHData() {
  const { data, error } = await supabase
    .from("spy_bnh_view")
    .select("as_of_date, equity")
    .order("as_of_date", { ascending: true });
      

  if (error) throw error; 

  return (data as { as_of_date: string; equity: number }[]).map((row) => ({
    date: formatDateOnly(row.as_of_date).toString(),
    equity: row.equity,
  })); 
}

function formatDateOnly(utcString: string) {
  return new Date(utcString).toISOString().slice(0, 10);
}

export async function fetchSharpeRatio(): Promise<number | null> {
  const { data, error } = await supabase.rpc("get_sharpe_ratio");
  if (error) throw error;
  return data ?? null;
}

export async function fetchExposure(): Promise<number | null> {
  const { data, error } = await supabase.rpc("get_gross_exposure_spy");
  if (error) throw error;
  return data ?? null;
}

export async function get_average_allocation(): Promise<number | null> {
  const { data, error } = await supabase.rpc("get_latest_daily_exposure_spy");
  if (error) throw error;
  return data ?? null;
}

export async function get_last_five_trade_signals(): Promise<{trade_day: string; signal: string}[] | null> {
  const { data, error } = await supabase.rpc("get_last_5_trade_signals");
  if (error) throw error;
  return data ?? null;
}

type EquitySnapshot = {
  current_equity: number
  current_as_of: string
  reference_equity: number
  reference_as_of: string
  reference_source: string
  weekly_return: number | null
}

export async function get_equity_now_and_reference(): Promise<EquitySnapshot | null> {
  const { data, error } = await supabase.rpc("get_equity_now_and_reference");

  console.log("equity data", data);

  if (error) throw error;

  if (!data || data.length === 0) return null;
  return data[0];
}
