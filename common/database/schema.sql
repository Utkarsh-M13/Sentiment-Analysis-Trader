-- =========================
-- Sentiment Trader: schema.sql
-- =========================

-- 0) Core admin tables
CREATE TABLE IF NOT EXISTS pipeline_state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS schema_snapshot (
  id BIGSERIAL PRIMARY KEY,
  label TEXT NOT NULL,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1) Raw news
CREATE TABLE IF NOT EXISTS news_headlines (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  provider TEXT NOT NULL,                    -- 'polygon'
  provider_id TEXT UNIQUE NOT NULL,
  tickers TEXT[] DEFAULT '{}',                  -- array of associated tickers
  published_utc TIMESTAMPTZ NOT NULL,
  url TEXT,
  title TEXT NOT NULL,
  description TEXT,
  source_name TEXT,
  content_hash TEXT GENERATED ALWAYS AS (
    md5(coalesce(title,'') || '|' || coalesce(description,''))
  ) STORED,
  inserted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (provider, provider_id)
);
CREATE INDEX IF NOT EXISTS idx_news_published ON news_headlines (published_utc DESC);
CREATE INDEX IF NOT EXISTS idx_news_ticker_time ON news_headlines (tickers, published_utc DESC);

-- -- 2) Model scores per headline
-- CREATE TABLE IF NOT EXISTS headline_scores (
--   id BIGSERIAL PRIMARY KEY,
--   headline_id BIGINT NOT NULL REFERENCES news_headlines(id) ON DELETE CASCADE,
--   model_name TEXT NOT NULL,                  -- 'finbert'
--   model_version TEXT NOT NULL,               -- '2025-10-30'
--   score REAL NOT NULL,                       -- raw sentiment or signed score
--   p_up REAL,                                 -- optional probability style output
--   extra JSONB,                               -- logits, spans, etc
--   scored_at TIMESTAMPTZ NOT NULL DEFAULT now(),
--   UNIQUE (headline_id, model_name, model_version)
-- );
-- CREATE INDEX IF NOT EXISTS idx_scores_headline ON headline_scores (headline_id);
-- CREATE INDEX IF NOT EXISTS idx_scores_model_time ON headline_scores (model_name, scored_at DESC);

-- -- 3) Daily features used by backtests
-- CREATE TABLE IF NOT EXISTS daily_features (
--   id BIGSERIAL PRIMARY KEY,
--   date DATE NOT NULL,
--   symbol TEXT NOT NULL,
--   global_sentiment REAL NOT NULL,
--   n_articles INT NOT NULL,
--   computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
--   UNIQUE (date, symbol)
-- );
-- CREATE INDEX IF NOT EXISTS idx_features_symbol_date ON daily_features (symbol, date);

-- -- 4) Trades and audit
-- CREATE TABLE IF NOT EXISTS trades (
--   id BIGSERIAL PRIMARY KEY,
--   symbol TEXT NOT NULL,
--   trade_time TIMESTAMPTZ NOT NULL,
--   side TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
--   qty NUMERIC(18,6) NOT NULL,
--   price NUMERIC(18,6) NOT NULL,
--   strategy TEXT NOT NULL,
--   order_id TEXT,
--   extra JSONB
-- );
-- CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades (symbol, trade_time);

-- -- 5) Optional app role setup (adjust names and secrets before running)
-- -- CREATE ROLE trader_app LOGIN PASSWORD 'change_me';
-- -- GRANT CONNECT ON DATABASE postgres TO trader_app;
-- -- GRANT USAGE ON SCHEMA public TO trader_app;
-- -- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO trader_app;
-- -- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO trader_app;

-- -- 6) Optional snapshot marker (edit the label if you want)
-- -- INSERT INTO schema_snapshot(label) VALUES ('initial_all_in_one') ON CONFLICT DO NOTHING;
