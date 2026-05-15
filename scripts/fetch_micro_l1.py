"""
Research: download SPY 1-minute historical bars (TRADES) from IB TWS/Gateway
and persist to DuckDB table `spy_l1_1min`.

Requires: ibapi, duckdb, pandas; TWS or IB Gateway running.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

_PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "src"))

from TradingApp import TradingApp  # noqa: E402
from training import DEFAULT_DB_PATH  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch SPY 1m L1 history into DuckDB.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4002, help="4002 paper / 4001 live (typical)")
    p.add_argument("--client-id", type=int, default=101)
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--months", type=float, default=3.0, help="Calendar months of history")
    p.add_argument("--db-path", default=str(_PROJ / "Data" / "quant.db") help="DuckDB database path")
    p.add_argument(
        "--duration-chunk",
        default="5 D",
        help="IB durationStr per request (keep under ~2000 1m bars)",
    )
    p.add_argument(
        "--ib-volume-x100",
        action="store_true",
        help="Multiply bar volume by 100 (some IB configs report lots)",
    )
    return p.parse_args()


def _ensure_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spy_l1_1min (
            ts TIMESTAMP NOT NULL,
            symbol VARCHAR NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            trades BIGINT,
            wap DOUBLE
        );
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_spy_l1_1min_symbol_ts
        ON spy_l1_1min (symbol, ts);
    """)


def _fetch_paginated(
    app: TradingApp,
    contract,
    *,
    months: float,
    duration_chunk: str,
    req_id_start: int = 7001,
) -> pd.DataFrame:
    tz = "America/New_York"
    end = pd.Timestamp.now(tz=tz)
    days_back = max(1, int(round(float(months) * 365.25 / 12.0)))
    start_cutoff = end - pd.Timedelta(days=days_back)
    parts: list[pd.DataFrame] = []
    req_id = req_id_start
    seen_min: pd.Timestamp | None = None
    max_iters = 120

    for _ in range(max_iters):
        end_str = end.strftime("%Y%m%d %H:%M:%S") + " US/Eastern"
        chunk = app.get_historical_data(
            req_id,
            contract,
            durationStr=duration_chunk,
            barSizeSetting="1 min",
            whatToShow="TRADES",
            endDateTime=end_str,
            historical_timeout=180.0,
        )
        req_id += 1
        if chunk is None or chunk.empty:
            break
        if not isinstance(chunk.index, pd.DatetimeIndex):
            chunk.index = pd.to_datetime(chunk.index)
        if chunk.index.tz is None:
            chunk.index = chunk.index.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        else:
            chunk.index = chunk.index.tz_convert(tz)

        oldest = chunk.index.min()
        if seen_min is not None and oldest >= seen_min - pd.Timedelta(minutes=1):
            break
        seen_min = oldest
        parts.append(chunk.sort_index())

        if oldest <= start_cutoff.tz_convert(tz):
            break
        end = oldest - pd.Timedelta(minutes=1)
        time.sleep(0.35)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, axis=0)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[out.index >= start_cutoff.tz_convert(tz)]
    return out


def main() -> None:
    args = _parse_args()
    app = TradingApp(apply_us_stock_volume_x100=args.ib_volume_x100)
    try:
        app.connect_and_run(args.host, args.port, args.client_id)
        contract = TradingApp.get_stock_contract(args.symbol)
        df = _fetch_paginated(
            app,
            contract,
            months=args.months,
            duration_chunk=args.duration_chunk,
        )
    finally:
        try:
            app.disconnect()
        except Exception:
            pass

    if df.empty:
        print("Nenhuma barra retornada; verifique TWS/Gateway, permissões de dados e horário.")
        return

    df = df.sort_index()
    idx_name = df.index.name or "index"
    df_out = df.reset_index().rename(columns={idx_name: "ts"})
    df_out["symbol"] = args.symbol
    for c in ("open", "high", "low", "close", "volume", "wap"):
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce")
    df_out["trades"] = pd.to_numeric(df_out["trades"], errors="coerce").fillna(0).astype("int64")
    df_out = df_out[["ts", "symbol", "open", "high", "low", "close", "volume", "trades", "wap"]]

    conn = duckdb.connect(args.db_path)
    try:
        _ensure_table(conn)
        conn.execute("DELETE FROM spy_l1_1min WHERE symbol = ?", [args.symbol])
        conn.register("_spy_l1_staging", df_out)
        conn.execute("""
            INSERT INTO spy_l1_1min (ts, symbol, open, high, low, close, volume, trades, wap)
            SELECT ts, symbol, open, high, low, close, volume, trades, wap
            FROM _spy_l1_staging;
        """)
        conn.unregister("_spy_l1_staging")
        n = conn.execute(
            "SELECT COUNT(*) FROM spy_l1_1min WHERE symbol = ?",
            [args.symbol],
        ).fetchone()[0]
    finally:
        conn.close()

    print(f"Gravadas {n} linhas em spy_l1_1min ({args.symbol}) em {args.db_path}")


if __name__ == "__main__":
    main()
