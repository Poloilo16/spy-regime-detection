"""
Research: plot SPY 1m price vs VPIN (last 5 days from DuckDB).

Requires: matplotlib, pandas, duckdb; table `spy_l1_1min` populated (e.g. fetch_micro_l1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ / "src"))

from micro_math import compute_vpin_pipeline  # noqa: E402
from training import DEFAULT_DB_PATH  # noqa: E402

TOXICITY_LEVEL = 0.60

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot SPY price vs VPIN (last N days).")
    p.add_argument("--db-path", default=str(_PROJ / "Data" / "quant.db"), type=str)
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--days", type=float, default=5.0, help="Trailing window in calendar days")
    p.add_argument("--bucket-volume", type=float, default=50_000.0)
    p.add_argument("--vpin-window-buckets", type=int, default=50)
    p.add_argument(
        "--auto-bucket-volume",
        action="store_true",
        help="Override bucket size as clamp(5 * mean 1m volume, 5k..200k)",
    )
    p.add_argument(
        "--price",
        choices=("wap", "close", "wap_close"),
        default="wap_close",
        help="wap_close: WAP when valid, else close",
    )
    p.add_argument("--out", type=Path, default=None, help="PNG path (default: logs/micro_vpin.png)")
    return p.parse_args()


def load_last_days(
    db_path: str,
    symbol: str,
    days: float,
) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    try:
        row = con.execute(
            "SELECT max(ts) AS mx FROM spy_l1_1min WHERE symbol = ?",
            [symbol],
        ).fetchone()
        if not row or row[0] is None:
            return pd.DataFrame()
        mx = pd.Timestamp(row[0])
        if mx.tzinfo is None:
            mx = mx.tz_localize("UTC")
        else:
            mx = mx.tz_convert("UTC")
        cutoff = mx - pd.Timedelta(days=float(days))
        df = con.execute(
            """
            SELECT ts, symbol, open, high, low, close, volume, trades, wap
            FROM spy_l1_1min
            WHERE symbol = ? AND ts >= ?
            ORDER BY ts
            """,
            [symbol, cutoff],
        ).df()
    finally:
        con.close()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _price_series(df: pd.DataFrame, mode: str) -> pd.Series:
    close = df["close"].astype(float)
    if mode == "close":
        s = close
    elif mode == "wap":
        s = df["wap"].astype(float)
    else:
        wap = df["wap"].astype(float)
        ok = wap.notna() & np.isfinite(wap.to_numpy(dtype=float)) & (wap > 0)
        s = wap.where(ok, close)
    s = pd.Series(s.to_numpy(dtype=float), index=pd.to_datetime(df["ts"], utc=True))
    s.index.name = "ts"
    return s


def _bucket_volume(df: pd.DataFrame, base: float, auto: bool) -> float:
    if not auto or df.empty:
        return float(base)
    mv = float(df["volume"].mean())
    if not np.isfinite(mv) or mv <= 0:
        return float(base)
    return float(max(5_000, min(200_000, round(5.0 * mv))))

def _vpin_cross_up_times(vpin_df: pd.DataFrame, level: float) -> pd.DatetimeIndex:
    if vpin_df.empty or "bucket_end_ts" not in vpin_df.columns:
        return pd.DatetimeIndex([], tz="UTC")
    v = vpin_df["vpin"].astype(float)
    prev = v.shift(1)
    cross = (prev <= level) & (v > level)
    
    # Converte para datetimes e depois força a ser um DatetimeIndex
    ts_series = pd.to_datetime(vpin_df.loc[cross, "bucket_end_ts"], utc=True)
    return pd.DatetimeIndex(ts_series)


def main() -> None:
    args = _parse_args()
    db_path = str(Path(args.db_path).resolve())
    df = load_last_days(db_path, args.symbol, args.days)
    if df.empty:
        print("Nenhuma linha em spy_l1_1min para o recorte; rode fetch_micro_l1 primeiro.")
        return

    bucket_vol = _bucket_volume(df, args.bucket_volume, args.auto_bucket_volume)
    vpin_df = compute_vpin_pipeline(
        df,
        bucket_volume=bucket_vol,
        vpin_window_buckets=args.vpin_window_buckets,
    )
    if vpin_df.empty or "bucket_end_ts" not in vpin_df.columns:
        print("VPIN vazio ou sem timestamps de bucket; verifique volume e bucket_volume.")
        return

    vpin_df = vpin_df.copy()
    vpin_df["bucket_end_ts"] = pd.to_datetime(vpin_df["bucket_end_ts"], utc=True)

    price = _price_series(df, args.price)
    cross_ts = _vpin_cross_up_times(vpin_df, TOXICITY_LEVEL)
    common = cross_ts.intersection(price.index)
    marker_prices = price.reindex(common)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax1.plot(price.index, price.values, color="black", lw=0.8, label=args.price)
    valid_m = marker_prices.notna()
    if valid_m.any():
        ax1.scatter(
            marker_prices.index[valid_m],
            marker_prices.values[valid_m],
            color="red",
            s=36,
            zorder=5,
            marker="o",
            label=f"VPIN cruza {TOXICITY_LEVEL} ↑",
        )
    ax1.set_ylabel("Preço")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title(
        f"{args.symbol} — últimos {args.days:g} d | bucket_vol={bucket_vol:,.0f} | "
        f"VPIN window={args.vpin_window_buckets} buckets"
    )

    ax2.plot(
        vpin_df["bucket_end_ts"],
        vpin_df["vpin"],
        color="steelblue",
        lw=0.9,
        label="VPIN",
    )
    ax2.axhline(
        TOXICITY_LEVEL,
        color="red",
        ls="--",
        lw=1.0,
        label=f"Toxicidade crítica ({TOXICITY_LEVEL})",
    )
    ax2.set_ylabel("VPIN")
    ax2.set_xlabel("Tempo (UTC)")
    ax2.legend(loc="upper left", fontsize=8)
    vmax = float(np.nanmax(vpin_df["vpin"].to_numpy(dtype=float)))
    if not np.isfinite(vmax):
        vmax = 1.0
    ax2.set_ylim(0.0, max(1.0, vmax * 1.05))

    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    out = args.out or (_PROJ / "logs" / "micro_vpin.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Figura salva em: {out}")


if __name__ == "__main__":
    main()
