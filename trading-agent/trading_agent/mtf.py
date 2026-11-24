"""Higher-timeframe trend filter and context.

Higher-timeframe candles are built by resampling the entry candles. Only
*completed* higher-timeframe candles are used (shift by one before aligning),
so a backtest never peeks at a candle that had not closed yet.
"""
from __future__ import annotations

import pandas as pd

from . import indicators as ind

RULE = {"1h": "1h", "4h": "4h", "1D": "1D"}


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


def trend(df: pd.DataFrame) -> pd.Series:
    """+1 up / -1 down / 0 unclear: EMA20 vs EMA50 with price on the same side."""
    e20, e50 = ind.ema(df.close, 20), ind.ema(df.close, 50)
    up = (e20 > e50) & (df.close > e50)
    dn = (e20 < e50) & (df.close < e50)
    return up.astype(int) - dn.astype(int)


def _align(htf_series: pd.Series, index: pd.DatetimeIndex, rule: str) -> pd.Series:
    # A higher-timeframe candle labelled t closes at t + rule; it is known from then on.
    s = htf_series.copy()
    s.index = s.index + pd.Timedelta(rule)
    return s.reindex(index, method="ffill")


def htf_frame(df: pd.DataFrame, htf=("4h", "1D")) -> pd.DataFrame:
    a, b = htf
    out = pd.DataFrame(index=df.index)
    for tf in (a, b):
        h = resample(df, RULE[tf])
        key = tf.lower()
        out[f"{key}_trend"] = _align(trend(h), df.index, RULE[tf])
        out[f"{key}_rsi"] = _align(ind.rsi(h.close), df.index, RULE[tf])
        out[f"{key}_adx"] = _align(ind.adx(h), df.index, RULE[tf])
        out[f"{key}_atr"] = _align(ind.atr(h), df.index, RULE[tf])
    ta, tb = out[f"{a.lower()}_trend"], out[f"{b.lower()}_trend"]
    # Bias only when both higher timeframes agree.
    out["htf_bias"] = ((ta == tb) * ta).fillna(0).astype(int)
    return out


def context(df: pd.DataFrame, htf=("4h", "1D")) -> dict:
    """Snapshot of the market for Claude to read."""
    h = htf_frame(df, htf).iloc[-1]
    last = df.iloc[-1]
    d1 = resample(df, "1D")
    prev_day = d1.iloc[-2] if len(d1) > 1 else d1.iloc[-1]
    d1_atr = float(ind.atr(d1).iloc[-1]) if len(d1) > 15 else float("nan")
    ctx = {
        "time_utc": str(df.index[-1]),
        "close": float(last.close),
        "entry_rsi": round(float(ind.rsi(df.close).iloc[-1]), 1),
        "entry_adx": round(float(ind.adx(df).iloc[-1]), 1),
        "entry_atr": float(ind.atr(df).iloc[-1]),
        "prev_day_high": float(prev_day.high),
        "prev_day_low": float(prev_day.low),
        "dist_from_prev_low_in_d1_atr": round((float(last.close) - float(prev_day.low)) / d1_atr, 2) if d1_atr == d1_atr else None,
        "dist_from_prev_high_in_d1_atr": round((float(prev_day.high) - float(last.close)) / d1_atr, 2) if d1_atr == d1_atr else None,
        "change_24_bars_pct": round(100 * (float(last.close) / float(df.close.iloc[-25]) - 1), 2) if len(df) > 25 else None,
        "htf_bias": int(h.htf_bias),
    }
    for k, v in h.items():
        if k != "htf_bias":
            ctx[k] = round(float(v), 5 if "atr" in k else 1)
    return ctx
