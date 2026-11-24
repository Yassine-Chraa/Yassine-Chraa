"""The trading theories.

Each theory looks at the entry candles and returns +1 (buy), -1 (sell) or 0 on
the candle where its trigger fires. Every theory also has a "+htf" version
that only keeps signals in the direction of the higher-timeframe trend.
Calibration decides which versions carry weight for each symbol.
"""
from __future__ import annotations

import pandas as pd

from . import indicators as ind
from .mtf import htf_frame


def _sig(buy: pd.Series, sell: pd.Series) -> pd.Series:
    return buy.astype(int) - sell.astype(int)


def ema_cross(df):
    f, s = ind.ema(df.close, 9), ind.ema(df.close, 21)
    return _sig(ind.cross_up(f, s), ind.cross_down(f, s))


def rsi_reversal(df):
    r = ind.rsi(df.close)
    return _sig(ind.cross_up(r, 30), ind.cross_down(r, 70))


def macd_momentum(df):
    line, sig = ind.macd(df.close)
    return _sig(ind.cross_up(line, sig) & (line < 0), ind.cross_down(line, sig) & (line > 0))


def donchian_breakout(df, n=20):
    hi = df.high.rolling(n).max().shift()
    lo = df.low.rolling(n).min().shift()
    return _sig(ind.cross_up(df.close, hi), ind.cross_down(df.close, lo))


def bollinger_reversion(df, n=20, k=2.0):
    m = df.close.rolling(n).mean()
    sd = df.close.rolling(n).std()
    return _sig(ind.cross_up(df.close, m - k * sd), ind.cross_down(df.close, m + k * sd))


def market_structure(df, w=5):
    """Break of the most recent confirmed swing high / swing low."""
    roll_hi = df.high.rolling(2 * w + 1, center=True).max()
    roll_lo = df.low.rolling(2 * w + 1, center=True).min()
    swing_hi = df.high.where(df.high == roll_hi)
    swing_lo = df.low.where(df.low == roll_lo)
    # A swing is only known w candles later.
    last_hi = swing_hi.shift(w).ffill()
    last_lo = swing_lo.shift(w).ffill()
    return _sig(ind.cross_up(df.close, last_hi), ind.cross_down(df.close, last_lo))


def supertrend_flip(df):
    d = ind.supertrend_dir(df)
    return _sig((d == 1) & (d.shift() == -1), (d == -1) & (d.shift() == 1))


def stoch_cross(df):
    k, d = ind.stoch(df)
    return _sig(ind.cross_up(k, d) & (k < 20), ind.cross_down(k, d) & (k > 80))


THEORIES = {
    "ema_cross": ema_cross,
    "rsi_reversal": rsi_reversal,
    "macd_momentum": macd_momentum,
    "donchian_breakout": donchian_breakout,
    "bollinger_reversion": bollinger_reversion,
    "market_structure": market_structure,
    "supertrend_flip": supertrend_flip,
    "stoch_cross": stoch_cross,
}


def all_signals(df: pd.DataFrame, htf=("4h", "1D"), bias: pd.Series | None = None) -> pd.DataFrame:
    if bias is None:
        bias = htf_frame(df, htf).htf_bias
    out = {}
    for name, fn in THEORIES.items():
        s = fn(df).fillna(0).astype(int)
        out[name] = s
        out[f"{name}+htf"] = s.where(s == bias, 0)
    return pd.DataFrame(out, index=df.index)
