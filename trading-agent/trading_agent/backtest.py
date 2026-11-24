"""Replay and calibration.

replay_theory: trade every signal of one theory with the live SL/TP rules and
measure the result in R (1R = the amount risked).

calibrate: split history into in-sample and out-of-sample parts. A theory gets
weight for a symbol only if it also made money out of sample. The weight is
its out-of-sample average R, so better theories count for more in the vote.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import indicators as ind
from .strategies import all_signals


def replay_signals(df: pd.DataFrame, sig: pd.Series, sl_atr: float, tp_atr: float, max_hold: int = 72):
    """One position at a time; enter at the next candle's open. Returns a list of R results."""
    a = ind.atr(df).to_numpy()
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    s = sig.to_numpy()
    results, entry_times = [], []
    i, n = 0, len(df)
    while i < n - 1:
        side = s[i]
        if side == 0 or not np.isfinite(a[i]) or a[i] <= 0:
            i += 1
            continue
        e = o[i + 1]
        risk = sl_atr * a[i]
        sl = e - side * risk
        tp = e + side * tp_atr * a[i]
        r = None
        j = i + 1
        while j < min(n, i + 1 + max_hold):
            if side > 0:
                if l[j] <= sl: r = -1.0; break
                if h[j] >= tp: r = tp_atr / sl_atr; break
            else:
                if h[j] >= sl: r = -1.0; break
                if l[j] <= tp: r = tp_atr / sl_atr; break
            j += 1
        if r is None:
            j = min(j, n - 1)
            r = side * (c[j] - e) / risk
        results.append(r)
        entry_times.append(df.index[i])
        i = j + 1
    return results, entry_times


def stats(rs):
    if not rs:
        return {"trades": 0, "avg_r": 0.0, "total_r": 0.0, "win_rate": 0.0, "max_dd_r": 0.0}
    arr = np.array(rs)
    eq = np.cumsum(arr)
    dd = float(np.max(np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]))
    return {"trades": len(rs), "avg_r": round(float(arr.mean()), 3), "total_r": round(float(arr.sum()), 2),
            "win_rate": round(float((arr > 0).mean()), 3), "max_dd_r": round(dd, 2)}


def calibrate_symbol(df, profile):
    sigs = all_signals(df, profile.htf)
    split = int(len(df) * (1 - profile.oos_fraction))
    split_time = df.index[split]
    report, weights = {}, {}
    for name in sigs.columns:
        rs, times = replay_signals(df, sigs[name], profile.sl_atr, profile.tp_atr)
        ins = [r for r, t in zip(rs, times) if t < split_time]
        oos = [r for r, t in zip(rs, times) if t >= split_time]
        si, so = stats(ins), stats(oos)
        report[name] = {"in_sample": si, "oos": so}
        ok = so["trades"] >= profile.min_oos_trades and so["avg_r"] >= profile.min_oos_avg_r and si["avg_r"] > 0
        weights[name] = round(so["avg_r"], 3) if ok else 0.0
    # Do not count a theory twice: keep only the better of the plain and +htf versions.
    for base in {k.replace("+htf", "") for k in weights}:
        a, b = weights.get(base, 0), weights.get(base + "+htf", 0)
        if a and b:
            if a >= b: weights[base + "+htf"] = 0.0
            else: weights[base] = 0.0
    return weights, report


def ensemble_replay(df, weights, profile, symbol):
    """Replay of the combined vote (no Claude, no news): a rough upper view of the mechanical part."""
    sigs = all_signals(df, profile.htf)
    vote = combined_vote(sigs, weights, profile.vote_memory(symbol))
    sig = pd.Series(0, index=df.index)
    sig[vote >= profile.ensemble_threshold] = 1
    sig[vote <= -profile.ensemble_threshold] = -1
    entries = sig.where(sig != sig.shift(), 0)
    split = df.index[int(len(df) * (1 - profile.oos_fraction))]
    rs, times = replay_signals(df, entries, profile.sl_atr, profile.tp_atr)
    return stats([r for r, t in zip(rs, times) if t >= split])


def combined_vote(sigs: pd.DataFrame, weights: dict, memory: int) -> pd.Series:
    """Each weighted theory votes its latest non-zero signal from the last `memory` candles.
    Result is normalised to [-1, 1] by the total weight."""
    active = {k: w for k, w in weights.items() if w > 0 and k in sigs}
    if not active:
        return pd.Series(0.0, index=sigs.index)
    total = sum(active.values())
    vote = pd.Series(0.0, index=sigs.index)
    for k, w in active.items():
        s = sigs[k].replace(0, np.nan).ffill(limit=memory - 1).fillna(0) if memory > 1 else sigs[k]
        vote += w * s
    return vote / total


def run_calibration(profile, broker, symbols=None):
    out_w, out_r = {}, {}
    for sym in symbols or profile.symbols:
        df = broker.candles(sym, profile.timeframe, profile.calibrate_bars)
        w, rep = calibrate_symbol(df, profile)
        out_w[sym] = w
        out_r[sym] = {"bars": len(df), "from": str(df.index[0]), "to": str(df.index[-1]), "theories": rep,
                      "ensemble_oos": ensemble_replay(df, w, profile, sym)}
        live = {k: v for k, v in w.items() if v > 0}
        print(f"{sym}: {len(df)} bars, weighted theories: {live or 'none'} | ensemble OOS {out_r[sym]['ensemble_oos']}")
    (profile.state_dir / "theory_weights.json").write_text(json.dumps(out_w, indent=1))
    (profile.state_dir / "calibration_report.json").write_text(json.dumps(
        {"generated_utc": datetime.now(timezone.utc).isoformat(), "symbols": out_r}, indent=1))
    return out_w
