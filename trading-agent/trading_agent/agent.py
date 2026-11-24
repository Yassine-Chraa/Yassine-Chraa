"""Command line for the trading agent.

    python -m trading_agent analyze  --profile forex   # market snapshot + theory votes -> analysis.json
    (Claude reads analysis.json, checks the news, writes decisions.json)
    python -m trading_agent execute  --profile forex   # trades only where Claude AND the votes agree
    python -m trading_agent status   --profile forex
    python -m trading_agent calibrate --profile forex  # re-pick which theories carry weight

Add --paper to any command to use synthetic data and a simulated account.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone

from . import config
from .backtest import combined_vote, run_calibration
from .broker import BrokerError, get_broker
from .mtf import context
from .strategies import all_signals
from . import indicators as ind

DAYS = "MTWTFSS"


def now_utc():
    return datetime.now(timezone.utc)


def load_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def journal(profile, event: dict):
    event = {"time_utc": now_utc().isoformat(timespec="seconds"), **event}
    with open(profile.state_dir / "journal.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")
    print(f"[journal] {event}")


def day_allowed(profile, t=None):
    t = t or now_utc()
    return profile.trading_days[t.weekday()] == DAYS[t.weekday()]


def new_trades_allowed(profile, t=None):
    t = t or now_utc()
    if not day_allowed(profile, t):
        return False, f"{DAYS[t.weekday()]} ({t:%A}) is not a trading day for {profile.name}"
    cut = profile.no_new_trades_after_friday_utc
    if cut is not None and t.weekday() == 4 and t.hour >= cut:
        return False, f"past the Friday {cut}:00 UTC cutoff (weekend gap)"
    return True, "ok"


def side_of(pos):
    return "BUY" if "BUY" in str(pos.get("type", "")) else "SELL"


# --------------------------------------------------------------------------- analyze
def analyze(profile, broker):
    weights = load_json(profile.state_dir / "theory_weights.json", {})
    if not weights:
        print("WARNING: no theory_weights.json yet; run `calibrate` first. Every vote will be 0.")
    calib = load_json(profile.state_dir / "calibration_report.json", {}) or {}
    acct = broker.account()
    positions = [p for p in broker.positions() if p.get("symbol") in profile.symbols]
    allowed, why = new_trades_allowed(profile)
    out = {
        "generated_utc": now_utc().isoformat(timespec="seconds"),
        "profile": profile.name, "stage": profile.stage,
        "account": {k: acct.get(k) for k in ("login", "server", "type", "currency", "balance", "equity")},
        "new_trades_allowed": allowed, "reason": why,
        "open_positions": [{k: p.get(k) for k in ("id", "symbol", "type", "volume", "openPrice", "stopLoss", "takeProfit", "profit", "time")} for p in positions],
        "symbols": {},
    }
    for sym in profile.symbols:
        try:
            df = broker.candles(sym, profile.timeframe, profile.history_bars)
        except BrokerError as e:
            out["symbols"][sym] = {"error": str(e)}
            continue
        last_open = df.index[-1]
        # Drop the candle that is still forming.
        if last_open + timedelta(minutes=60 if profile.timeframe == "1h" else 15) > now_utc().replace(tzinfo=None):
            df = df.iloc[:-1]
        sigs = all_signals(df, profile.htf)
        w = weights.get(sym, {})
        mem = profile.vote_memory(sym)
        vote = combined_vote(sigs, w, mem)
        recent = sigs.tail(mem)
        votes = {}
        for k, wt in w.items():
            if wt <= 0:
                continue
            nz = recent[k][recent[k] != 0]
            votes[k] = {"weight": wt, "vote": int(nz.iloc[-1]) if len(nz) else 0,
                        "oos": (calib.get("symbols", {}).get(sym, {}).get("theories", {}).get(k, {}) or {}).get("oos")}
        firing_unweighted = {k: int(v) for k, v in sigs.iloc[-1].items() if v != 0 and w.get(k, 0) <= 0}
        ens = round(float(vote.iloc[-1]), 3)
        out["symbols"][sym] = {
            "ensemble": ens,
            "ensemble_side": "BUY" if ens >= profile.ensemble_threshold else "SELL" if ens <= -profile.ensemble_threshold else "NONE",
            "weighted_votes": votes,
            "firing_but_unweighted": firing_unweighted,
            "context": context(df, profile.htf),
            "vote_memory_bars": mem,
        }
    (profile.state_dir / "analysis.json").write_text(json.dumps(out, indent=1, default=str))
    print_analysis(out)
    return out


def print_analysis(a):
    print(f"[{a['profile']}] {a['generated_utc']} stage={a['stage']} account={a['account'].get('login')} "
          f"({a['account'].get('type')}) equity={a['account'].get('equity')} {a['account'].get('currency')}")
    print(f"  new trades allowed: {a['new_trades_allowed']} ({a['reason']}); open positions: {len(a['open_positions'])}")
    for sym, s in a["symbols"].items():
        if "error" in s:
            print(f"  {sym}: ERROR {s['error']}")
            continue
        c = s["context"]
        v = {k: x["vote"] for k, x in s["weighted_votes"].items() if x["vote"]}
        print(f"  {sym}: ensemble {s['ensemble']:+.2f} ({s['ensemble_side']}) | htf_bias {c['htf_bias']:+d} "
              f"| 4h trend {c.get('4h_trend')} rsi {c.get('4h_rsi')} adx {c.get('4h_adx')} | 1d trend {c.get('1d_trend')} rsi {c.get('1d_rsi')} "
              f"| votes {v or '-'}")


# --------------------------------------------------------------------------- execute
def position_size(broker, sym, equity, risk_frac, sl_dist):
    spec = broker.spec(sym)
    price = broker.price(sym)
    tick = float(spec.get("tickSize") or 0)
    tick_val = float(price.get("lossTickValue") or price.get("profitTickValue") or 0)
    if tick <= 0 or tick_val <= 0:
        raise BrokerError(f"{sym}: missing tick size/value ({tick}, {tick_val})")
    loss_per_lot = sl_dist / tick * tick_val
    step = float(spec.get("volumeStep") or 0.01)
    vmin, vmax = float(spec.get("minVolume") or step), float(spec.get("maxVolume") or 100)
    raw = equity * risk_frac / loss_per_lot
    vol = math.floor(raw / step) * step
    if vol < vmin:
        # Only accept the broker minimum if it risks at most twice the target.
        if vmin * loss_per_lot <= 2 * equity * risk_frac:
            vol = vmin
        else:
            return 0.0, spec, price, loss_per_lot
    vol = min(vol, vmax)
    digits = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return round(vol, digits), spec, price, loss_per_lot


def round_to_tick(x, tick):
    if not tick:
        return x
    digits = max(0, -int(math.floor(math.log10(tick))))
    return round(round(x / tick) * tick, digits)


def execute(profile, broker, dry_run=False):
    sd = profile.state_dir
    analysis = load_json(sd / "analysis.json")
    decisions = load_json(sd / "decisions.json")
    if not analysis:
        raise SystemExit("No analysis.json: run `analyze` first.")
    t = now_utc()
    a_age = t - datetime.fromisoformat(analysis["generated_utc"])
    if a_age > timedelta(minutes=profile.decision_max_age_min):
        raise SystemExit(f"analysis.json is {a_age} old; run `analyze` again.")
    if not decisions:
        print("No decisions.json: no Claude decision, so no new trades.")
        return
    d_time = datetime.fromisoformat(decisions["generated_utc"].replace("Z", "+00:00"))
    if t - d_time > timedelta(minutes=profile.decision_max_age_min):
        print(f"Claude decision is stale ({t - d_time} old): no new trades.")
        return

    # Safety: the account type must match the stage.
    demo = broker.is_demo()
    if profile.stage == "DEMO" and not demo:
        raise SystemExit("Stage is DEMO but the connected account is not a demo account. Refusing to trade.")
    if profile.stage == "LIVE" and not config.ALLOW_LIVE_TRADING:
        raise SystemExit("Stage is LIVE but ALLOW_LIVE_TRADING=1 is not set. Refusing to trade.")

    acct = broker.account()
    equity = float(acct.get("equity") or acct.get("balance"))
    day = load_json(sd / "day_start.json", {}) or {}
    today = t.date().isoformat()
    if day.get("date") != today:
        day = {"date": today, "equity": equity}
        (sd / "day_start.json").write_text(json.dumps(day))
    day_pnl = (equity - day["equity"]) / day["equity"]

    positions = [p for p in broker.positions() if p.get("symbol") in profile.symbols]
    allowed, why = new_trades_allowed(profile, t)

    for d in decisions.get("decisions", []):
        sym = d.get("symbol", "")
        if sym not in profile.symbols:
            sym2 = next((s for s in profile.symbols if s.rstrip("m") == sym or s == sym + "m"), None)
            if not sym2:
                print(f"  skip unknown symbol {sym}")
                continue
            sym = sym2
        action = str(d.get("action", "HOLD")).upper()
        conf = float(d.get("confidence", 0))
        mine = [p for p in positions if p["symbol"] == sym]
        tag = f"{sym} {action} conf={conf:.2f} news={d.get('news')}"

        if d.get("close_open_trade") and mine:
            for p in mine:
                if dry_run:
                    print(f"  [dry-run] would close {p['id']} {sym}")
                    continue
                broker.close(p["id"])
                journal(profile, {"event": "close", "symbol": sym, "position": p["id"], "profit": p.get("profit"),
                                  "reasoning": d.get("reasoning")})
            positions = [p for p in positions if p["symbol"] != sym]
            mine = []

        if action not in ("BUY", "SELL"):
            print(f"  {tag}: hold")
            continue
        s = analysis["symbols"].get(sym, {})
        reasons = []
        if not allowed: reasons.append(why)
        if conf < profile.min_claude_confidence: reasons.append(f"confidence < {profile.min_claude_confidence}")
        if d.get("news") == "contradicts": reasons.append("news contradicts")
        if s.get("ensemble_side") != action: reasons.append(f"theory vote is {s.get('ensemble_side')} ({s.get('ensemble')})")
        if mine: reasons.append("already have a position")
        if len(positions) >= profile.max_open_positions: reasons.append("max open positions")
        if day_pnl <= -profile.max_daily_loss: reasons.append(f"daily loss limit hit ({day_pnl:.1%})")
        if reasons:
            print(f"  {tag}: NOT trading: {'; '.join(reasons)}")
            journal(profile, {"event": "skip", "symbol": sym, "action": action, "why": reasons})
            continue

        atr_v = float(s["context"]["entry_atr"])
        sl_dist = profile.sl_atr * atr_v
        vol, spec, price, lpl = position_size(broker, sym, equity, profile.risk_per_trade, sl_dist)
        if vol <= 0:
            print(f"  {tag}: NOT trading: minimum lot would risk too much")
            continue
        entry = float(price["ask"] if action == "BUY" else price["bid"])
        sgn = 1 if action == "BUY" else -1
        tick = float(spec.get("tickSize") or 0)
        sl = round_to_tick(entry - sgn * sl_dist, tick)
        tp = round_to_tick(entry + sgn * profile.tp_atr * atr_v, tick)
        info = {"event": "open", "symbol": sym, "side": action, "volume": vol, "entry": entry, "sl": sl, "tp": tp,
                "risk_usd": round(vol * lpl, 2), "ensemble": s.get("ensemble"), "confidence": conf,
                "trusted": d.get("trusted"), "reasoning": d.get("reasoning")}
        if dry_run:
            print(f"  [dry-run] {info}")
            continue
        res = broker.open(sym, action, vol, sl, tp, f"ta:{profile.name}")
        info["position"] = res.get("positionId")
        journal(profile, info)
        positions.append({"symbol": sym})


# --------------------------------------------------------------------------- status
def status(profile, broker):
    acct = broker.account()
    print(f"[{profile.name}] account {acct.get('login')} on {acct.get('server')} ({acct.get('type')}) "
          f"balance {acct.get('balance')} equity {acct.get('equity')} {acct.get('currency')}")
    for p in broker.positions():
        if p.get("symbol") in profile.symbols:
            print(f"  {p['symbol']} {side_of(p)} {p.get('volume')} @ {p.get('openPrice')} "
                  f"SL {p.get('stopLoss')} TP {p.get('takeProfit')} P/L {p.get('profit')}")
    j = profile.state_dir / "journal.jsonl"
    if j.exists():
        lines = j.read_text().splitlines()[-8:]
        print("  recent journal:")
        for ln in lines:
            e = json.loads(ln)
            print(f"   {e['time_utc']} {e['event']} {e.get('symbol')} {e.get('side') or e.get('action') or ''} {e.get('why') or ''}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trading_agent")
    ap.add_argument("command", choices=["analyze", "execute", "status", "calibrate"])
    ap.add_argument("--profile", default="forex")
    ap.add_argument("--paper", action="store_true", help="synthetic data + simulated account")
    ap.add_argument("--dry-run", action="store_true", help="execute: show what would be traded")
    args = ap.parse_args(argv)
    profile = config.get_profile(args.profile)
    broker = get_broker(profile, paper=args.paper)
    if args.command == "analyze":
        analyze(profile, broker)
    elif args.command == "execute":
        execute(profile, broker, dry_run=args.dry_run)
    elif args.command == "status":
        status(profile, broker)
    elif args.command == "calibrate":
        run_calibration(profile, broker)
