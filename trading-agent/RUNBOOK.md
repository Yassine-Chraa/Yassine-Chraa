# Runbook for Claude's scheduled run

This is the procedure the scheduled task follows each time it fires (once an hour).
Claude is the decision maker; the code only proposes, sizes and executes. No trade is
opened unless **both** the weighted theory vote and Claude's decision point the same way.

Run everything from `trading-agent/`. `PROFILE` is `forex` (Mon–Fri) or `crypto` (Fri–Sun).

## 1. Analyze
```
pip install -q -r requirements.txt
python -m trading_agent analyze --profile PROFILE
```
If `new trades allowed: False` and there are no open positions, stop here and report "market closed / outside trading window".
If `state/PROFILE/theory_weights.json` is missing or older than 30 days, run `python -m trading_agent calibrate --profile PROFILE` first.

## 2. News
Search the web for news from the last 24 hours that moves each currency/asset in the profile
(central banks, CPI/NFP/GDP releases, risk sentiment, for gold: real yields and USD; for crypto: equities risk-on/off, ETF flows).
Check the economic calendar for high-impact events in the next 2 hours: if one is due for a symbol's currency, HOLD that symbol.

## 3. Decide
Read `state/PROFILE/analysis.json`. For every symbol write one decision to `state/PROFILE/decisions.json`:

```json
{
  "generated_utc": "<now, ISO 8601 with +00:00>",
  "decisions": [
    {"symbol": "EURUSDm", "action": "BUY|SELL|HOLD", "confidence": 0.0,
     "news": "supports|neutral|contradicts", "trusted": ["theory names you rely on"],
     "close_open_trade": false, "reasoning": "two or three sentences"}
  ]
}
```

Rules:
- Only BUY/SELL in the direction of `ensemble_side`. If it is `NONE`, HOLD.
- Prefer theories with more out-of-sample trades and higher `oos.avg_r`; list them in `trusted`.
- Lower confidence when price is stretched (4h RSI < 25 or > 75 against the trade's entry), close to the previous day's high/low in the trade's direction (< 0.3 D1 ATR), or the news contradicts.
- `confidence` ≥ 0.55 is needed to trade; be honest, HOLD is the normal answer.
- `close_open_trade: true` only when the reason for an open position is clearly gone (trend flipped on both higher timeframes, or major news against it).

## 4. Execute
```
python -m trading_agent execute --profile PROFILE
```
The code re-checks every guard: demo-only stage, trading days, Friday cutoff, confidence, news,
vote agreement, max 3 positions, −3 % daily loss stop, 1 % risk per trade with ATR stop and target.

## 5. Save and report
Commit `state/` (journal, decisions, analysis) and push, so the next run and the person can see history.
Finish with a short summary: account equity, open positions, what was traded or why nothing was.
