# Trading agent (cloud version)

Rebuild of the MT5 forex/crypto autopilot so it runs without a Windows PC.
It trades the Exness MT5 **demo** account through [MetaApi](https://metaapi.cloud)
and is driven by an hourly Claude scheduled task, which you follow from the Claude app on your phone.

## How it decides
1. **Theories** (`strategies.py`): EMA cross, RSI reversal, MACD momentum, Donchian breakout,
   Bollinger reversion, market structure break, Supertrend flip, Stochastic cross. Each also has a
   `+htf` version that only trades with the 4h and daily trend.
2. **Calibration** (`backtest.py`): about a year of H1 candles is replayed. A theory only gets weight
   on a symbol if it also made money on the last 35 % of the data (out of sample). Weight = its OOS average R.
3. **Vote**: weighted theories vote; a vote stays active for 3 candles (6 for gold).
4. **Claude**: reads the vote, the multi-timeframe context and the news, and writes a decision (see `RUNBOOK.md`).
5. **Execute** (`agent.py`): trades only when vote and Claude agree, with 1 % risk, ATR stop/target,
   max 3 positions, −3 % daily stop, no new forex trades after Friday 18:00 UTC, demo account only.

## Commands
```
pip install -r requirements.txt
python -m trading_agent calibrate --profile forex
python -m trading_agent analyze   --profile forex
python -m trading_agent execute   --profile forex [--dry-run]
python -m trading_agent status    --profile forex
```
Add `--paper` to use synthetic data and a simulated account (for testing only).

## Settings
`trading_agent/config.py` holds symbols, days, risk and vote settings.
Environment: `METAAPI_ACCOUNT_ID`, `METAAPI_REGION` (e.g. `london`), and the MetaApi token
(`METAAPI_TOKEN`, or injected by the Claude cloud proxy as the `auth-token` header).
