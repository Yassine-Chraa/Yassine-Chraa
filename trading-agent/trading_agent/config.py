"""Settings for each trading profile (forex, crypto).

Everything a person might want to tune lives here. Secrets (the MetaApi token
and account id) never do: they come from environment variables, or, in a
Claude cloud session, from the network secret the proxy injects.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_ROOT = ROOT / "state"


@dataclass
class Profile:
    name: str
    symbols: list[str]
    trading_days: str                 # letters of UTC weekdays allowed, Mon..Sun = "MTWTFSS"
    timeframe: str = "1h"             # entry timeframe; the scheduled task runs once per candle
    htf: tuple[str, str] = ("4h", "1D")   # higher timeframes for the trend filter and context
    history_bars: int = 1500          # entry-timeframe candles fetched for a live run
    calibrate_bars: int = 6500        # about one year of H1 candles for calibration

    # Voting
    vote_memory_bars: int = 3         # a theory's vote stays active this many candles after it fires
    vote_memory_by_symbol: dict = field(default_factory=dict)
    ensemble_threshold: float = 0.25  # |weighted vote| needed before a trade is even considered

    # Calibration (out-of-sample filter that decides which theories carry weight)
    oos_fraction: float = 0.35        # last 35% of history is the out-of-sample test
    min_oos_trades: int = 8
    min_oos_avg_r: float = 0.05

    # Risk
    risk_per_trade: float = 0.01      # 1% of equity per trade
    sl_atr: float = 1.5               # stop loss = 1.5 x ATR(14)
    tp_atr: float = 2.5               # take profit = 2.5 x ATR(14)
    max_open_positions: int = 3
    max_daily_loss: float = 0.03      # no new trades after -3% on the UTC day
    min_claude_confidence: float = 0.55
    decision_max_age_min: int = 90    # a Claude decision older than this is ignored
    no_new_trades_after_friday_utc: int | None = 18   # weekend-gap protection (forex)

    # Stage: DEMO trades only on a demo account; LIVE needs ALLOW_LIVE_TRADING=1 too.
    stage: str = "DEMO"

    @property
    def state_dir(self) -> Path:
        d = STATE_ROOT / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def vote_memory(self, symbol: str) -> int:
        return self.vote_memory_by_symbol.get(symbol, self.vote_memory_bars)


PROFILES: dict[str, Profile] = {
    "forex": Profile(
        name="forex",
        symbols=["EURUSDm", "GBPUSDm", "XAUUSDm", "USDJPYm", "AUDUSDm", "AUDJPYm"],
        trading_days="MTWTF__",
        vote_memory_by_symbol={"XAUUSDm": 6},   # gold did better with a longer memory in replays
    ),
    "crypto": Profile(
        name="crypto",
        symbols=["BTCUSDm", "ETHUSDm"],
        trading_days="____FSS",
        no_new_trades_after_friday_utc=None,
    ),
}


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise SystemExit(f"Unknown profile {name!r}. Choose from: {', '.join(PROFILES)}")
    return PROFILES[name]


# Broker connection (MetaApi). Region is where MetaApi hosts the account, e.g. "london", "new-york".
METAAPI_TOKEN = os.environ.get("METAAPI_TOKEN", "proxy-injected")
METAAPI_ACCOUNT_ID = os.environ.get("METAAPI_ACCOUNT_ID", "")
METAAPI_REGION = os.environ.get("METAAPI_REGION", "london")
ALLOW_LIVE_TRADING = os.environ.get("ALLOW_LIVE_TRADING") == "1"
