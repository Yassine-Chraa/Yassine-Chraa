"""Broker access.

MetaApiBroker talks to MetaApi's REST API, which connects to the Exness MT5
account in the cloud, so no Windows MetaTrader terminal is needed.
PaperBroker uses synthetic candles and simulated fills, for testing offline.
"""
from __future__ import annotations

import json
import math
import zlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from . import config

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


class BrokerError(RuntimeError):
    pass


class MetaApiBroker:
    def __init__(self, account_id=None, token=None, region=None, timeout=30):
        self.account_id = account_id or config.METAAPI_ACCOUNT_ID
        if not self.account_id:
            raise BrokerError("METAAPI_ACCOUNT_ID is not set.")
        region = region or config.METAAPI_REGION
        self.client = f"https://mt-client-api-v1.{region}.agiliumtrade.ai/users/current/accounts/{self.account_id}"
        self.market = f"https://mt-market-data-client-api-v1.{region}.agiliumtrade.ai/users/current/accounts/{self.account_id}"
        self.s = requests.Session()
        # In a Claude cloud session the proxy replaces this header with the saved secret.
        self.s.headers["auth-token"] = token or config.METAAPI_TOKEN
        self.timeout = timeout

    def _req(self, method, url, **kw):
        r = self.s.request(method, url, timeout=self.timeout, **kw)
        if r.status_code >= 400:
            raise BrokerError(f"{method} {url.split('/accounts/')[-1]} -> HTTP {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else None

    # ---- market data -------------------------------------------------------
    def candles(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        rows, start = [], None
        while len(rows) < count:
            params = {"limit": min(1000, count - len(rows))}
            if start:
                params["startTime"] = start
            batch = self._req("GET", f"{self.market}/historical-market-data/symbols/{symbol}/timeframes/{timeframe}/candles", params=params)
            if not batch:
                break
            rows.extend(batch)
            earliest = min(c["time"] for c in batch)
            start = (pd.Timestamp(earliest) - pd.Timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
            if len(batch) < params["limit"]:
                break
        if not rows:
            raise BrokerError(f"No candles for {symbol} {timeframe}")
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        df = df.drop_duplicates("time").sort_values("time").set_index("time")
        df["volume"] = df.get("tickVolume", df.get("volume", 0))
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    def price(self, symbol):
        return self._req("GET", f"{self.client}/symbols/{symbol}/current-price")

    def spec(self, symbol):
        return self._req("GET", f"{self.client}/symbols/{symbol}/specification")

    # ---- account -----------------------------------------------------------
    def account(self):
        return self._req("GET", f"{self.client}/account-information")

    def is_demo(self) -> bool:
        return "DEMO" in str(self.account().get("type", "")).upper()

    def positions(self):
        return self._req("GET", f"{self.client}/positions") or []

    def deals_since(self, start: datetime):
        end = datetime.now(timezone.utc) + timedelta(minutes=5)
        f = lambda t: t.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        return self._req("GET", f"{self.client}/history-deals/time/{f(start)}/{f(end)}") or []

    # ---- trading -----------------------------------------------------------
    def _trade(self, body):
        res = self._req("POST", f"{self.client}/trade", json=body)
        if str(res.get("stringCode")) not in ("TRADE_RETCODE_DONE", "TRADE_RETCODE_PLACED", "ERR_NO_ERROR"):
            raise BrokerError(f"Trade refused: {res}")
        return res

    def open(self, symbol, side, volume, sl, tp, comment):
        return self._trade({
            "actionType": "ORDER_TYPE_BUY" if side == "BUY" else "ORDER_TYPE_SELL",
            "symbol": symbol, "volume": volume, "stopLoss": sl, "takeProfit": tp,
            "comment": comment[:26],
        })

    def close(self, position_id):
        return self._trade({"actionType": "POSITION_CLOSE_ID", "positionId": str(position_id)})


class PaperBroker:
    """Synthetic market + simulated account. State lives in a JSON file."""

    def __init__(self, state_file):
        self.file = state_file
        try:
            self.st = json.loads(self.file.read_text())
        except Exception:
            self.st = {"balance": 1000.0, "positions": [], "deals": [], "next_id": 1}

    def _save(self):
        self.file.write_text(json.dumps(self.st, indent=1))

    def candles(self, symbol, timeframe, count):
        seed = zlib.crc32(symbol.encode())
        rng = np.random.default_rng(seed)
        base = {"XAU": 2000.0, "BTC": 60000.0, "ETH": 3000.0, "JPY": 150.0}.get(
            next((k for k in ("XAU", "BTC", "ETH", "JPY") if k in symbol), ""), 1.1)
        vol = 0.0025
        rets = rng.normal(0, vol, count) + 0.0006 * np.sin(np.arange(count) / 90)
        close = base * np.exp(np.cumsum(rets))
        opn = np.r_[close[0], close[:-1]]
        spread = np.abs(rng.normal(0, vol, count)) * close
        end = pd.Timestamp.utcnow().tz_localize(None).floor("h")
        idx = pd.date_range(end=end, periods=count, freq=f"{TF_MINUTES[timeframe]}min")
        return pd.DataFrame({"open": opn, "high": np.maximum(opn, close) + spread,
                             "low": np.minimum(opn, close) - spread, "close": close,
                             "volume": 100.0}, index=idx)

    def price(self, symbol):
        c = float(self.candles(symbol, "1h", 50).close.iloc[-1])
        return {"bid": c, "ask": c * 1.0001, "lossTickValue": 1.0, "profitTickValue": 1.0}

    def spec(self, symbol):
        c = float(self.candles(symbol, "1h", 50).close.iloc[-1])
        tick = 10 ** (math.floor(math.log10(c)) - 4)
        return {"symbol": symbol, "tickSize": tick, "minVolume": 0.01, "maxVolume": 100, "volumeStep": 0.01}

    def account(self):
        return {"type": "ACCOUNT_TRADE_MODE_DEMO", "balance": self.st["balance"], "equity": self.st["balance"],
                "currency": "USD", "login": "paper", "server": "paper"}

    def is_demo(self):
        return True

    def positions(self):
        return self.st["positions"]

    def deals_since(self, start):
        return [d for d in self.st["deals"] if d["time"] >= start.isoformat()]

    def open(self, symbol, side, volume, sl, tp, comment):
        p = self.price(symbol)
        pos = {"id": str(self.st["next_id"]), "symbol": symbol, "volume": volume, "stopLoss": sl, "takeProfit": tp,
               "type": "POSITION_TYPE_BUY" if side == "BUY" else "POSITION_TYPE_SELL", "comment": comment,
               "openPrice": p["ask"] if side == "BUY" else p["bid"], "profit": 0.0,
               "time": datetime.now(timezone.utc).isoformat()}
        self.st["next_id"] += 1
        self.st["positions"].append(pos)
        self._save()
        return {"stringCode": "TRADE_RETCODE_DONE", "positionId": pos["id"]}

    def close(self, position_id):
        self.st["positions"] = [p for p in self.st["positions"] if p["id"] != str(position_id)]
        self._save()
        return {"stringCode": "TRADE_RETCODE_DONE"}


def get_broker(profile, paper=False):
    if paper:
        return PaperBroker(profile.state_dir / "paper_account.json")
    return MetaApiBroker()
