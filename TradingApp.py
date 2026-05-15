# ========== BASIC LIBS ============
from ibapi.client import EClient 
from ibapi.wrapper import EWrapper 
from ibapi.contract import Contract 
from ibapi.order import Order
from ibapi.common import BarData
import time
import threading
from datetime import datetime
from typing import Dict, List
import pandas as pd 
import logging
import numpy as np 

# =========================
# Logging config (file + terminal)
# =========================
file_handler = logging.FileHandler("historical_fetch.log", mode="w", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
    force=True
)
logger = logging.getLogger(__name__)

class TradingApp(EWrapper, EClient):
    def __init__(self, apply_us_stock_volume_x100: bool = False):
        EClient.__init__(self, self)
        self.data: Dict[int, pd.DataFrame] = {}
        self._done = threading.Event()
        self.apply_us_stock_volume_x100 = apply_us_stock_volume_x100

        # signals/buffers
        self._connected = threading.Event()
        self._cd_event = threading.Event()
        self._cd_results: Dict[int, List] = {}
        self._historical_data_events: Dict[int, threading.Event] = {}

    # -------------------- CONNECTION --------------------
    def nextValidId(self, orderId: int):
        logging.info(f"nextValidId={orderId} (connected)")
        self._connected.set()

    def connect_and_run(self, host: str, port: int, clientId: int):
        self.connect(host, port, clientId)
        thread = threading.Thread(target=self.run)
        thread.start()
        self._connected.wait(timeout=10) # Wait for connection to be established
        if not self._connected.is_set():
            logging.error("Failed to connect to IB TWS/Gateway.")
            raise ConnectionError("Could not connect to IB TWS/Gateway.")

    def disconnect(self):
        # Corrected: Call the disconnect method from the parent class (EClient)
        super().disconnect()
        logging.info("Disconnected from IB TWS/Gateway.")

    # -------------------- HISTORICAL DATA --------------------
    def get_historical_data(
        self,
        reqId: int,
        contract: Contract,
        durationStr: str = "1 D",
        barSizeSetting: str = "1 min",
        whatToShow: str = "TRADES",
        endDateTime: str | None = None,
        *,
        historical_timeout: float = 120.0,
    ) -> pd.DataFrame:
        if endDateTime is None:
            endDateTime = datetime.now().strftime("%Y%m%d %H:%M:%S US/Eastern")
        self.data[reqId] = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "trades", "wap"]
        )
        self.data[reqId].index.name = "time"
        self._historical_data_events[reqId] = threading.Event()

        self.reqHistoricalData(
            reqId=reqId,
            contract=contract,
            durationStr=durationStr,
            barSizeSetting=barSizeSetting,
            whatToShow=whatToShow,
            endDateTime=endDateTime,             # "" = now
            useRTH=0,
            formatDate=2,
            keepUpToDate=False,
            chartOptions=[],
        )

        # Wait for historical data to be received
        self._historical_data_events[reqId].wait(timeout=historical_timeout)
        if not self._historical_data_events[reqId].is_set():
            logging.warning(f"Timeout waiting for historical data for reqId {reqId}")

        return self.data[reqId]

    def historicalData(self, reqId: int, bar: BarData) -> None:
        df = self.data[reqId]
        # Try to read it as a timestamp string w Hour
        ts = pd.to_datetime(bar.date, format="%Y%m%d %H:%M:%S", errors="coerce")
        if pd.isna(ts):
            # Try to read it as a timestamp string w Data (only)
            ts = pd.to_datetime(bar.date, format="%Y%m%d", errors="coerce")
        # If it is numerical (epoch seconds)
        if pd.isna(ts):
            try:
                ts = pd.to_datetime(int(bar.date), unit="s", utc=True, errors="coerce")
            except ValueError:
                pass  # It is Nat if it's not a string or epoch 

        vol = bar.volume * (100 if self.apply_us_stock_volume_x100 else 1)
        trades = int(getattr(bar, "barCount", 0) or 0)
        wap_raw = getattr(bar, "average", float("nan"))
        try:
            wap = float(wap_raw)
        except (TypeError, ValueError):
            wap = float("nan")
        df.loc[ts, ["open", "high", "low", "close", "volume", "trades", "wap"]] = [
            bar.open, bar.high, bar.low, bar.close, vol, trades, wap,
        ]

        self.data[reqId] = df.astype({
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
            "trades": "int64",
            "wap": "float64",
        })

        
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        logging.info(f"Historical data for reqId {reqId} ended. Start: {start}, End: {end}")
        if reqId in self._historical_data_events:
            self._historical_data_events[reqId].set()

    # -------------------- CONTRACT AUX --------------------
    @staticmethod
    def get_stock_contract(symbol: str) -> Contract:
        c = Contract()
        c.symbol = symbol
        c.secType = "STK"
        c.exchange = "SMART"
        c.currency = "USD"
        c.primaryExchange = "NASDAQ"  # helps disambiguate
        return c

    @staticmethod
    def get_fx_contract(symbol: str, currency: str = 'USD') -> Contract: #Written to retrieve FX data of currency pairs
        c = Contract()
        c.symbol = symbol
        c.secType = "CASH"
        c.exchange = "IDEALPRO"
        c.currency = currency
        return c

    # -------------------- CONTRACT DETAILS (STOCKS) --------------------
    def request_stock_contract_details(self, symbol: str, req_id: int = 1001):
        c = Contract()
        c.symbol = symbol
        c.secType = "STK"
        c.currency = "USD"
        c.exchange = "SMART"
        c.primaryExchange = "NASDAQ"

        self._cd_results[req_id] = []
        self._cd_event.clear()
        self.reqContractDetails(req_id, c)

    def contractDetails(self, reqId: int, details) -> None:
        self._cd_results.setdefault(reqId, []).append(details)
        c = details.contract
        logging.info(f"[CONTRACT] reqId={reqId} conId={c.conId} "
                     f"symbol={c.symbol} localSymbol={c.localSymbol} "
                     f"exchange={c.exchange}/{getattr(c, 'primaryExchange', '')} currency={c.currency}")
        logging.info(f"[END] reqId={reqId}")
        self._cd_event.set()

    def wait_contract_details(self, timeout: float = 20.0) -> Dict[int, List]:
        self._cd_event.wait(timeout=timeout)
        return dict(self._cd_results)

    # -------------------- DF helper --------------------
    def get_contract_details_df(self, req_id: int) -> pd.DataFrame:
        recs = []
        for d in self._cd_results.get(req_id, []):
            c = d.contract
            recs.append({
                "conId": c.conId,
                "symbol": c.symbol,
                "localSymbol": c.localSymbol,
                "exchange": c.exchange,
                "primaryExchange": getattr(c, "primaryExchange", ""),
                "currency": c.currency,
                "tradingClass": getattr(c, "tradingClass", ""),
                "minTick": getattr(d, "minTick", None),
                "longName": getattr(d, "longName", ""),
                "timeZoneId": getattr(d, "timeZoneId", ""),
                "tradingHours": getattr(d, "tradingHours", ""),
                "liquidHours": getattr(d, "liquidHours", "")
            })
        return pd.DataFrame.from_records(recs)

    # -------------------- VOLATILITY CALCULATION --------------------
    @staticmethod
    def calculate_daily_returns(df: pd.DataFrame) -> pd.Series:
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain a \'close\' column.")
        return df["close"].pct_change().dropna()
    @staticmethod
    def calculate_annualized_volatility(returns: pd.Series, trading_days_per_year: int = 252) -> float:
        if returns.empty:
            return 0.0
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(trading_days_per_year)
        return annualized_volatility