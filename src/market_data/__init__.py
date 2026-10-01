"""
The Market Data Engine.

    candles.py     Candle (one OHLCV bar), DataKind, errors
    validation.py  checks every candle; rejects bad data, never repairs it
    dataset.py     MarketDataSet: validated + labelled candles for one symbol
    indicators.py  SMA 20/50, RSI 14, VWAP, average volume
    providers.py   the provider interface + offline CSV / in-memory providers

Typical use:
    from src.market_data import CSVHistoricalProvider, compute_indicators
    data = CSVHistoricalProvider().get_candles("SPY", min_candles=50)
    print(compute_indicators(data).describe())

This package never makes up prices. Missing or bad data raises
MarketDataError. It does not connect to Robinhood or any broker, and the
paper trader does not import it (prices reach the trader as a plain dict).
"""

from src.market_data.candles import (Candle, DataKind, InsufficientDataError,
                                     MarketDataError)
from src.market_data.dataset import MarketDataSet, latest_prices
from src.market_data.indicators import (MIN_CANDLES_FOR_INDICATORS,
                                        IndicatorSnapshot, average_volume,
                                        compute_indicators, latest_session,
                                        rsi, session_vwap, sma, vwap)
from src.market_data.providers import (CSVHistoricalProvider, InMemoryProvider,
                                       MarketDataProvider, clean_symbol)
from src.market_data.validation import find_problems, validate_candles
