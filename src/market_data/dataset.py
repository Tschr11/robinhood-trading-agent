"""
dataset.py - A validated, labelled series of candles for one symbol.

A MarketDataSet checks itself the moment it is created. If anything is
wrong, creating it fails with MarketDataError - so any MarketDataSet that
exists in the program is guaranteed to hold valid candles.

It is also labelled:
    kind    DataKind.HISTORICAL or DataKind.LIVE
    source  where it came from, e.g. "csv:data/market/SPY.csv"
"""

from dataclasses import dataclass

from src.market_data.candles import Candle, DataKind, MarketDataError
from src.market_data.validation import find_problems


@dataclass(frozen=True)
class MarketDataSet:
    symbol: str
    kind: DataKind
    source: str
    candles: tuple[Candle, ...]   # oldest first

    def __post_init__(self):
        problems = []
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            problems.append(f"Symbol must be a ticker such as 'SPY' (got {self.symbol!r}).")
        else:
            # frozen dataclass: object.__setattr__ is how we tidy a field once.
            object.__setattr__(self, "symbol", self.symbol.strip().upper())
        if not isinstance(self.kind, DataKind):
            problems.append("kind must be DataKind.HISTORICAL or DataKind.LIVE "
                            f"(got {self.kind!r}).")
        if not isinstance(self.source, str) or not self.source.strip():
            problems.append(f"source must say where the data came from (got {self.source!r}).")
        if isinstance(self.candles, list):
            object.__setattr__(self, "candles", tuple(self.candles))
        problems += find_problems(self.candles)
        if isinstance(self.candles, tuple) and not self.candles:
            problems.append("A dataset needs at least 1 candle.")
        if problems:
            raise MarketDataError(problems)

    # -- Labels -----------------------------------------------------------------

    @property
    def is_historical(self) -> bool:
        return self.kind is DataKind.HISTORICAL

    @property
    def is_live(self) -> bool:
        return self.kind is DataKind.LIVE

    def require_kind(self, expected: DataKind) -> "MarketDataSet":
        """
        Stop if this is the wrong kind of data, e.g. so historical prices
        are never treated as live ones. Returns the dataset for chaining.
        """
        if self.kind is not expected:
            raise MarketDataError(
                f"{self.symbol} data from {self.source} is {self.kind.value}, "
                f"but {expected.value} data is required.")
        return self

    # -- Convenient views ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.candles)

    @property
    def closes(self) -> list[float]:
        return [c.close for c in self.candles]

    @property
    def latest(self) -> Candle:
        return self.candles[-1]

    @property
    def latest_close(self) -> float:
        return self.candles[-1].close

    def last(self, count: int) -> "MarketDataSet":
        """A new dataset with only the most recent `count` candles."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise MarketDataError(f"count must be a whole number of at least 1 (got {count!r}).")
        return MarketDataSet(self.symbol, self.kind, self.source, self.candles[-count:])


def latest_prices(datasets) -> dict[str, float]:
    """
    Turn datasets into the plain {"SPY": 494.00} dict that the paper
    trader's check_exits() accepts. This is the ONLY bridge between market
    data and trading: the paper trader never imports this package.
    """
    prices = {}
    for dataset in datasets:
        if not isinstance(dataset, MarketDataSet):
            raise MarketDataError(f"Expected a MarketDataSet (got {type(dataset).__name__}).")
        if dataset.symbol in prices:
            raise MarketDataError(f"{dataset.symbol} was supplied more than once.")
        prices[dataset.symbol] = dataset.latest_close
    return prices
