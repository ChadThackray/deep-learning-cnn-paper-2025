"""Yahoo Finance data fetching."""

from datetime import date

import pandas as pd
import yfinance as yf


def fetch_ohlc(
    ticker: str,
    start: date,
    end: date,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLC data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        start: Start date
        end: End date
        interval: Data interval ('1d', '1h', '5m', etc.)

    Returns:
        DataFrame with columns: open, high, low, close (lowercase)

    Raises:
        ValueError: If no data is returned or ticker is invalid
    """
    stock = yf.Ticker(ticker)

    df = stock.history(
        start=start.isoformat(),
        end=end.isoformat(),
        interval=interval,
    )

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
    })

    return df[["open", "high", "low", "close"]].copy()
