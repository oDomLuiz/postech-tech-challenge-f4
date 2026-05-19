from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timezone
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

from src.utils import normalize_ticker, raw_data_path


def _flatten_yfinance_columns(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    data = data.reset_index()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            column[0] if column[1] in ("", symbol) else "_".join(map(str, column))
            for column in data.columns
        ]
    return data


def _validate_download(data: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
    if data.empty:
        raise ValueError(f"No data returned for ticker {symbol} from {source}.")
    if "Close" not in data.columns:
        raise ValueError(f"Downloaded data for {symbol} from {source} has no Close column.")
    return data


def _date_to_epoch(value: str) -> int:
    parsed_date = date.fromisoformat(value)
    parsed_datetime = datetime.combine(parsed_date, time.min, tzinfo=timezone.utc)
    return int(parsed_datetime.timestamp())


def _download_from_yahoo_chart(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    period1 = _date_to_epoch(start_date)
    period2 = _date_to_epoch(end_date)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    result = payload.get("chart", {}).get("result")
    if not result:
        raise ValueError(f"No data returned for ticker {symbol} from Yahoo chart API.")

    chart = result[0]
    timestamps = chart.get("timestamp", [])
    quote = chart.get("indicators", {}).get("quote", [{}])[0]
    adjusted = chart.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])

    data = pd.DataFrame(
        {
            "Date": [datetime.fromtimestamp(item, tz=timezone.utc).date().isoformat() for item in timestamps],
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
            "Adj Close": adjusted or quote.get("close", []),
            "Volume": quote.get("volume", []),
        }
    )
    return _validate_download(data.dropna(subset=["Close"]), symbol, "Yahoo chart API")


def _download_from_yfinance(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    data = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return _validate_download(_flatten_yfinance_columns(data, symbol), symbol, "Yahoo Finance")


def _download_from_stooq(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    stooq_symbol = symbol.lower()
    if "." not in stooq_symbol:
        stooq_symbol = f"{stooq_symbol}.us"

    url = (
        "https://stooq.com/q/d/l/"
        f"?s={stooq_symbol}&d1={start_date.replace('-', '')}"
        f"&d2={end_date.replace('-', '')}&i=d"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        data = pd.read_csv(response)
    return _validate_download(data, symbol, "Stooq")


def download_stock_data(ticker: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    symbol = normalize_ticker(ticker)
    final_end_date = end_date or date.today().isoformat()

    for downloader in (_download_from_yahoo_chart, _download_from_yfinance):
        try:
            return downloader(symbol, start_date, final_end_date)
        except Exception:
            continue

    try:
        return _download_from_stooq(symbol, start_date, final_end_date)
    except Exception as error:
        raise ValueError(
            f"Could not download data for {symbol}. "
            "Yahoo chart API, yfinance and Stooq all failed. "
            "Check your internet connection, proxy/VPN/firewall, ticker symbol, "
            "or try passing --end-date explicitly."
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Download historical stock prices.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol, e.g. AAPL.")
    parser.add_argument("--start-date", default="2018-01-01", help="YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD. Defaults to today.")
    args = parser.parse_args()

    data = download_stock_data(args.ticker, args.start_date, args.end_date)
    output_path = raw_data_path(args.ticker)
    data.to_csv(output_path, index=False)
    print(f"Saved {len(data)} rows to {output_path}")


if __name__ == "__main__":
    main()
