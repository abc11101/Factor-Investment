"""
Retrieve stock prices and P/E ratios with yfinance.

Usage:
  python fetch_prices_pe.py                      # default tickers
  python fetch_prices_pe.py AAPL MSFT NVDA       # custom tickers
  python fetch_prices_pe.py AAPL --period 5y --interval 1mo
"""
import argparse

import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


def get_prices(tickers, period="1y", interval="1d"):
    """Adjusted close prices, one column per ticker."""
    data = yf.download(tickers, period=period, interval=interval,
                       auto_adjust=True, progress=False)
    prices = data["Close"]
    if isinstance(prices, pd.Series):        # single ticker
        prices = prices.to_frame(tickers[0])
    return prices.dropna(how="all")


def get_pe(tickers):
    """Current price, trailing/forward P/E and EPS for each ticker."""
    rows = []
    for t in tickers:
        info = yf.Ticker(t).info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        eps_ttm = info.get("trailingEps")
        pe_ttm = info.get("trailingPE")
        if pe_ttm is None and price and eps_ttm and eps_ttm > 0:
            pe_ttm = price / eps_ttm             # fallback: compute it ourselves
        rows.append({
            "ticker": t,
            "price": price,
            "trailing_pe": pe_ttm,
            "forward_pe": info.get("forwardPE"),
            "eps_ttm": eps_ttm,
            "eps_fwd": info.get("forwardEps"),
            "market_cap": info.get("marketCap"),
        })
    return pd.DataFrame(rows).set_index("ticker")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS)
    ap.add_argument("--period", default="1y")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--save", action="store_true", help="write CSVs")
    args = ap.parse_args()
    tickers = [t.upper() for t in args.tickers]

    prices = get_prices(tickers, args.period, args.interval)
    pe = get_pe(tickers)

    pd.set_option("display.float_format", "{:,.2f}".format)
    print(f"\nPrices (last 5 rows, {args.period} @ {args.interval}):")
    print(prices.tail())
    print("\nValuation:")
    print(pe)

    if args.save:
        prices.to_csv("prices.csv")
        pe.to_csv("pe_ratios.csv")
        print("\nSaved prices.csv and pe_ratios.csv")


if __name__ == "__main__":
    main()
