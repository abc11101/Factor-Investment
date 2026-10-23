"""
Retrieve stock prices and P/E ratios with yfinance.

Usage:
  python fetch_prices_pe.py                      # prompts for tickers/period/interval
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


def ask(prompt, default):
    """Prompt the user; empty answer keeps the default."""
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def prompt_for_inputs(args):
    """Fill in args interactively when no tickers were given on the command line."""
    raw = ask("Tickers (space or comma separated)", " ".join(DEFAULT_TICKERS))
    args.tickers = raw.replace(",", " ").split()
    args.period = ask("Period (1mo, 6mo, 1y, 5y, max, ...)", args.period)
    args.interval = ask("Interval (1d, 1wk, 1mo, ...)", args.interval)
    if not args.save:
        args.save = ask("Save CSVs? (y/n)", "n").lower().startswith("y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--period", default="1y")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--save", action="store_true", help="write CSVs")
    args = ap.parse_args()
    if not args.tickers:
        prompt_for_inputs(args)
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
