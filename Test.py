import argparse
import numpy as np
import pandas as pd
import yfinance as yf

MARKET = "^GSPC"
DAYS_1M, DAYS_6M, DAYS_12M = 21, 126, 252


def get_prices(tickers, period="1y", interval="1d"):
    """Adjusted close prices, one column per ticker."""
    data = yf.download(tickers, period=period, interval=interval,
                       auto_adjust=True, progress=False)
    prices = data["Close"]
    if isinstance(prices, pd.Series):        # single ticker
        prices = prices.to_frame(tickers[0])
    return prices.dropna(how="all")


def div(a, b):
    """a / b, or NaN when either side is missing or b is zero."""
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def _row(df, names):
    """First matching statement row (newest period first), or None."""
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            return df.loc[n].sort_index(ascending=False)
    return None


def latest(df, *names):
    r = _row(df, names)
    r = r.dropna() if r is not None else []
    return r.iloc[0] if len(r) else np.nan


def ttm(quarterly, annual, *names):
    """Sum of the last 4 quarters if all are reported, else latest annual value."""
    r = _row(quarterly, names)
    if r is not None and len(r) >= 4 and r.iloc[:4].notna().all():
        return r.iloc[:4].sum()
    return latest(annual, *names)


def growth(annual, *names):
    """Latest annual value vs the prior year."""
    r = _row(annual, names)
    r = r.dropna() if r is not None else []
    if len(r) < 2:
        return np.nan
    return div(r.iloc[0], r.iloc[1]) - 1


def get_market_factors(tickers):
    """Beta, momentum, return, volatility and turnover from daily price/volume."""
    data = yf.download(tickers + [MARKET], period="5y", interval="1d",
                       auto_adjust=True, progress=False)
    close, volume = data["Close"], data["Volume"]
    monthly = close.resample("ME").last().pct_change()

    rows = {}
    for t in tickers:
        p = close[t].dropna()
        m = monthly[[t, MARKET]].dropna().tail(36)
        daily = p.pct_change().tail(DAYS_12M)
        rows[t] = {
            "beta_36m": m[t].cov(m[MARKET]) / m[MARKET].var() if len(m) >= 24 else np.nan,
            "mom_12m": p.iloc[-1 - DAYS_1M] / p.iloc[-1 - DAYS_12M] - 1 if len(p) > DAYS_12M else np.nan,
            "mom_6m": p.iloc[-1 - DAYS_1M] / p.iloc[-1 - DAYS_6M] - 1 if len(p) > DAYS_6M else np.nan,
            "ret_1m": p.iloc[-1] / p.iloc[-1 - DAYS_1M] - 1 if len(p) > DAYS_1M else np.nan,
            "volatility": daily.std() * np.sqrt(252),
            "avg_volume": volume[t].tail(DAYS_12M).mean(),
            "avg_dollar_volume": (volume[t] * close[t]).tail(DAYS_12M).mean(),
        }
    return pd.DataFrame(rows).T


def get_fundamentals(t):
    """Valuation, profitability, growth and payout characteristics for one ticker."""
    tk = yf.Ticker(t)
    info = tk.info
    qi, ai = tk.quarterly_income_stmt, tk.income_stmt
    qc, ac = tk.quarterly_cashflow, tk.cashflow
    qb, ab = tk.quarterly_balance_sheet, tk.balance_sheet

    def bs(*names):                          # most recent balance sheet value
        v = latest(qb, *names)
        return v if pd.notna(v) else latest(ab, *names)

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    shares = info.get("sharesOutstanding")
    mcap = info.get("marketCap") or (price * shares if price and shares else np.nan)

    revenue = ttm(qi, ai, "Total Revenue")
    gross_profit = ttm(qi, ai, "Gross Profit")
    ebit = ttm(qi, ai, "EBIT")
    ebitda = ttm(qi, ai, "EBITDA")
    net_income = ttm(qi, ai, "Net Income Common Stockholders", "Net Income")
    interest = ttm(qi, ai, "Interest Expense")
    tax_rate = latest(ai, "Tax Rate For Calcs")
    tax_rate = 0.21 if pd.isna(tax_rate) else tax_rate

    ocf = ttm(qc, ac, "Operating Cash Flow")
    capex = ttm(qc, ac, "Capital Expenditure")          # reported as negative
    fcf = ocf + capex if pd.notna(capex) else ocf
    ufcf = fcf + (0 if pd.isna(interest) else abs(interest)) * (1 - tax_rate)

    # Payout flows are reported as negative outflows / positive inflows
    dividends_paid = ttm(qc, ac, "Cash Dividends Paid", "Common Stock Dividend Paid")
    net_issuance = ttm(qc, ac, "Net Common Stock Issuance")
    if pd.isna(net_issuance):
        net_issuance = (np.nan_to_num(ttm(qc, ac, "Repurchase Of Capital Stock"))
                        + np.nan_to_num(ttm(qc, ac, "Issuance Of Capital Stock")))
    net_payout = -(np.nan_to_num(dividends_paid) + np.nan_to_num(net_issuance))

    equity = bs("Stockholders Equity", "Common Stock Equity")
    assets = bs("Total Assets")
    debt = bs("Total Debt")
    cash = bs("Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents")
    ev = mcap + np.nan_to_num(debt) - np.nan_to_num(cash)
    if pd.isna(ev):
        ev = info.get("enterpriseValue", np.nan)

    dps = tk.dividends
    if len(dps):
        dps = dps[dps.index >= pd.Timestamp.now(tz=dps.index.tz) - pd.DateOffset(years=1)]
    dps_ttm = float(dps.sum()) if len(dps) else 0.0

    return {
        "price": price,
        "market_cap": mcap,
        "enterprise_value": ev,
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "book_to_market": div(equity, mcap),
        "earnings_yield": div(net_income, mcap),
        "roe": div(net_income, equity),
        "gross_margin": div(gross_profit, revenue),
        "debt_to_assets": div(debt, assets),
        "revenue_growth": growth(ai, "Total Revenue"),
        "asset_growth": growth(ab, "Total Assets"),
        "turnover": None,                    # filled in from market data
        "ebit_ev": div(ebit, ev),
        "ebitda_ev": div(ebitda, ev),
        "sales_ev": div(revenue, ev),
        "gross_profit_ev": div(gross_profit, ev),
        "unlevered_fcf_ev": div(ufcf, ev),
        "sales_to_price": div(revenue, mcap),
        "cash_flow_yield": div(ocf, mcap),
        "fcf_yield": div(fcf, mcap),
        "dividend_per_share_ttm": dps_ttm,
        "dividend_yield": div(dps_ttm, price),
        "net_payout_yield": div(net_payout, mcap),
        "shares_outstanding": shares,
    }


def get_factors(tickers):
    """All characteristics, one row per ticker."""
    fund = pd.DataFrame({t: get_fundamentals(t) for t in tickers}).T
    mkt = get_market_factors(tickers)
    fund["turnover"] = mkt["avg_volume"] / fund["shares_outstanding"].astype(float)
    df = fund.join(mkt.drop(columns="avg_volume"))
    order = ["price", "market_cap", "enterprise_value", "beta_36m", "mom_12m", "mom_6m",
             "ret_1m", "volatility", "trailing_pe", "forward_pe", "book_to_market",
             "earnings_yield", "roe", "gross_margin", "debt_to_assets", "revenue_growth",
             "asset_growth", "turnover", "avg_dollar_volume", "ebit_ev", "ebitda_ev",
             "sales_ev", "gross_profit_ev", "unlevered_fcf_ev", "sales_to_price",
             "cash_flow_yield", "fcf_yield", "dividend_per_share_ttm", "dividend_yield",
             "net_payout_yield"]
    return df[order].astype(float)


def prompt_for_tickers():
    """Ask for tickers interactively when none were given on the command line."""
    tickers = []
    while not tickers:
        raw = input("Tickers (space or comma separated): ")
        tickers = raw.replace(",", " ").split()
    return tickers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--period", default="1y")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--save", action="store_true", help="write CSVs")
    args = ap.parse_args()
    if not args.tickers:
        args.tickers = prompt_for_tickers()
    tickers = [t.upper() for t in args.tickers]

    prices = get_prices(tickers, args.period, args.interval)
    factors = get_factors(tickers)

    pd.set_option("display.float_format", "{:,.4f}".format)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(f"\nPrices (last 5 rows, {args.period} @ {args.interval}):")
    print(prices.tail())
    print("\nFactors:")
    print(factors.T)

    if args.save:
        prices.to_csv("prices.csv")
        factors.to_csv("factors.csv")
        print("\nSaved prices.csv and factors.csv")


if __name__ == "__main__":
    main()