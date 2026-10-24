"""
24-factor library (equal weight = 1/24 = 4.17% each).

INPUT: long monthly panel `df`, one row per (date, ticker), POINT-IN-TIME
(fundamentals lagged to filing availability, forward-filled to month end).

Market / trading
  date, ticker, sector
  ret         monthly total return (decimal)
  prc         month-end price
  shs_out     shares outstanding (split-adjusted)
  vol_sh      monthly share volume
  mkt_ret     market return for that month (same for all tickers on a date)

Fundamentals (TTM = trailing twelve months; balance-sheet items = latest)
  at, equity (book equity), debt (ST + LT), pref, minority, cash
  rev_ttm, gp_ttm, ebit_ttm, da_ttm, ni_ttm
  cfo_ttm, capex_ttm (positive number), d_nwc_ttm (increase in net working capital)
  div_ttm, buyback_ttm, issuance_ttm (all positive cash amounts)
  tax_rate (optional; falls back to `default_tax`)
"""
import numpy as np
import pandas as pd

N_FACTORS = 24
WEIGHT = 1.0 / N_FACTORS            # 4.1667% each

# Expected sign of each factor's relationship with next-month return (economic prior).
# Used ONLY for the equal-weight baseline composite; ML models learn their own signs.
SIGNS = {
    "market_cap": -1, "beta_36m": -1, "mom_12m": +1, "mom_6m": +1, "ret_1m": -1,
    "volatility": -1, "book_to_market": +1, "earnings_yield": +1, "roe": +1,
    "gross_margin": +1, "debt_to_assets": -1, "revenue_growth": +1, "asset_growth": -1,
    "turnover": -1, "ebit_ev": +1, "ebitda_ev": +1, "sales_ev": +1, "gross_profit_ev": +1,
    "ufcf_ev": +1, "sales_to_price": +1, "cash_flow_yield": +1, "fcf_yield": +1,
    "dividend_yield": +1, "net_payout_yield": +1,
}


# ----------------------------------------------------------------- helpers
def _g(df, s):
    return s.groupby(df["ticker"])


def _roll(df, s, w, fn="mean", minp=None):
    minp = minp or max(2, int(w * 0.75))
    return _g(df, s).transform(lambda x: getattr(x.rolling(w, min_periods=minp), fn)())


def _lag(df, s, k):
    return _g(df, s).shift(k)


def _div(a, b):
    """Safe division: non-positive or missing denominators -> NaN."""
    return a / b.where(b > 0)


def _prep(df):
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------- size / risk
def market_cap(df):
    """Log market capitalization (price x shares)."""
    return np.log((df["prc"] * df["shs_out"]).where(lambda x: x > 0))


def beta_36m(df, window=36):
    """Rolling market beta = cov(r, r_m) / var(r_m) over 36 months."""
    r, m = df["ret"], df["mkt_ret"]
    e_rm, e_r = _roll(df, r * m, window), _roll(df, r, window)
    e_m, e_m2 = _roll(df, m, window), _roll(df, m ** 2, window)
    var_m = (e_m2 - e_m ** 2).where(lambda x: x > 0)
    return (e_rm - e_r * e_m) / var_m


def volatility(df, window=12, annualize=True):
    """Std. dev. of monthly returns over `window` months."""
    v = _roll(df, df["ret"], window, "std", minp=int(window * 0.75))
    return v * np.sqrt(12) if annualize else v


# ----------------------------------------------------------------- price momentum
def _momentum(df, window, skip=1):
    """Cumulative return from t-window to t-skip (skip the most recent month to avoid reversal)."""
    index = _g(df, 1 + df["ret"].fillna(0)).cumprod()      # total-return index
    return _lag(df, index, skip) / _lag(df, index, window) - 1


def mom_12m(df, skip=1):
    return _momentum(df, 12, skip)


def mom_6m(df, skip=1):
    return _momentum(df, 6, skip)


def ret_1m(df):
    return df["ret"]


# ----------------------------------------------------------------- liquidity
def turnover(df, window=3):
    """Average monthly share turnover (volume / shares outstanding)."""
    return _roll(df, df["vol_sh"] / df["shs_out"], window, "mean", minp=2)


# ----------------------------------------------------------------- value (equity-based)
def _mcap(df):
    return df["prc"] * df["shs_out"]


def book_to_market(df):
    return _div(df["equity"], _mcap(df)).where(df["equity"] > 0)


def earnings_yield(df):
    """Net income TTM / market cap (inverse P/E)."""
    return df["ni_ttm"] / _mcap(df).where(lambda x: x > 0)


def sales_to_price(df):
    return df["rev_ttm"] / _mcap(df).where(lambda x: x > 0)


def cash_flow_yield(df):
    """Operating cash flow TTM / market cap."""
    return df["cfo_ttm"] / _mcap(df).where(lambda x: x > 0)


def fcf_yield(df):
    """(CFO - capex) TTM / market cap."""
    return (df["cfo_ttm"] - df["capex_ttm"]) / _mcap(df).where(lambda x: x > 0)


def dividend_yield(df):
    """Cash dividends TTM / market cap."""
    return df["div_ttm"].fillna(0) / _mcap(df).where(lambda x: x > 0)


def net_payout_yield(df):
    """(Dividends + buybacks - equity issuance) TTM / market cap."""
    net = df["div_ttm"].fillna(0) + df["buyback_ttm"].fillna(0) - df["issuance_ttm"].fillna(0)
    return net / _mcap(df).where(lambda x: x > 0)


# ----------------------------------------------------------------- value (enterprise-value-based)
def enterprise_value(df):
    """EV = market cap + debt + preferred + minority interest - cash."""
    ev = (_mcap(df) + df["debt"].fillna(0) + df.get("pref", 0) + df.get("minority", 0)
          - df["cash"].fillna(0))
    return ev.where(ev > 0)


def ebit_ev(df):
    return df["ebit_ttm"] / enterprise_value(df)


def ebitda_ev(df):
    return (df["ebit_ttm"] + df["da_ttm"]) / enterprise_value(df)


def sales_ev(df):
    return df["rev_ttm"] / enterprise_value(df)


def gross_profit_ev(df):
    return df["gp_ttm"] / enterprise_value(df)


def ufcf_ev(df, default_tax=0.21):
    """Unlevered FCF = EBIT(1-t) + D&A - capex - change in NWC, over EV."""
    t = df["tax_rate"].fillna(default_tax) if "tax_rate" in df else default_tax
    ufcf = df["ebit_ttm"] * (1 - t) + df["da_ttm"] - df["capex_ttm"] - df["d_nwc_ttm"].fillna(0)
    return ufcf / enterprise_value(df)


# ----------------------------------------------------------------- quality / growth / leverage
def roe(df):
    """Net income TTM / average book equity (current and 12 months ago)."""
    avg_eq = (df["equity"] + _lag(df, df["equity"], 12)) / 2
    return df["ni_ttm"] / avg_eq.where(avg_eq > 0)


def gross_margin(df):
    return df["gp_ttm"] / df["rev_ttm"].where(df["rev_ttm"] > 0)


def debt_to_assets(df):
    return df["debt"] / df["at"].where(df["at"] > 0)


def revenue_growth(df):
    """Year-over-year growth in TTM revenue."""
    prev = _lag(df, df["rev_ttm"], 12)
    return df["rev_ttm"] / prev.where(prev > 0) - 1


def asset_growth(df):
    prev = _lag(df, df["at"], 12)
    return df["at"] / prev.where(prev > 0) - 1


# ----------------------------------------------------------------- assembly
FACTORS = {
    "market_cap": market_cap, "beta_36m": beta_36m, "mom_12m": mom_12m, "mom_6m": mom_6m,
    "ret_1m": ret_1m, "volatility": volatility, "book_to_market": book_to_market,
    "earnings_yield": earnings_yield, "roe": roe, "gross_margin": gross_margin,
    "debt_to_assets": debt_to_assets, "revenue_growth": revenue_growth,
    "asset_growth": asset_growth, "turnover": turnover, "ebit_ev": ebit_ev,
    "ebitda_ev": ebitda_ev, "sales_ev": sales_ev, "gross_profit_ev": gross_profit_ev,
    "ufcf_ev": ufcf_ev, "sales_to_price": sales_to_price, "cash_flow_yield": cash_flow_yield,
    "fcf_yield": fcf_yield, "dividend_yield": dividend_yield, "net_payout_yield": net_payout_yield,
}
assert len(FACTORS) == N_FACTORS


def build_factors(df):
    """Raw factor values, one column per factor."""
    df = _prep(df)
    out = df[["date", "ticker"] + (["sector"] if "sector" in df else [])].copy()
    for name, fn in FACTORS.items():
        out[name] = fn(df)
    return out.replace([np.inf, -np.inf], np.nan)


def standardize(f, sector_neutral=True, clip=(0.01, 0.99)):
    """Per-date winsorize, optional sector-demean, z-score. NaN stays NaN."""
    out = f.copy()

    def _wz(s):
        lo, hi = s.quantile(list(clip))
        s = s.clip(lo, hi)
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd > 0 else s * 0

    for c in FACTORS:
        out[c] = out.groupby("date")[c].transform(_wz)
        if sector_neutral and "sector" in out:
            out[c] -= out.groupby(["date", "sector"])[c].transform("mean")
            out[c] = out.groupby("date")[c].transform(_wz)
    return out


def composite_score(f_std, weights=None):
    """Equal-weight (4.17% each) sum of sign-adjusted z-scores. Missing factors count as 0."""
    w = weights or {k: WEIGHT for k in FACTORS}
    score = sum(w[k] * SIGNS[k] * f_std[k].fillna(0) for k in FACTORS)
    return score.rename("composite")


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    dates = pd.date_range("2016-01-31", periods=72, freq="ME")
    tick = [f"T{i}" for i in range(80)]
    d = pd.MultiIndex.from_product([dates, tick], names=["date", "ticker"]).to_frame(index=False)
    n = len(d)
    d["sector"] = rng.choice(list("ABCD"), n)
    d["mkt_ret"] = d["date"].map(pd.Series(rng.normal(0.008, 0.04, len(dates)), index=dates))
    d["ret"] = d["mkt_ret"] + rng.normal(0, 0.07, n)
    d["prc"] = rng.uniform(10, 100, n); d["shs_out"] = rng.uniform(1e7, 5e7, n)
    d["vol_sh"] = rng.uniform(1e5, 1e6, n)
    for c in ["at", "equity", "debt", "cash", "rev_ttm", "gp_ttm", "ebit_ttm", "da_ttm", "ni_ttm",
              "cfo_ttm", "capex_ttm", "d_nwc_ttm", "div_ttm", "buyback_ttm", "issuance_ttm"]:
        d[c] = rng.uniform(1e7, 1e9, n)
    f = build_factors(d)
    z = standardize(f)
    s = composite_score(z)
    print(f[list(FACTORS)].notna().mean().round(2).to_string())
    print(len(FACTORS), "factors; weight each = %.4f%%" % (100 * WEIGHT))
    print(s.describe().round(3))
