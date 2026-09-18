"""Approximate constant-maturity Treasury total returns, not inverse yields."""
import numpy as np
import pandas as pd


def treasury_total_return(yield_percent: pd.Series, maturity_years: float) -> pd.Series:
    """Daily rolled par bond, semiannual coupons, flat yield curve.

    Carry uses elapsed calendar time; repricing discounts the remaining cash
    flows at today's yield. This is a model, not a historical ETF NAV series.
    Missing yields remain missing before the first observation. No backfill.
    """
    y = yield_percent.sort_index().astype(float).ffill().dropna() / 100
    if y.empty:
        return pd.Series(index=yield_percent.index, dtype=float)
    if not np.isfinite(y).all() or (y <= -2).any() or maturity_years < .5:
        raise ValueError('Invalid Treasury yield or maturity.')
    times = np.arange(.5, maturity_years + 1e-9, .5)
    values = np.ones(len(y))
    for i in range(1, len(y)):
        elapsed = (y.index[i] - y.index[i-1]).days / 365.25
        if elapsed >= .5:
            raise ValueError('Treasury yield gap exceeds a coupon period.')
        flows = np.full(len(times), y.iloc[i-1] / 2)
        flows[-1] += 1
        gross = np.sum(flows / (1 + y.iloc[i]/2) ** (2 * (times-elapsed)))
        values[i] = values[i-1] * gross
    return pd.Series(values, index=y.index).reindex(yield_percent.index)
