# strategies/haa.py
"""
Hybrid Asset Allocation (HAA) Strategy by Wouter J. Keller.

- Rebalancing: Monthly
- Canary Asset: TIP (US Treasury Inflation-Protected Securities)
- Offensive Universe: SPY, IWM, VEA, VWO, VNQ, DBC, IEF, TLT
- Defensive Universe: BIL, IEF

Rules:
1.  Calculate momentum for all assets. Momentum is the average of 1, 3, 6, and 12-month returns.
2.  On the last trading day of the month, check the momentum of the canary asset (TIP).
3.  If momentum(TIP) <= 0 (Risk-Off):
    - Allocate 100% to the best asset in the defensive universe (higher momentum between BIL and IEF).
4.  If momentum(TIP) > 0 (Risk-On):
    - Select the top 4 assets from the offensive universe with the highest momentum.
    - For each of the top 4 assets:
        - If its own momentum is > 0, allocate 25% to it.
        - If its own momentum is <= 0, allocate its 25% portion to the best defensive asset instead.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd

from utils.macro_data import load_unemployment_rate
from strategies.laa import _is_recession, _is_market_uptrend

# --- Helper Functions (similar to dm_rp.py) ---

def _calc_momentum(prices: pd.Series, lookbacks: List[int]) -> float:
    """
    Calculate average momentum over multiple lookback periods.
    """
    prices = prices.dropna().astype(float)
    if not lookbacks or len(prices) < min(lookbacks) + 2:
        return np.nan

    rets = []
    for lb in lookbacks:
        if len(prices) > lb:
            r = prices.iloc[-1] / prices.iloc[-1 - lb] - 1.0
            rets.append(r)

    if not rets:
        return np.nan

    return float(np.nanmean(rets))

# --- HAA Core Logic ---

def haa_signal(prices: pd.DataFrame, unrate: pd.Series | None = None, verbose: bool = False) -> Dict[str, float]:
    """
    Calculate HAA portfolio weights for a single point in time.
    """
    OFFENSIVE_UNIVERSE = ["SPY", "IWM", "VEA", "VWO", "VNQ", "DBC", "IEF", "TLT"]
    DEFENSIVE_UNIVERSE = ["BIL", "IEF"]
    CANARY_ASSET = "TIP"
    TOP_N = 4

    # Filter universes to only available tickers
    offensive_candidates = [t for t in OFFENSIVE_UNIVERSE if t in prices.columns]
    defensive_candidates = [t for t in DEFENSIVE_UNIVERSE if t in prices.columns]
    
    cash_ticker = "SGOV" if "SGOV" in prices.columns else "BIL"

    lookbacks_mom = [21, 63, 126, 252] # 1, 3, 6, 12 months in trading days

    # --- Calculate momentum for all assets ---
    all_assets = list(set(offensive_candidates + defensive_candidates + [CANARY_ASSET]))
    mom_scores = {}
    for ticker in all_assets:
        if ticker in prices.columns:
            mom_scores[ticker] = _calc_momentum(prices[ticker], lookbacks_mom)

    mom_scores = {k: v for k, v in mom_scores.items() if pd.notna(v)}
    
    if not mom_scores:
        if verbose:
            print("HAA Signal: Not enough data to calculate any momentum scores.")
        return {cash_ticker: 1.0}

    # --- Canary Signal ---
    canary_mom = np.nan
    # TIP 데이터가 있고, 모멘텀 계산에 충분한지 확인 (최소 12개월 + 여유분)
    if CANARY_ASSET in prices.columns and len(prices[CANARY_ASSET].dropna()) > 252 + 5:
        canary_mom = mom_scores.get(CANARY_ASSET)

    if pd.notna(canary_mom):
        # 1. TIP 데이터가 충분하면, 원래 HAA 로직대로 TIP 모멘텀 사용
        is_risk_on = canary_mom > 0
        if verbose:
            print(f"HAA Signal: Using TIP momentum ({canary_mom:.4f}) -> {'Risk-On' if is_risk_on else 'Risk-Off'}")
    else:
        # 2. TIP 데이터가 없거나 부족하면, LAA의 경기/추세 판단 로직(GT)으로 대체
        if verbose:
            print("HAA Signal: TIP data not available or insufficient. Using GT fallback signal.")
        try:
            # GT 로직에 SPY가 필수
            if "SPY" not in prices.columns or prices["SPY"].dropna().empty:
                 raise ValueError("SPY price data is required for GT fallback.")

            if unrate is None:
                unrate = load_unemployment_rate()
            
            # 현재 시점까지의 데이터만 사용
            unrate_sub = unrate.loc[:prices.index[-1]]
            
            recession = _is_recession(unrate_sub)
            uptrend = _is_market_uptrend(prices["SPY"])
            
            # 리스크온 조건: (불경기 + 하락장)이 아닐 때
            is_risk_on = not (recession and not uptrend)
            if verbose:
                print(f"  GT Fallback: Recession={recession}, Uptrend={uptrend} -> {'Risk-On' if is_risk_on else 'Risk-Off'}")

        except (ValueError, IndexError) as e:
            # GT 시그널 계산 실패 시 (데이터 부족 등) 가장 안전한 Risk-Off로 결정
            if verbose:
                print(f"  GT Fallback failed: {e}. Defaulting to Risk-Off.")
            is_risk_on = False

    weights = {}

    if is_risk_on:
        # --- Risk-On Logic ---
        mom_scores_off = {t: mom_scores.get(t, -np.inf) for t in offensive_candidates}
        
        top_assets = sorted(mom_scores_off.items(), key=lambda item: item[1], reverse=True)[:TOP_N]
        
        # 2. 상승장: 모멘텀 상위 4개 자산에 각 25%씩 투자 (개별 모멘텀 체크 없음)
        for ticker, score in top_assets:
            weights[ticker] = weights.get(ticker, 0.0) + 1.0 / TOP_N

    else:
        # --- Risk-Off Logic ---
        # 3. 하락장: IEF 모멘텀이 0보다 크면 IEF, 아니면 현금으로 대피
        mom_ief = mom_scores.get("IEF", -np.inf)
        
        # '현금보다 낮으면'은 현금(SGOV)의 모멘텀이 0에 가깝기 때문에, 0보다 큰지로 판단
        if mom_ief > 0:
            defensive_asset = "IEF"
        else:
            defensive_asset = cash_ticker
        
        weights[defensive_asset] = 1.0
        if verbose:
            print(f"  Risk-Off: IEF momentum is {mom_ief:.4f}. Investing in '{defensive_asset}'.")

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        for ticker in weights:
            weights[ticker] /= total_weight

    return weights

# --- Backtest Interface ---

def _haa_weights_timeseries(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Generate monthly rebalanced weight timeseries for HAA strategy.
    """
    prices = prices.sort_index()
    
    all_possible_tickers = list(set(["SPY", "IWM", "VEA", "VWO", "VNQ", "DBC", "IEF", "TLT", "BIL", "TIP", "SGOV"]))
    cols = [t for t in all_possible_tickers if t in prices.columns]

    # 실업률 데이터 한 번만 로드
    unrate_full = load_unemployment_rate().dropna()

    monthly_idx = prices.resample("ME").last().index

    rows = []
    idxs = []

    for dt in monthly_idx:
        price_sub = prices.loc[:dt]
        unrate_sub = unrate_full[unrate_full.index <= dt]
        
        if len(price_sub) < 252 + 5:
            continue

        w_dict = haa_signal(price_sub, unrate=unrate_sub, verbose=False)
        row = {c: 0.0 for c in cols}
        row.update({t: w for t, w in w_dict.items() if t in row})
        rows.append(row)
        idxs.append(dt)

    if not rows:
        return pd.DataFrame(columns=cols)

    wdf = pd.DataFrame(rows, index=pd.to_datetime(idxs))
    return wdf.reindex(columns=cols).fillna(0.0)

def get_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Standard interface for runBacktest.py.
    """
    return _haa_weights_timeseries(prices)
