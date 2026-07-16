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
from io import StringIO
import urllib.request

# --- Helper Functions (similar to dm_rp.py) ---

def _calc_momentum(prices: pd.Series, lookbacks: List[int]) -> float:
    """
    Calculate average momentum over multiple lookback periods.
    """
    prices = prices.dropna().astype(float)
    if not lookbacks or len(prices) < min(lookbacks) + 2: # type: ignore
        return np.nan

    rets = []
    for lb in lookbacks:
        if len(prices) > lb:
            r = prices.iloc[-1] / prices.iloc[-1 - lb] - 1.0
            rets.append(r)

    if not rets:
        return np.nan

    return float(np.nanmean(rets))

_TIPS_YIELD_CACHE = None

def _load_tips_yield_data(start="1995-01-01") -> pd.Series:
    """
    FRED에서 10년 만기 물가연동국채(TIPS) 금리(DFII10)를 다운로드하고 캐시합니다.
    yfinance가 FRED 티커를 안정적으로 지원하지 않으므로, FRED 웹사이트에서 직접 CSV를 읽어옵니다.
    네트워크 불안정/서버 차단에 대응하기 위해 User-Agent를 포함한 urllib로 다운로드를 시도합니다.
    """
    global _TIPS_YIELD_CACHE
    if _TIPS_YIELD_CACHE is not None:
        # 캐시된 데이터가 충분히 이른 시점부터 시작하는지 확인
        if not _TIPS_YIELD_CACHE.empty and _TIPS_YIELD_CACHE.index[0] <= pd.to_datetime(start):
            return _TIPS_YIELD_CACHE

    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10"

        # 브라우저처럼 보이도록 User-Agent 헤더 추가
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            csv_data = response.read().decode('utf-8')

        # 다운로드한 데이터를 문자열 스트림으로 변환하여 pandas가 읽도록 함
        df = pd.read_csv(StringIO(csv_data), parse_dates=["DATE"])

        df = df.rename(columns={"DATE": "date", "DFII10": "yield"})
        df = df.set_index("date").sort_index()

        # FRED는 '.'으로 누락 값을 표시할 수 있으므로, 숫자로 변환합니다.
        df['yield'] = pd.to_numeric(df['yield'], errors='coerce')

        tips_yield = df['yield'].dropna()
        _TIPS_YIELD_CACHE = tips_yield
        return tips_yield
    except Exception as e:
        print(f"Warning: Could not download TIPS yield data (DFII10) from FRED: {e}")
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))

def _calc_yield_momentum(yields: pd.Series, lookbacks: List[int]) -> float:
    """
    금리(yield) 시계열로부터 가격 모멘텀을 계산합니다.
    채권 가격은 금리에 반비례하므로, 수익률 공식은 (Y_old / Y_new) - 1 입니다.
    """
    yields = yields.dropna().astype(float)
    if not lookbacks or len(yields) < min(lookbacks) + 2: # type: ignore
        return np.nan

    rets = []
    for lb in lookbacks:
        if len(yields) > lb:
            current_yield = yields.iloc[-1]
            past_yield = yields.iloc[-1 - lb]
            # 금리가 0이 아닐 때만 계산
            if pd.notna(current_yield) and pd.notna(past_yield) and current_yield != 0:
                r = past_yield / current_yield - 1.0
                rets.append(r)

    if not rets:
        return np.nan

    return float(np.nanmean(rets))


# --- HAA Core Logic ---

def haa_signal(
    prices: pd.DataFrame,
    tips_yield: pd.Series | None = None,
    verbose: bool = False
) -> Dict[str, float]:
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

    # --- Best defensive asset for both modes ---
    defensive_scores = {t: mom_scores.get(t, -np.inf) for t in defensive_candidates if t in mom_scores}
    if defensive_scores:
        best_defensive_asset = max(defensive_scores, key=defensive_scores.get)
        if verbose:
            print(f"  Defensive Asset Check: Scores={defensive_scores}, Best='{best_defensive_asset}'")
    else:
        best_defensive_asset = cash_ticker # Fallback

    # --- Canary Signal ---
    canary_mom = mom_scores.get(CANARY_ASSET, np.nan)
    proxy_used = False

    # 1. TIP ETF 모멘텀이 계산되지 않은 경우 (NaN)
    if pd.isna(canary_mom):
        if verbose:
            print(f"HAA Signal: Canary asset '{CANARY_ASSET}' momentum not available. Trying TIPS yield proxy (DFII10).")
        
        # 2. 프록시로 TIPS 금리 데이터 사용
        if tips_yield is None:
            # 백테스트가 아닌 단일 시점 조회용 경로
            tips_yield = _load_tips_yield_data(start=prices.index[0].strftime("%Y-%m-%d"))
        
        # 현재 시점까지의 데이터만 사용
        tips_yield_sub = tips_yield.loc[:prices.index[-1]]
        
        if not tips_yield_sub.empty:
            proxy_mom = _calc_yield_momentum(tips_yield_sub, lookbacks=lookbacks_mom)
            if pd.notna(proxy_mom):
                canary_mom = proxy_mom # 프록시 모멘텀으로 canary_mom 업데이트
                proxy_used = True
                if verbose:
                    print(f"  Using TIPS yield proxy momentum: {canary_mom:.4f}")

    # 3. 최종적으로 canary_mom 값으로 리스크 온/오프 결정
    if pd.isna(canary_mom):
        # TIP ETF도, TIPS 금리 프록시도 실패한 경우 -> 가장 안전한 Risk-Off
        is_risk_on = False
        if verbose:
            print(f"HAA Signal: All canary signals failed. Defaulting to Risk-Off.")
    else:
        # canary_mom 값이 있으면 (ETF든 프록시든) 그걸로 결정
        is_risk_on = canary_mom > 0
        if verbose:
            proxy_msg = " (from yield proxy)" if proxy_used else ""
            print(f"HAA Signal: Final Canary Momentum{proxy_msg} is {canary_mom:.4f} -> {'Risk-On' if is_risk_on else 'Risk-Off'}")

    weights = {}

    if is_risk_on:
        # --- Risk-On Logic ---
        mom_scores_off = {t: mom_scores.get(t, -np.inf) for t in offensive_candidates if t in mom_scores}
        
        top_assets = sorted(mom_scores_off.items(), key=lambda item: item[1], reverse=True)[:TOP_N]
        
        # For each of the top 4, invest if momentum is positive, otherwise switch to best defensive.
        for ticker, score in top_assets:
            if score > 0:
                weights[ticker] = weights.get(ticker, 0.0) + 1.0 / TOP_N
            else:
                weights[best_defensive_asset] = weights.get(best_defensive_asset, 0.0) + 1.0 / TOP_N

    else:
        # --- Risk-Off Logic ---
        # Invest 100% in the best defensive asset.
        weights[best_defensive_asset] = 1.0
        if verbose:
            print(f"  Risk-Off: Investing in best defensive asset '{best_defensive_asset}'.")

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        for ticker in weights:
            weights[ticker] /= total_weight

    return weights

# --- Backtest Interface ---

def _haa_weights_timeseries(prices: pd.DataFrame) -> pd.DataFrame:
    """
    HAA 전략의 월별 리밸런싱 weight 시계열을 생성합니다.
    성능 저하(무한 루프처럼 보이는 현상)를 방지하기 위해 모든 모멘텀 점수를 미리 계산합니다.
    """
    prices = prices.sort_index()

    # Define constants from haa_signal
    OFFENSIVE_UNIVERSE = ["SPY", "IWM", "VEA", "VWO", "VNQ", "DBC", "IEF", "TLT"]
    DEFENSIVE_UNIVERSE = ["BIL", "IEF"]
    CANARY_ASSET = "TIP"
    TOP_N = 4
    lookbacks_mom = [21, 63, 126, 252]

    all_possible_tickers = list(set(["SPY", "IWM", "VEA", "VWO", "VNQ", "DBC", "IEF", "TLT", "BIL", "TIP", "SGOV"]))
    cols = [t for t in all_possible_tickers if t in prices.columns]

    # --- 1. 모든 자산의 모멘텀 점수를 미리 계산 ---
    mom_scores_df = pd.DataFrame(index=prices.index)
    for ticker in cols:
        if ticker == "SGOV": continue
        
        rets = {}
        for lb in lookbacks_mom:
            price_lb = prices[ticker].shift(lb).replace(0, np.nan)
            rets[f'ret_{lb}'] = prices[ticker] / price_lb - 1.0
        
        mom_scores_df[ticker] = pd.DataFrame(rets).mean(axis=1)

    # --- 2. TIPS 금리 기반 프록시 모멘텀 미리 계산 ---
    start_date_str = prices.index[0].strftime("%Y-%m-%d")
    tips_yield_full = _load_tips_yield_data(start=start_date_str)
    tips_yield_mom_df = pd.DataFrame(index=prices.index)
    if not tips_yield_full.empty:
        rets = {}
        for lb in lookbacks_mom:
            current_yield = tips_yield_full.replace(0, np.nan)
            past_yield = tips_yield_full.shift(lb)
            rets[f'ret_{lb}'] = past_yield / current_yield - 1.0
        tips_yield_mom_df['DFII10'] = pd.DataFrame(rets).mean(axis=1)
    else:
        tips_yield_mom_df['DFII10'] = np.nan

    # --- 3. 월말에 루프를 돌며 가중치 생성 ---
    monthly_idx = prices.resample("ME").last().index.intersection(mom_scores_df.index)

    rows = []
    idxs = []

    for dt in monthly_idx:
        mom_scores = mom_scores_df.loc[dt].to_dict()
        if pd.isna(list(mom_scores.values())).all():
            continue

        # --- haa_signal 로직 시작 ---
        offensive_candidates = [t for t in OFFENSIVE_UNIVERSE if t in cols]
        defensive_candidates = [t for t in DEFENSIVE_UNIVERSE if t in cols]
        cash_ticker = "SGOV" if "SGOV" in cols else "BIL"

        defensive_scores = {t: mom_scores.get(t, -np.inf) for t in defensive_candidates if pd.notna(mom_scores.get(t))}
        best_defensive_asset = max(defensive_scores, key=defensive_scores.get) if defensive_scores else cash_ticker

        canary_mom = mom_scores.get(CANARY_ASSET, np.nan)
        if pd.isna(canary_mom):
            canary_mom = tips_yield_mom_df.loc[dt, 'DFII10']

        is_risk_on = canary_mom > 0 if pd.notna(canary_mom) else False

        weights = {}
        if is_risk_on:
            mom_scores_off = {t: mom_scores.get(t, -np.inf) for t in offensive_candidates if pd.notna(mom_scores.get(t))}
            top_assets = sorted(mom_scores_off.items(), key=lambda item: item[1], reverse=True)[:TOP_N]
            for ticker, score in top_assets:
                weights[ticker if score > 0 else best_defensive_asset] = weights.get(ticker if score > 0 else best_defensive_asset, 0.0) + 1.0 / TOP_N
        else:
            weights[best_defensive_asset] = 1.0
        # --- haa_signal 로직 종료 ---

        row = {c: weights.get(c, 0.0) for c in cols}
        rows.append(row); idxs.append(dt)

    if not rows:
        return pd.DataFrame(columns=cols)

    wdf = pd.DataFrame(rows, index=pd.to_datetime(idxs))
    return wdf.reindex(columns=cols).fillna(0.0)

def get_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Standard interface for runBacktest.py.
    """
    return _haa_weights_timeseries(prices)
