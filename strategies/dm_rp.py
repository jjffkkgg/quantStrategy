# strategies/dm_rp.py

from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 유틸 함수들
# ------------------------------------------------------------

def _choose_cash_ticker(prices: pd.DataFrame) -> str:
    """
    현금 대용 티커 선택:
    - SGOV 있으면 SGOV
    - 없으면 SHY
    - 둘 다 없으면 'CASH' (백테스트에서는 수익률 0으로 처리)
    """
    cols = prices.columns
    if "SGOV" in cols:
        return "SGOV"
    if "SHY" in cols:
        return "SHY"
    return "CASH"


def _calc_momentum(
    prices: pd.Series,
    lookbacks: List[int],
) -> float:
    """
    여러 구간 (1M/3M/6M/12M 등)의 수익률을 평균해서 모멘텀 점수로 사용.

    - 최근 lookback 일 수익률 = price / price.shift(lookback) - 1
    - 유효한 구간만 평균
    """
    prices = prices.dropna().astype(float)
    if len(prices) < min(lookbacks) + 2:
        return np.nan

    rets = []
    for lb in lookbacks:
        if len(prices) > lb:
            r = prices.iloc[-1] / prices.iloc[-lb - 1] - 1.0
            rets.append(r)

    if not rets:
        return np.nan

    return float(np.nanmean(rets))


def _inverse_vol_weights(
    prices: pd.DataFrame,
    lookback: int = 60,
) -> Dict[str, float]:
    """
    주어진 가격들에 대해 역변동성(inverse-vol) 기반 Risk-Parity weight 계산.

    - 각 자산의 일일 수익률 표준편차(최근 lookback일)를 구한 뒤
      1/vol 를 비중으로 사용 (정규화해서 합=1).
    - 데이터 부족하거나 vol==0 인 자산은 제외.
    """
    prices = prices.dropna(how="all")
    if len(prices) < lookback + 2:
        # 데이터가 너무 짧으면 균등분배
        valid_cols = [c for c in prices.columns if prices[c].dropna().shape[0] > 1]
        if not valid_cols:
            return {}
        n = len(valid_cols)
        return {c: 1.0 / n for c in valid_cols}

    rets = prices.pct_change().dropna()
    rets_lb = rets.tail(lookback)

    vols = rets_lb.std()
    vols = vols.replace(0, np.nan).dropna()
    if vols.empty:
        return {}

    inv_vol = 1.0 / vols
    w = inv_vol / inv_vol.sum()

    return w.to_dict()


# ------------------------------------------------------------
# 듀얼모멘텀: 리스크온/오프 판단 (SPY vs EFA)
# ------------------------------------------------------------

def _dm_risk_regime(prices: pd.DataFrame) -> str:
    """
    듀얼모멘텀 방식으로 리스크온/오프 모드 판별.

    - 공격 모드(absolute momentum OK):
        * 자산군: SPY vs EFA
        * 12개월 수익률이 더 큰 쪽이 100%를 가져갈 수 있는 상태
        * 단, SPY/EFA 모두 12개월 수익률 < 0 이면 방어 모드로 진입

    Returns
    -------
    str: "risk_on" 또는 "risk_off"
    """
    cols = prices.columns
    if "SPY" not in cols or "EFA" not in cols:
        # SPY/EFA 둘 다 없으면 보수적으로 risk_off
        return "risk_off"

    lookback_12m = 252
    if len(prices) <= lookback_12m:
        return "risk_off"

    mom_12m = prices[["SPY", "EFA"]].pct_change(lookback_12m, fill_method=None).iloc[-1]

    # 절대 모멘텀 체크
    if (mom_12m < 0).all():
        return "risk_off"

    return "risk_on"


# ------------------------------------------------------------
# 리스크온/오프 별 포트 구성 로직
# ------------------------------------------------------------

def _build_risk_on_weights(price_sub: pd.DataFrame) -> Dict[str, float]:
    """
    리스크온 모드일 때:

    - 위험 자산 유니버스 예시:
        SPY, QQQ, IWD, IWM, EFA 중 존재하는 것
    - 각 자산에 대해 (1M, 3M, 6M, 12M) 모멘텀 점수 계산
    - 상위 3개를 선택
    - 선택된 3개에 대해 60일 기준 inverse-vol weight 적용
    """
    risk_candidates = [t for t in ["SPY", "QQQ", "IWD", "IWM", "EFA"]
                       if t in price_sub.columns]

    if not risk_candidates:
        # 위험 자산이 없으면 방어 모드와 동일하게 처리
        return _build_risk_off_weights(price_sub)

    lookbacks = [21, 63, 126, 252]   # 1/3/6/12M (영업일 기준 근사)
    mom_scores = {}

    for t in risk_candidates:
        score = _calc_momentum(price_sub[t], lookbacks)
        mom_scores[t] = score

    # 모멘텀 점수가 NaN 이 아닌 자산만 사용
    mom_scores = {k: v for k, v in mom_scores.items() if not np.isnan(v)}
    if not mom_scores:
        return _build_risk_off_weights(price_sub)

    # 상위 3개까지 선택
    top = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    selected = [t for t, _ in top]

    # 선택된 자산의 가격만 모아서 inverse-vol
    prices_sel = price_sub[selected]
    w = _inverse_vol_weights(prices_sel, lookback=60)
    if not w:
        # 혹시라도 실패하면 균등분배
        n = len(selected)
        return {t: 1.0 / n for t in selected}

    return w


def _build_risk_off_weights(price_sub: pd.DataFrame) -> Dict[str, float]:
    """
    리스크오프 모드일 때:

    - 안전 자산 유니버스:
        SHY, IEF, TLT, TIP, LQD, HYG, BWX, EMB, SGOV 중 존재하는 것
    - 최근 6개월(≈126 거래일) 수익률 기준 상위 3개 선택
    - 선택된 3개에 대해 inverse-vol weight 적용
    - 단, 선택된 자산의 6개월 수익률이 모두 <= 0 이면 전부 현금(SGOV/SHY)으로
    """
    safe_all = ["SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "BWX", "EMB", "SGOV"]
    safe_universe = [t for t in safe_all if t in price_sub.columns]

    cash_ticker = _choose_cash_ticker(price_sub)

    if len(safe_universe) == 0 or len(price_sub) < 30:
        return {cash_ticker: 1.0}

    lookback_6m = 126
    if len(price_sub) <= lookback_6m:
        return {cash_ticker: 1.0}

    rets_6m = price_sub[safe_universe].pct_change(lookback_6m, fill_method=None).iloc[-1]
    rets_6m = rets_6m.dropna()
    if rets_6m.empty:
        return {cash_ticker: 1.0}

    top = rets_6m.sort_values(ascending=False).head(3)
    selected = list(top.index)
    if len(selected) == 0:
        return {cash_ticker: 1.0}

    # 모두 수익률 <= 0 이면 전부 현금
    if (top <= 0).all():
        return {cash_ticker: 1.0}

    prices_sel = price_sub[selected]
    w = _inverse_vol_weights(prices_sel, lookback=60)
    if not w:
        # 실패 시 균등분배
        n = len(selected)
        base = 1.0 / n
        w = {t: base for t in selected}

    # 역모멘텀 자산(6M 수익률 <= 0)은 현금으로 전환
    final_w: Dict[str, float] = {}
    cash_w = 0.0
    for t, wt in w.items():
        r = float(top.get(t, 0.0))
        if r <= 0:
            cash_w += wt
        else:
            final_w[t] = wt

    if cash_w > 0:
        final_w[cash_ticker] = final_w.get(cash_ticker, 0.0) + cash_w

    if not final_w:
        return {cash_ticker: 1.0}

    # 혹시 합이 1이 아니면 정규화
    s = sum(final_w.values())
    if s <= 0:
        return {cash_ticker: 1.0}
    for k in list(final_w.keys()):
        final_w[k] /= s

    return final_w


# ------------------------------------------------------------
# 단일 시점 시그널 함수 (main.py 용)
# ------------------------------------------------------------

def dm_rp_signal(
    prices: pd.DataFrame,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Dual Momentum + Risk Parity 전략의 '현재 시점' 포트폴리오 비중.

    - 리밸런싱은 월말 기준 (백테스트에서는 월말 weight 사용)
    - 이 함수는 단순히 전체 데이터(prices)를 보고
      마지막 날짜 기준의 weights 를 계산해서 반환.
    """
    prices = prices.sort_index()
    if prices.empty:
        raise ValueError("가격 데이터가 비어 있습니다.")

    regime = _dm_risk_regime(prices)

    if regime == "risk_on":
        w = _build_risk_on_weights(prices)
    else:
        w = _build_risk_off_weights(prices)

    if verbose:
        last_date = prices.index[-1].date()
        print("=== DM_RP (Dual Momentum + Risk Parity) Signal ===")
        print(f"Last date     : {last_date}")
        print(f"Regime        : {regime}")
        print("Weights       :")
        for k, v in w.items():
            print(f"  {k:5s}: {v*100:5.1f}%")
        print("----------------------------------------")

    return w


# ------------------------------------------------------------
# 백테스트용 weight 시계열 생성 (월말 리밸런싱)
# ------------------------------------------------------------

def _dm_rp_weights_timeseries(prices: pd.DataFrame) -> pd.DataFrame:
    """
    DM_RP 전략의 월말 weight 시계열 생성.

    - 월말 인덱스 기준으로,
      해당 시점까지의 price_sub만 사용해서 dm_rp_signal 계산
      (룩어헤드 방지).
    """
    prices = prices.sort_index()
    cols = list(prices.columns)

    monthly_idx = prices.resample("ME").last().index

    rows = []
    idxs = []

    for dt in monthly_idx:
        price_sub = prices.loc[:dt]

        weights = dm_rp_signal(price_sub, verbose=False)

        row = {c: 0.0 for c in cols}
        for t, w in weights.items():
            if t in row:
                row[t] = w

        rows.append(row)
        idxs.append(dt)

    if not rows:
        raise ValueError("DM_RP weight 시계열이 비어 있습니다. 데이터 기간을 확인하세요.")

    wdf = pd.DataFrame(rows, index=pd.to_datetime(idxs))
    return wdf.reindex(columns=cols)


def get_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """
    runBacktest.py 에서 사용하는 표준 인터페이스.
    """
    return _dm_rp_weights_timeseries(prices)
