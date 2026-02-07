# strategies/dual_momentum.py

from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd


def _choose_cash_ticker(prices: pd.DataFrame) -> str:
    """
    현금 대용 티커 선택:
    - SGOV 있으면 SGOV
    - 없으면 SHY
    - 둘 다 없으면 'CASH' (백테스트에서는 수익률 0으로 처리)
    """
    if "SGOV" in prices.columns:
        return "SGOV"
    if "SHY" in prices.columns:
        return "SHY"
    return "CASH"


def _defensive_allocation(prices: pd.DataFrame) -> Dict[str, float]:
    """
    방어 모드일 때 안전자산 배분 규칙 (이미지 내용 반영):

    1) 안전자산 universe:
       SHY, IEF, TLT, TIP, LQD, HYG, BWX, EMB (존재하는 것만 사용)
    2) 최근 6개월(≈126 거래일) 수익률 기준 상위 3개 ETF 선택
    3) 각 1/3 비중
    4) 단, 선택된 ETF의 6개월 수익률이 0 이하이면 그 1/3은 투자 안 하고 현금 전환
       → 현금 비중은 0, 1/3, 2/3, 1.0

    반환값: {티커: 비중}
    """

    safe_universe_all = ["SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "BWX", "EMB"]
    safe_universe = [t for t in safe_universe_all if t in prices.columns]

    cash_ticker = _choose_cash_ticker(prices)

    if len(safe_universe) == 0 or len(prices) < 30:
        return {cash_ticker: 1.0}

    lookback = 126  # ~ 6개월
    if len(prices) <= lookback:
        return {cash_ticker: 1.0}

    rets_6m = prices[safe_universe].pct_change(lookback, fill_method=None).iloc[-1]
    rets_6m = rets_6m.dropna()
    if rets_6m.empty:
        return {cash_ticker: 1.0}

    top = rets_6m.sort_values(ascending=False).head(3)
    selected = list(top.index)
    n_sel = len(selected)
    if n_sel == 0:
        return {cash_ticker: 1.0}

    base_w = 1.0 / n_sel
    weights: Dict[str, float] = {}
    cash_w = 0.0

    for t in selected:
        r = top[t]
        if r > 0:
            weights[t] = base_w
        else:
            cash_w += base_w

    if cash_w > 0:
        weights[cash_ticker] = weights.get(cash_ticker, 0.0) + cash_w

    # 혹시 전부 0 이하라서 다 현금으로 간 경우
    if all(w <= 0 for w in weights.values()):
        return {cash_ticker: 1.0}

    return weights


def dual_momentum_signal(prices: pd.DataFrame) -> Dict[str, float]:
    """
    듀얼 모멘텀 (변경판):

    - 공격 모드(absolute momentum OK):
        * 자산군: SPY vs EFA
        * 12개월 수익률이 더 큰 쪽에 100% 투자 → {"SPY":1.0} 또는 {"EFA":1.0}
        * 단, SPY/EFA 모두 12개월 수익률 < 0 이면 방어 모드로 진입

    - 방어 모드(absolute momentum FAIL):
        * _defensive_allocation() 규칙 적용
          (최근 6개월 수익률 기준, 안전자산 3개 + 현금)

    반환: {티커: 비중}
    """

    required_risk = [t for t in ["SPY", "EFA"] if t in prices.columns]
    if len(required_risk) < 2:
        raise ValueError("DualMomentum: SPY와 EFA 데이터가 모두 필요합니다.")

    lookback_12m = 252
    if len(prices) <= lookback_12m:
        # 데이터 부족하면 방어 모드
        return _defensive_allocation(prices)

    mom_12m = prices[required_risk].pct_change(lookback_12m, fill_method=None).iloc[-1]

    # 절대 모멘텀 체크
    if (mom_12m < 0).all():
        return _defensive_allocation(prices)

    # 상대 모멘텀: 둘 중 더 좋은 쪽 100%
    best = mom_12m.idxmax()
    return {best: 1.0}
