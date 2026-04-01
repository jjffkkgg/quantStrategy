# strategies/ma2.py
"""
MA2 전략

- 기본 구조:
    * IAU 25%, IEF 25%는 항상 보유
    * QQQ 25% : QQQ MA 정배열 전략으로 ON/OFF
    * IWD 25% : IWD MA 정배열 전략으로 ON/OFF
      → OFF 된 비중은 SGOV 로 이동

- QQQ/IWD MA 로직:
    * strategies.customMA._ma_alignment_weights 를 사용
    * 완전 정배열 → 1.0, 매도 상태 → 0.0, 그 외 직전값 유지
    * 여기서는 0.5 이상이면 "ON" 으로 간주
"""
from typing import Dict

import pandas as pd

from utils.data_loader import load_close_for_ma
from strategies.customMA import _ma_alignment_weights


# ----------------------------------------------------------------------
# 1. 백테스트용 weight 시계열
# ----------------------------------------------------------------------


def _ma2_weights_timeseries(prices: pd.DataFrame) -> pd.DataFrame:
    """
    MA2 전략의 일별 weight 시계열 생성.

    - 항상:
        IAU 25%, IEF 25%
    - QQQ 25%:
        QQQ MA ON  → QQQ 25%
        QQQ MA OFF → SGOV 25%
    - IWD 25%:
        IWD MA ON  → IWD 25%
        IWD MA OFF → SGOV 25%
    """
    prices = prices.sort_index()
    cols = list(prices.columns)

    required = ["QQQ", "IWD", "IAU", "IEF", "SGOV"]
    missing = [t for t in required if t not in cols]
    if missing:
        raise ValueError(f"MA2 전략에 필요한 ETF 데이터가 없습니다: {missing}")

    idx = prices.index
    start_str = idx[0].strftime("%Y-%m-%d")

    # ------------------------------
    # 1-1) QQQ MA 기반 온/오프 시리즈
    # ------------------------------
    qqq_close = load_close_for_ma("QQQ", start=start_str)
    qqq_close = qqq_close.reindex(idx).ffill()

    w_qqq = _ma_alignment_weights(qqq_close)
    w_qqq = w_qqq.reindex(idx).ffill().fillna(0.0)
    qqq_on_mask = w_qqq >= 0.5

    # ------------------------------
    # 1-2) IWD MA 기반 온/오프 시리즈
    # ------------------------------
    iwd_close = load_close_for_ma("IWD", start=start_str)
    iwd_close = iwd_close.reindex(idx).ffill()

    w_iwd = _ma_alignment_weights(iwd_close)
    w_iwd = w_iwd.reindex(idx).ffill().fillna(0.0)
    iwd_on_mask = w_iwd >= 0.5

    # ------------------------------
    # 1-3) weight DataFrame 구성
    # ------------------------------
    weight_df = pd.DataFrame(0.0, index=idx, columns=cols)

    # 항상 보유 50%
    weight_df["IAU"] = 0.25
    weight_df["IEF"] = 0.25

    # QQQ 슬롯 25%
    weight_df.loc[qqq_on_mask, "QQQ"] = 0.25
    weight_df.loc[~qqq_on_mask, "SGOV"] += 0.25

    # IWD 슬롯 25%
    weight_df.loc[iwd_on_mask, "IWD"] = 0.25
    weight_df.loc[~iwd_on_mask, "SGOV"] += 0.25

    weight_df = weight_df.fillna(0.0)
    return weight_df


def get_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """
    runBacktest.py 에서 사용하는 표준 인터페이스.
    """
    return _ma2_weights_timeseries(prices)


# ----------------------------------------------------------------------
# 2. 최신 시점 시그널 (runStrategy.py 용)
# ----------------------------------------------------------------------


def ma2_signal(
    prices: pd.DataFrame,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    마지막 날짜 기준 MA2 포트폴리오 비중 dict 반환.
    """
    prices = prices.sort_index()
    weight_df = _ma2_weights_timeseries(prices)

    last_date = prices.index[-1]
    last_w = weight_df.iloc[-1]

    weights = {k: float(v) for k, v in last_w.items() if abs(v) > 1e-12}

    if verbose:
        print("=== MA2 Signal (Latest) ===")
        print(f"Date : {last_date.date()}")
        for k, v in weights.items():
            print(f"  {k:5s}: {v*100:5.1f}%")
        print("---------------------------------------")

    return weights
