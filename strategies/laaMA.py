# strategies/laaMA.py

from typing import Dict
import pandas as pd

from strategies.customMA import _ma_alignment_weights
from utils.data_loader import load_close_for_ma


def laa_ma_signal(
    prices: pd.DataFrame,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    LAA_MA (QQQ MA 기반 리스크 온/오프) 단일 시점 시그널.

    - QQQ 에 대해 5/10/20/60일 MA 정배열/역배열 기반 _ma_alignment_weights 로
      0~1 weight 시리즈를 만든다.
    - 마지막 날짜 기준 weight == 1.0  → 리스크온
    - weight == 0.0                   → 리스크오프

    리스크온  → QQQ / IEF / IWD / IAU  각 25%
    리스크오프 → SGOV / IEF / IWD / IAU 각 25%
    """

    required = ["QQQ", "IEF", "IWD", "IAU", "SGOV"]
    missing = [t for t in required if t not in prices.columns]
    if missing:
        raise ValueError(f"LAA_MA 전략에 필요한 ETF 데이터가 없습니다: {missing}")

    prices = prices.sort_index()

    qqq = prices["QQQ"].dropna().astype(float)
    if qqq.empty:
        raise ValueError("QQQ 가격 데이터가 비어 있습니다.")

    # QQQ에 대한 MA 기반 0/1 weight (일별)
    w = _ma_alignment_weights(qqq)
    last_date = w.index[-1]
    last_w = float(w.iloc[-1])

    risk_on = last_w >= 0.5  # 여기서는 0 또는 1 이지만, 방어적으로 0.5 기준

    if verbose:
        print("=== LAA_MA (QQQ MA 기반) Decision ===")
        print(f"Last MA date    : {last_date.date()}")
        print(f"QQQ MA weight   : {last_w:.2f}")
        print(f"Risk ON?        : {risk_on}")
        print("---------------------------------------")

    if risk_on:
        return {
            "QQQ": 0.25,
            "IEF": 0.25,
            "IWD": 0.25,
            "IAU": 0.25,
        }
    else:
        return {
            "SGOV": 0.25,
            "IEF": 0.25,
            "IWD": 0.25,
            "IAU": 0.25,
        }


# ==================== 백테스트용: 일별 weight 시계열 ====================

def _laa_ma_weights_timeseries(prices: pd.DataFrame) -> pd.DataFrame:
    """
    LAA_MA 전략의 '일별' 리밸런싱 weight 시계열 생성.

    - QQQ 에 대해 _ma_alignment_weights 로 0/1 weight 시리즈를 만든다.
    - 해당 일자 weight == 1.0 → 리스크온:
        QQQ / IEF / IWD / IAU  각 25%
      weight == 0.0 → 리스크오프:
        SGOV / IEF / IWD / IAU 각 25%

    → 즉, QQQ MA 상태가 바뀌는 날마다 바로 다음 날부터 포트 구성이 바뀌게 됨
      (runBacktest 에서 shift_weight=True 이므로, 실제 적용은 다음 거래일부터)
    """
    prices = prices.sort_index()
    cols = list(prices.columns)

    required = ["QQQ", "IEF", "IWD", "IAU", "SGOV"]
    missing = [t for t in required if t not in cols]
    if missing:
        raise ValueError(f"LAA_MA 전략에 필요한 ETF 데이터가 없습니다: {missing}")

    # 1) QQQ Close 시리즈 로드 (MA 전용)
    start_str = prices.index[0].strftime("%Y-%m-%d")
    qqq_close = load_close_for_ma("QQQ", start=start_str)

    # 2) 백테스트용 Adj Close 인덱스에 맞춰 정렬
    qqq = qqq_close.reindex(prices.index).ffill()
    w_stock = _ma_alignment_weights(qqq)  # index = qqq.index, values = 0 or 1

    # 전체 price_df 인덱스로 맞추고, NA 는 0으로 (초기 구간 등)
    w_stock = w_stock.reindex(prices.index).ffill().fillna(0.0)

    # 2) 결과 weight DataFrame 초기화 (모든 자산 0%)
    weight_df = pd.DataFrame(0.0, index=prices.index, columns=cols)

    # 3) 리스크온/오프 마스크
    risk_on_mask = w_stock >= 0.5   # 1.0이면 True, 0.0이면 False

    # 리스크온 구간: QQQ/IEF/IWD/IAU = 25%씩
    on_cols = ["QQQ", "IEF", "IWD", "IAU"]
    # 리스크오프 구간: SGOV/IEF/IWD/IAU = 25%씩
    off_cols = ["SGOV", "IEF", "IWD", "IAU"]

    # 존재하는 컬럼만 사용 (혹시 일부 누락된 경우 방어)
    on_cols = [c for c in on_cols if c in weight_df.columns]
    off_cols = [c for c in off_cols if c in weight_df.columns]

    # 리스크온 날들
    if on_cols:
        weight_df.loc[risk_on_mask, on_cols] = 0.25

    # 리스크오프 날들
    if off_cols:
        weight_df.loc[~risk_on_mask, off_cols] = 0.25

    return weight_df.reindex(columns=cols)


def get_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """
    runBacktest.py 등에서 사용하는 표준 인터페이스.
    LAA_MA 전략의 일별 weight 시계열을 반환.
    """
    return _laa_ma_weights_timeseries(prices)
