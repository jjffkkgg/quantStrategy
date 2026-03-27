# strategies/laa2.py
"""
LAA2 전략 (LAA + IWD 동시 리스크온/오프)

- 기본 구조:
    * IAU 25%, IEF 25%는 항상 보유
    * QQQ 25% + IWD 25%를 하나의 리스크 블록처럼 같이 ON/OFF

- 경기/추세 판정은 기존 LAA(GT)와 동일:
    * 실업률 > 12M 이동평균 → recession = True (불경기)
    * SPY  < 200D MA      → uptrend = False (하락장)

- 최종 규칙:
    * recession & (not uptrend) 이면 (불경기 & 하락장)
        → QQQ / IWD 50%를 전부 SGOV로 이동
        → SGOV 50%, IAU 25%, IEF 25%

    * 그 외 모든 경우
        → QQQ 25%, IWD 25%, IAU 25%, IEF 25%
"""

from typing import Dict, Optional

import pandas as pd

from utils.macro_data import load_unemployment_rate
from strategies.laa import _is_recession, _is_market_uptrend


# ----------------------------------------------------------------------
# 1. 시그널 (마지막 날짜 기준)
# ----------------------------------------------------------------------


def laa2_signal(
    prices: pd.DataFrame,
    unrate: Optional[pd.Series] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    LAA2 전략 시그널 (마지막 날짜 기준 포트 비중 dict).

    필요 티커:
        QQQ, IWD, IAU, IEF, SGOV, SPY
    """
    required = ["QQQ", "IWD", "IAU", "IEF", "SGOV", "SPY"]
    missing = [t for t in required if t not in prices.columns]
    if missing:
        raise ValueError(f"LAA2 전략에 필요한 ETF 데이터가 없습니다: {missing}")

    prices = prices.sort_index()

    # --- 실업률 데이터 준비 ---
    if unrate is None:
        unrate = load_unemployment_rate()
    unrate = unrate.dropna()
    if len(unrate) < 13:
        raise ValueError("실업률 데이터가 12개월 이동평균을 계산할 만큼 충분하지 않습니다.")

    last_unemp_date = unrate.index[-1]
    last_unemp_value = float(unrate.iloc[-1])
    sma12 = float(unrate.tail(12).mean())

    recession = _is_recession(unrate)

    # --- SPY 200일 이동평균 기반 추세 판단 ---
    spy = prices["SPY"].dropna().astype(float)
    uptrend = _is_market_uptrend(spy)

    # --- 최종 리스크 블록(QQQ+IWD) 온/오프 ---
    if recession and (not uptrend):
        # 불경기 + 하락장 → 리스크오프: QQQ/IWD 0, SGOV 50
        w_q = 0.0
        w_iwd = 0.0
        w_cash = 0.50
    else:
        # 그 외 → 리스크온: QQQ 25, IWD 25
        w_q = 0.25
        w_iwd = 0.25
        w_cash = 0.0

    weights: Dict[str, float] = {
        "QQQ": w_q,
        "IWD": w_iwd,
        "IAU": 0.25,
        "IEF": 0.25,
        "SGOV": w_cash,
    }

    if verbose:
        print("=== LAA2 Signal Info ===")
        print(f"Latest UNRATE date   : {last_unemp_date.date()}")
        print(f"Latest UNRATE value  : {last_unemp_value:.2f}%")
        print(f"12M moving average   : {sma12:.2f}%")
        print(f"Recession?           : {recession}")
        spy_last_date = spy.index[-1]
        spy_last_price = float(spy.iloc[-1])
        spy_ma200 = float(spy.rolling(200).mean().iloc[-1])
        print(f"SPY last date        : {spy_last_date.date()}")
        print(f"SPY last price       : {spy_last_price:.2f}")
        print(f"SPY 200D MA          : {spy_ma200:.2f}")
        print(f"Market uptrend?      : {uptrend}")
        print(f"Final weights        :")
        for k, v in weights.items():
            print(f"  {k:5s}: {v*100:5.1f}%")
        print("---------------------------------------")

    return weights


# ----------------------------------------------------------------------
# 2. 백테스트용 월말 weight 시계열
# ----------------------------------------------------------------------


def _laa2_weights_timeseries(prices: pd.DataFrame) -> pd.DataFrame:
    """
    LAA2 전략의 월말 리밸런싱 weight 시계열 생성.

    - 매 월말마다, 그 시점까지의 가격/실업률 정보만 사용해서
      laa2_signal()을 호출.
    - 결과는 월말 기준 weight 를 담은 DataFrame.
    """
    prices = prices.sort_index()
    cols = list(prices.columns)

    # 월말 인덱스 (마지막 영업일 기준)
    monthly_idx = prices.resample("ME").last().index

    # 실업률 전체 시계열
    unrate_full = load_unemployment_rate().dropna()

    weight_rows = []
    idxs = []

    for dt in monthly_idx:
        price_sub = prices.loc[:dt]
        unrate_sub = unrate_full[unrate_full.index <= dt]

        # 실업률/가격 데이터 부족하면 건너뜀
        if len(unrate_sub) < 13 or price_sub["SPY"].dropna().shape[0] < 200:
            continue

        try:
            w_dict = laa2_signal(price_sub, unrate=unrate_sub, verbose=False)
        except ValueError:
            # 초기 구간 등에서 조건 미충족 시 스킵
            continue

        row = {c: 0.0 for c in cols}
        for t, w in w_dict.items():
            if t in row:
                row[t] = w

        weight_rows.append(row)
        idxs.append(dt)

    if not weight_rows:
        raise ValueError("LAA2 weight 시계열이 비어 있습니다. 데이터 기간을 확인하세요.")

    weight_df = pd.DataFrame(weight_rows, index=pd.DatetimeIndex(idxs))
    weight_df = weight_df.reindex(columns=cols)

    return weight_df


def get_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """
    runBacktest.py 에서 사용하는 표준 인터페이스.
    LAA2 전략의 월별 리밸런싱 weight DataFrame을 반환.
    """
    return _laa2_weights_timeseries(prices)
