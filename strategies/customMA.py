# strategies/sp500_ma.py
import pandas as pd
import numpy as np


def _ma_alignment_weights(price: pd.Series) -> pd.Series:
    """
    단일 자산(예: S&P500) 가격 시계열에 대해
    5/10/20/60일 MA 정배열/역배열 상태 기반으로 0~1 weight 시리즈 생성.

    - 완전 정배열: ma5 > ma10 > ma20 > ma60 → weight = 1.0
    - 매도(0%):
        (1) 역배열 상태: (ma5<ma10, ma10<ma20, ma20<ma60) 중 2개 이상 이거나
        (2) 5일선이 10일선, 20일선 둘 다 아래에 있는 상태 (ma5 < ma10, ma5 < ma20)
    - 그 외: 직전 weight 유지
    """
    price = price.astype(float)

    ma5 = price.rolling(5).mean()
    ma10 = price.rolling(10).mean()
    ma20 = price.rolling(20).mean()
    ma60 = price.rolling(60).mean()

    ma_df = pd.DataFrame(
        {
            "ma5": ma5,
            "ma10": ma10,
            "ma20": ma20,
            "ma60": ma60,
        },
        index=price.index,
    )

    # 1) 완전 정배열 조건
    bullish_full = (
        (ma_df["ma5"] > ma_df["ma10"])
        & (ma_df["ma10"] > ma_df["ma20"])
        & (ma_df["ma20"] > ma_df["ma60"])
    )

    # 2) 역배열 "상태" (위에서 두 개 이상 거꾸로)
    rev_5_10 = ma_df["ma5"] < ma_df["ma10"]
    rev_10_20 = ma_df["ma10"] < ma_df["ma20"]
    rev_20_60 = ma_df["ma20"] < ma_df["ma60"]

    num_reversed = (
        rev_5_10.astype(int)
        + rev_10_20.astype(int)
        + rev_20_60.astype(int)
    )
    bearish_alignment = num_reversed >= 2

    # 3) 5일선이 10일/20일 둘 다 아래에 있는 상태
    five_below_10_20 = (ma_df["ma5"] < ma_df["ma10"]) & (ma_df["ma5"] < ma_df["ma20"])

    # 최종 매도 상태: 역배열 2개 이상 or 5일선이 10/20 둘 다 아래
    # bearish_state = bearish_alignment | five_below_10_20  # 역배열 2개 이상 or 5일선이 10/20 둘 다 아래
    bearish_state = bearish_alignment # 역배열 2개 이상

    # 4) 비중 생성
    weights = pd.Series(index=price.index, dtype=float)
    if len(weights) == 0:
        return weights

    # 시작은 0 (현금)
    weights.iloc[0] = 0.0

    for i in range(1, len(price)):
        if bullish_full.iloc[i]:
            # 완전 정배열 → 100% 투자
            weights.iloc[i] = 1.0
        elif bearish_state.iloc[i]:
            # 매도 상태 → 0% (전량 현금)
            weights.iloc[i] = 0.0
        else:
            # 그 외에는 직전 비중 그대로
            weights.iloc[i] = weights.iloc[i - 1]

    # MA60이 형성되지 않은 초기 구간은 0% 강제
    insufficient_ma = ma_df["ma60"].isna()
    weights[insufficient_ma] = 0.0

    weights.name = "weight"
    return weights


def sp500_ma_signal(
    prices: pd.DataFrame,
    sp500_ticker: str = "^GSPC",
    verbose: bool = False,
):
    """
    프로젝트 구조에 맞춘 S&P500 MA 전략 시그널 함수.

    Parameters
    ----------
    prices : pd.DataFrame
        load_prices(TICKERS, start=...) 결과.
        columns 에 sp500_ticker 가 있어야 함.
    sp500_ticker : str
        S&P500 으로 쓸 티커 (예: '^GSPC' 또는 'SPY').
    verbose : bool
        True 면 마지막 날짜 / weight / MA 상태를 콘솔에 출력.

    Returns
    -------
    float
        마지막 날짜 기준 주식 비중 (0.0 또는 1.0 사이, 보통 0 또는 1).
    """
    if sp500_ticker not in prices.columns:
        raise ValueError(f"prices에 '{sp500_ticker}' 컬럼이 없습니다.")

    price = prices[sp500_ticker].dropna()
    if price.empty:
        raise ValueError(f"'{sp500_ticker}' 가격 데이터가 비어 있습니다.")

    weights = _ma_alignment_weights(price)
    last_date = weights.index[-1]
    last_weight = float(weights.iloc[-1])

    if verbose:
        print("\n[My S&P500 MA 전략]")
        print(f" - 기준 티커     : {sp500_ticker}")
        print(f" - 마지막 데이터  : {last_date.date()}")
        print(f" - 현재 주식 비중 : {last_weight*100:5.1f}%")

    weights.attrs["last_weight"] = last_weight

    return last_weight
