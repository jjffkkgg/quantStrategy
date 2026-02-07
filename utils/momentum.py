# utils/momentum.py
import pandas as pd


def compute_momentum(prices: pd.DataFrame, months, trading_days_per_month=21):
    """
    마지막 시점 기준으로 여러 개월 모멘텀(수익률)을 계산.
    :param prices: 가격 DataFrame
    :param months: [3, 6, 12] 같은 개월 리스트
    :param trading_days_per_month: 한 달 거래일 수 가정 (기본 21)
    :return: DataFrame(index=tickers, columns=f"{m}m")
    """
    last_row = prices.iloc[-1]
    mom_dict = {}
    for m in months:
        window = m * trading_days_per_month
        if len(prices) <= window:
            # 데이터가 부족하면 NaN
            mom_dict[f"{m}m"] = pd.Series(index=prices.columns, dtype=float)
        else:
            ret = prices.pct_change(window).iloc[-1]
            mom_dict[f"{m}m"] = ret

    return pd.DataFrame(mom_dict)
