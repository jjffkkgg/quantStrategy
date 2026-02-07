# strategies/laaMA2.py
"""
LAA_MA2 전략

- 기본 구조:
    * IAU 25%, IEF 25%는 항상 보유
    * QQQ 25% : QQQ MA 정배열 전략으로 ON/OFF
    * IWD 25% : 기존 LAA(GT)에서 쓰던
                "불경기 & 하락장" 구간에서만 OFF, 그 외에는 ON
      → OFF 된 비중은 SGOV 로 이동

- QQQ MA 로직:
    * strategies.customMA._ma_alignment_weights 를 사용
    * 완전 정배열 → 1.0, 매도 상태 → 0.0, 그 외 직전값 유지
    * 여기서는 0.5 이상이면 "ON" 으로 간주

- IWD 로직:
    * 실업률 (UNRATE) > 12M 평균 → recession = True
    * SPY < 200D MA → uptrend = False
    * recession & (not uptrend) 인 날만 IWD OFF (SGOV 로 이동),
      나머지 날짜는 IWD 25% 보유
"""

from typing import Dict

import pandas as pd

from utils.data_loader import load_close_for_ma
from utils.macro_data import load_unemployment_rate
from strategies.customMA import _ma_alignment_weights
from strategies.laa import _is_recession, _is_market_uptrend


# ----------------------------------------------------------------------
# 1. 경기/추세 regime 시리즈 계산 (월말 기준 후 일별로 ffill)
# ----------------------------------------------------------------------


def _compute_regime_flags(prices: pd.DataFrame) -> pd.DataFrame:
    """
    prices (Adj Close) 를 받아서,
    월말 기준으로 recession / uptrend 플래그를 구한 뒤
    일별로 forward-fill 한 DataFrame을 반환.

    반환:
        index  : prices.index (일별)
        columns: ["recession", "uptrend"] (bool)
    """
    prices = prices.sort_index()
    if "SPY" not in prices.columns:
        raise ValueError("LAA_MA2 전략에는 'SPY' 가격 데이터가 필요합니다.")

    # 전체 실업률 시계열
    unrate_full = load_unemployment_rate().dropna()

    # 월말 인덱스 (마지막 영업일 기준)
    monthly_idx = prices.resample("ME").last().index

    rows = []
    idxs = []

    for dt in monthly_idx:
        # dt 시점까지의 실업률 / SPY 정보만 사용 (룩어헤드 방지)
        unrate_sub = unrate_full[unrate_full.index <= dt]
        spy_sub = prices["SPY"].loc[:dt].dropna()

        # 실업률 12M + SPY 200D 계산 가능한지 체크
        if len(unrate_sub) < 13:
            continue
        if len(spy_sub) < 200:
            continue

        rec_flag = _is_recession(unrate_sub)
        up_flag = _is_market_uptrend(spy_sub)

        rows.append({"recession": rec_flag, "uptrend": up_flag})
        idxs.append(dt)

    if not rows:
        # 초기 구간에는 regime 정보가 없을 수도 있음
        regime_m = pd.DataFrame(
            columns=["recession", "uptrend"],
            index=pd.DatetimeIndex([]),
        )
    else:
        regime_m = pd.DataFrame(rows, index=pd.DatetimeIndex(idxs))

    # 일별 인덱스로 맞추고 ffill
    regime_d = regime_m.reindex(prices.index).ffill()

    # NaN 이 남아있으면 기본값 설정:
    #   recession: False (불경기 아님)
    #   uptrend  : True  (상승장으로 가정)
    regime_d["recession"] = regime_d["recession"].fillna(False)
    regime_d["uptrend"] = regime_d["uptrend"].fillna(True)

    return regime_d


# ----------------------------------------------------------------------
# 2. 백테스트용 weight 시계열
# ----------------------------------------------------------------------


def _laa_ma2_weights_timeseries(prices: pd.DataFrame) -> pd.DataFrame:
    """
    LAA_MA2 전략의 일별 weight 시계열 생성.

    - 항상:
        IAU 25%, IEF 25%
    - QQQ 25%:
        QQQ MA ON  → QQQ 25%
        QQQ MA OFF → SGOV 25%
    - IWD 25%:
        (recession & 하락장) → IWD OFF → SGOV 25%
        그 외               → IWD 25%
    """
    prices = prices.sort_index()
    cols = list(prices.columns)

    required = ["QQQ", "IWD", "IAU", "IEF", "SGOV", "SPY"]
    missing = [t for t in required if t not in cols]
    if missing:
        raise ValueError(f"LAA_MA2 전략에 필요한 ETF 데이터가 없습니다: {missing}")

    idx = prices.index
    start_str = idx[0].strftime("%Y-%m-%d")

    # ------------------------------
    # 2-1) QQQ MA 기반 온/오프 시리즈
    # ------------------------------
    qqq_close = load_close_for_ma("QQQ", start=start_str)
    qqq_close = qqq_close.reindex(idx).ffill()

    w_qqq = _ma_alignment_weights(qqq_close)        # 0~1
    w_qqq = w_qqq.reindex(idx).ffill().fillna(0.0)

    qqq_on_mask = w_qqq >= 0.5

    # ------------------------------
    # 2-2) 경기/추세 regime 시리즈
    # ------------------------------
    regime = _compute_regime_flags(prices)
    rec = regime["recession"]
    up = regime["uptrend"]

    # IWD 온/오프:
    #   recession & (not uptrend) → OFF
    #   그 외 → ON
    iwd_on_mask = ~(rec & (~up))

    # ------------------------------
    # 2-3) weight DataFrame 구성
    # ------------------------------
    weight_df = pd.DataFrame(0.0, index=idx, columns=cols)

    # 항상 보유 50%
    weight_df["IAU"] = 0.25
    weight_df["IEF"] = 0.25

    # QQQ 슬롯 25%
    if "QQQ" in weight_df.columns:
        weight_df.loc[qqq_on_mask, "QQQ"] = 0.25
    if "SGOV" in weight_df.columns:
        weight_df.loc[~qqq_on_mask, "SGOV"] += 0.25

    # IWD 슬롯 25%
    if "IWD" in weight_df.columns:
        weight_df.loc[iwd_on_mask, "IWD"] = 0.25
    if "SGOV" in weight_df.columns:
        weight_df.loc[~iwd_on_mask, "SGOV"] += 0.25

    # 혹시 모를 NaN 방지
    weight_df = weight_df.fillna(0.0)

    return weight_df


def get_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """
    runBacktest.py 에서 사용하는 표준 인터페이스.
    """
    return _laa_ma2_weights_timeseries(prices)


# ----------------------------------------------------------------------
# 3. 최신 시점 시그널 (runStrategy.py 용)
# ----------------------------------------------------------------------


def laa_ma2_signal(
    prices: pd.DataFrame,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    마지막 날짜 기준 LAA_MA2 포트폴리오 비중 dict 반환.
    (runStrategy.py 에서 현재 시그널 출력용)
    """
    prices = prices.sort_index()
    weight_df = _laa_ma2_weights_timeseries(prices)

    last_date = prices.index[-1]
    last_w = weight_df.iloc[-1]

    # 0인 애들은 굳이 안 보고 싶으면 여기서 필터링해도 됨
    weights = {k: float(v) for k, v in last_w.items() if abs(v) > 1e-12}

    if verbose:
        print("=== LAA_MA2 Signal (Latest) ===")
        print(f"Date : {last_date.date()}")
        for k, v in weights.items():
            print(f"  {k:5s}: {v*100:5.1f}%")
        print("---------------------------------------")

    return weights
