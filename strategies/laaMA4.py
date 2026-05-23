# strategies/laaMA4.py
"""
LAA_MA4: LAA_MA2F 기반 IAU 동적 자산배분 전략

LAA_MA4는 LAA_MA2F 전략을 기반으로, 금(IAU) 투자 비중을 동적으로 조절하는 새로운 규칙을 추가한 파생 전략입니다.
포트폴리오는 4개의 25% 슬롯으로 구성되며, 각 슬롯은 다음과 같은 규칙에 따라 운용됩니다.

1.  **고정 보유 자산 (25%):**
    -   `IEF` (미국 중기 국채 ETF): 25% 비중을 항상 유지합니다. 포트폴리오의 안정적인 기반 역할을 합니다.

2.  **QQQ 동적 운용 (25%):**
    -   `LAA_MA2F`와 동일한 로직을 사용합니다.
    -   **신호:** QQQ의 단기/중기/장기 이동평균선들의 정배열 상태(`_ma_alignment_weights`)를 기반으로 매일 ON/OFF 신호를 생성합니다. (w >= 0.5 이면 ON)
    -   **재진입 지연 필터:** 잦은 매매를 방지하기 위해, QQQ 매도(OFF) 신호 발생 후 `QQQ_COOLDOWN_DAYS` (기본 30 거래일) 동안은 매수(ON) 신호가 다시 발생해도 무시하고 현금(SGOV) 보유 상태를 유지합니다.
    -   **배분:**
        -   ON 신호: QQQ 25% 보유
        -   OFF 신호: 해당 25%를 SGOV (단기 국채 ETF)로 전환

3.  **IWD 동적 운용 (25%):**
    -   `LAA_MA2F`와 동일한 로직을 사용합니다.
    -   **신호:** 거시 경제 지표(실업률)와 시장 추세(SPY 200일 이평선)를 조합하여 '불경기 & 하락장' 여부를 판단합니다.
        -   `_is_recession`: 미국 실업률 > 12개월 이동평균
        -   `_is_market_uptrend`: SPY 종가 > 200일 이동평균
    -   **배분:**
        -   '불경기'이면서 '하락장'인 구간: IWD를 매도하고 해당 25%를 SGOV로 전환 (리스크 오프)
        -   그 외 모든 구간: IWD 25% 보유 (리스크 온)

4.  **IAU 동적 운용 (25%) - (LAA_MA4의 핵심 변경점):**
    -   **신호:** '월말'에만 아래 두 가지 조건을 동시에 만족하는지 체크합니다.
        1.  `IAU`의 최근 1년 수익률 > 0
        2.  `IEF`의 최근 1년 수익률 > 0
    -   **배분:**
        -   월말에 두 조건을 모두 만족: 다음 한 달간 `IAU` 25% 보유
        -   월말에 하나라도 만족하지 못함: 다음 한 달간 해당 25%를 `SGOV`로 전환

이 전략의 목표는 기존 LAA_MA2F의 QQQ/IWD 동적 운용을 유지하면서,
금(IAU)이 채권(IEF)과 함께 동반 상승하는 국면에서만 금에 투자하고,
그렇지 않은 시기에는 현금으로 전환하여 안정성을 높이는 것입니다.
"""

from __future__ import annotations
from typing import Dict
import pandas as pd

from utils.data_loader import load_close_for_ma
from utils.macro_data import load_unemployment_rate
from strategies.customMA import _ma_alignment_weights
from strategies.laa import _is_recession, _is_market_uptrend

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------

QQQ_COOLDOWN_DAYS = 30  # QQQ 재진입 쿨다운 (laaMA2F와 동일)
LOOKBACK_1Y = 252 # 1년 수익률 계산을 위한 lookback 기간 (영업일 기준)

# ----------------------------------------------------------------------
# 1) (LAA_MA2F와 동일) 경기/추세 regime 시리즈 계산
# ----------------------------------------------------------------------

def _compute_regime_flags(prices: pd.DataFrame) -> pd.DataFrame:
    """
    laaMA2.py와 동일한 구현.
    월말 기준으로 recession / uptrend 계산 후, 일별로 forward-fill.
    """
    prices = prices.sort_index()
    if "SPY" not in prices.columns:
        raise ValueError("LAA_MA4 전략에는 'SPY' 가격 데이터가 필요합니다.")

    unrate_full = load_unemployment_rate().dropna()
    monthly_idx = prices.resample("ME").last().index

    rows = []
    idxs = []

    for dt in monthly_idx:
        unrate_sub = unrate_full[unrate_full.index <= dt]
        spy_sub = prices["SPY"].loc[:dt].dropna()

        if len(unrate_sub) < 13:
            continue
        if len(spy_sub) < 200:
            continue

        rec_flag = _is_recession(unrate_sub)
        up_flag = _is_market_uptrend(spy_sub)

        rows.append({"recession": rec_flag, "uptrend": up_flag})
        idxs.append(dt)

    if not rows:
        regime_m = pd.DataFrame(columns=["recession", "uptrend"], index=pd.DatetimeIndex([]))
    else:
        regime_m = pd.DataFrame(rows, index=pd.DatetimeIndex(idxs))

    regime_d = regime_m.reindex(prices.index).ffill()
    regime_d["recession"] = regime_d["recession"].fillna(False)
    regime_d["uptrend"] = regime_d["uptrend"].fillna(True)

    return regime_d

# ----------------------------------------------------------------------
# 2) (LAA_MA2F와 동일) QQQ ON/OFF 마스크 "재진입 쿨다운" 필터
# ----------------------------------------------------------------------

def _apply_reentry_cooldown_mask(qqq_on: pd.Series, cooldown_days: int) -> pd.Series:
    """
    laaMA2F.py와 동일한 구현.
    """
    if cooldown_days is None or cooldown_days <= 0:
        return qqq_on.astype(bool)

    s = qqq_on.astype(bool).copy()
    if s.empty:
        return s

    on = bool(s.iloc[0])
    cd = cooldown_days

    out = pd.Series(index=s.index, dtype=bool)
    out.iloc[0] = on

    for i in range(1, len(s)):
        raw_on = bool(s.iloc[i])

        if on:
            if not raw_on:
                on = False
                cd = 0
                out.iloc[i] = False
            else:
                out.iloc[i] = True
        else:
            if cd < cooldown_days:
                cd += 1
            if raw_on and cd >= cooldown_days:
                on = True
                out.iloc[i] = True
            else:
                out.iloc[i] = False
    return out

# ----------------------------------------------------------------------
# 3) 백테스트용 weight 시계열 (일별)
# ----------------------------------------------------------------------

def _laa_ma4_weights_timeseries(prices: pd.DataFrame) -> pd.DataFrame:
    """
    LAA_MA4 전략의 일별 weight 시계열 생성.
    """
    prices = prices.sort_index()
    cols = list(prices.columns)
    idx = prices.index

    required = ["QQQ", "IWD", "IAU", "IEF", "SGOV", "SPY"]
    missing = [t for t in required if t not in cols]
    if missing:
        raise ValueError(f"LAA_MA4 전략에 필요한 ETF 데이터가 없습니다: {missing}")

    start_str = idx[0].strftime("%Y-%m-%d")

    # ------------------------------
    # 3-1) QQQ MA 기반 ON/OFF (laaMA2F와 동일)
    # ------------------------------
    qqq_close = load_close_for_ma("QQQ", start=start_str)
    qqq_close = qqq_close.reindex(idx).ffill()

    w_qqq = _ma_alignment_weights(qqq_close)
    w_qqq = w_qqq.reindex(idx).ffill().fillna(0.0)
    qqq_on_raw = (w_qqq >= 0.5)
    qqq_on = _apply_reentry_cooldown_mask(qqq_on_raw, cooldown_days=QQQ_COOLDOWN_DAYS)

    # ------------------------------
    # 3-2) IWD 경기/추세 기반 ON/OFF (laaMA2F와 동일)
    # ------------------------------
    regime = _compute_regime_flags(prices)
    rec = regime["recession"]
    up = regime["uptrend"]
    iwd_on = ~(rec & (~up))

    # ------------------------------
    # 3-3) IAU 1년 수익률 기반 ON/OFF (신규 로직)
    # ------------------------------
    ret_iau_1y = prices["IAU"].pct_change(LOOKBACK_1Y, fill_method=None)
    ret_ief_1y = prices["IEF"].pct_change(LOOKBACK_1Y, fill_method=None)
    
    # 월말에만 시그널 계산
    iau_on_signal = (ret_iau_1y > 0) & (ret_ief_1y > 0)
    # 월말(Month End)의 시그널을 가져옴
    iau_on_monthly = iau_on_signal.resample("ME").last()

    # 일별로 시그널 확장 (forward-fill)
    iau_on = iau_on_monthly.reindex(idx, method='ffill')
    
    iau_on = iau_on.fillna(False) # 데이터 부족 구간은 OFF

    # ------------------------------
    # 3-4) weight DataFrame 구성
    # ------------------------------
    weight_df = pd.DataFrame(0.0, index=idx, columns=cols)

    # 항상 보유 25%
    weight_df["IEF"] = 0.25

    # QQQ 슬롯 25%
    weight_df.loc[qqq_on, "QQQ"] = 0.25
    weight_df.loc[~qqq_on, "SGOV"] += 0.25

    # IWD 슬롯 25%
    weight_df.loc[iwd_on, "IWD"] = 0.25
    weight_df.loc[~iwd_on, "SGOV"] += 0.25
    
    # IAU 슬롯 25%
    weight_df.loc[iau_on, "IAU"] = 0.25
    weight_df.loc[~iau_on, "SGOV"] += 0.25

    weight_df = weight_df.fillna(0.0)
    return weight_df


def get_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """runBacktest.py에서 사용하는 표준 인터페이스"""
    return _laa_ma4_weights_timeseries(prices)

# ----------------------------------------------------------------------
# 4) 최신 시점 시그널 (runStrategy.py 용)
# ----------------------------------------------------------------------

def laa_ma4_signal(prices: pd.DataFrame, verbose: bool = False) -> Dict[str, float]:
    """
    마지막 날짜 기준 LAA_MA4 포트폴리오 비중 dict 반환.
    """
    prices = prices.sort_index()
    weight_df = _laa_ma4_weights_timeseries(prices)

    last_date = prices.index[-1]
    last_w = weight_df.iloc[-1]
    weights = {k: float(v) for k, v in last_w.items() if abs(v) > 1e-12}

    if verbose:
        print("=== LAA_MA4 Signal (Latest) ===")
        print(f"Date : {last_date.date()}")
        print("IAU signal is checked at month-end and held for the next month.")
        
        monthly_idx = prices.resample("ME").last().index
        relevant_monthly_dates = monthly_idx[monthly_idx <= last_date]
        
        if not relevant_monthly_dates.empty:
            last_signal_date = relevant_monthly_dates[-1]
            print(f"Last IAU Signal Date: {last_signal_date.date()}")

            if len(prices.loc[:last_signal_date]) > LOOKBACK_1Y:
                ret_iau_1y = prices["IAU"].pct_change(LOOKBACK_1Y, fill_method=None).asof(last_signal_date)
                ret_ief_1y = prices["IEF"].pct_change(LOOKBACK_1Y, fill_method=None).asof(last_signal_date)
                iau_on_at_signal = (ret_iau_1y > 0) and (ret_ief_1y > 0) if pd.notna(ret_iau_1y) and pd.notna(ret_ief_1y) else False
                print(f"  IAU 1Y Return: {ret_iau_1y*100:.2f}%")
                print(f"  IEF 1Y Return: {ret_ief_1y*100:.2f}%")
                print(f"  -> IAU Hold  : {iau_on_at_signal}")
            else:
                print("  (Not enough data for IAU signal calculation at last signal date)")
        else:
            print("  (No month-end signal date found in the given price range)")

        print("---------------------------------------")
        print("Final Weights:")
        for k, v in weights.items():
            print(f"  {k:5s}: {v*100:5.1f}%")
        print(f"QQQ_COOLDOWN_DAYS = {QQQ_COOLDOWN_DAYS}")
        print("---------------------------------------")

    return weights