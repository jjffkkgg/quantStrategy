# strategies/laaMA3.py
"""
LAA_MA3 전략 (LAA_MA2F에서 IWD 로직 변경)

- LAA_MA2F의 QQQ 로직은 그대로 사용.
- IWD의 ON/OFF 로직을 아래와 같이 변경:
    - 매일 체크.
    - OFF 조건 (매도):
        - '불경기'(실업률>12M SMA) 상태이고,
        - SPY가 200일 이평선 아래에 '3일 연속'으로 있었을 경우,
        - 3일째 되는 날 다음 날부터 OFF(SGOV로 전환).
    - ON 조건 (재진입):
        - OFF로 전환된 후, 최소 '30 거래일'의 재진입 금지 기간(cooldown)을 가짐.
        - 30 거래일이 지난 후, '호경기' 또는 '상승장'(SPY>200MA) 조건이 다시 충족되면 ON.
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
IWD_COOLDOWN_DAYS = 30  # IWD 재진입 쿨다운 (새로 추가)

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
        raise ValueError("LAA_MA3 전략에는 'SPY' 가격 데이터가 필요합니다.")

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

def _laa_ma3_weights_timeseries(prices: pd.DataFrame) -> pd.DataFrame:
    """
    LAA_MA3 전략의 일별 weight 시계열 생성.
    """
    prices = prices.sort_index()
    cols = list(prices.columns)
    idx = prices.index

    required = ["QQQ", "IWD", "IAU", "IEF", "SGOV", "SPY"]
    missing = [t for t in required if t not in cols]
    if missing:
        raise ValueError(f"LAA_MA3 전략에 필요한 ETF 데이터가 없습니다: {missing}")

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
    # 3-2) IWD ON/OFF 로직 (신규)
    # ------------------------------
    # 일별 경기/추세 플래그
    regime = _compute_regime_flags(prices)
    recession = regime["recession"]
    
    spy = prices["SPY"].dropna()
    spy_ma200 = spy.rolling(200).mean()
    spy_uptrend = (spy > spy_ma200).reindex(idx).ffill().fillna(True)

    # IWD 매도 조건: 불경기 & 3일 연속 하락장
    is_downtrend_3_days = (spy_uptrend == False).rolling(3).sum() >= 3
    sell_trigger = recession & is_downtrend_3_days

    # IWD 매수 조건: 호경기 또는 상승장
    buy_condition = ~recession | spy_uptrend

    # IWD ON/OFF 마스크 생성 (상태 머신)
    iwd_on = pd.Series(True, index=idx)
    on_state = True
    cooldown_counter = IWD_COOLDOWN_DAYS + 1
    
    first_valid_index = sell_trigger.first_valid_index()
    if first_valid_index is None:
        iwd_on.iloc[:] = True
    else:
        start_loc = idx.get_loc(first_valid_index)
        iwd_on.iloc[:start_loc] = True

        for i in range(start_loc, len(idx)):
            if on_state:
                if sell_trigger.iloc[i]:
                    on_state = False
                    cooldown_counter = 0
            else:
                cooldown_counter += 1
                if cooldown_counter >= IWD_COOLDOWN_DAYS and buy_condition.iloc[i]:
                    on_state = True
            
            iwd_on.iloc[i] = on_state

    # ------------------------------
    # 3-3) weight DataFrame 구성
    # ------------------------------
    weight_df = pd.DataFrame(0.0, index=idx, columns=cols)

    # 항상 보유 50%
    weight_df["IAU"] = 0.25
    weight_df["IEF"] = 0.25

    # QQQ 슬롯 25%
    weight_df.loc[qqq_on, "QQQ"] = 0.25
    weight_df.loc[~qqq_on, "SGOV"] += 0.25

    # IWD 슬롯 25%
    weight_df.loc[iwd_on, "IWD"] = 0.25
    weight_df.loc[~iwd_on, "SGOV"] += 0.25

    weight_df = weight_df.fillna(0.0)
    return weight_df


def get_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """runBacktest.py에서 사용하는 표준 인터페이스"""
    return _laa_ma3_weights_timeseries(prices)

# ----------------------------------------------------------------------
# 4) 최신 시점 시그널 (runStrategy.py 용)
# ----------------------------------------------------------------------

def laa_ma3_signal(prices: pd.DataFrame, verbose: bool = False) -> Dict[str, float]:
    """
    마지막 날짜 기준 LAA_MA3 포트폴리오 비중 dict 반환.
    """
    prices = prices.sort_index()
    weight_df = _laa_ma3_weights_timeseries(prices)

    last_date = prices.index[-1]
    last_w = weight_df.iloc[-1]
    weights = {k: float(v) for k, v in last_w.items() if abs(v) > 1e-12}

    if verbose:
        print("=== LAA_MA3 Signal (Latest) ===")
        print(f"Date : {last_date.date()}")
        for k, v in weights.items():
            print(f"  {k:5s}: {v*100:5.1f}%")
        print(f"QQQ_COOLDOWN_DAYS = {QQQ_COOLDOWN_DAYS}")
        print(f"IWD_COOLDOWN_DAYS = {IWD_COOLDOWN_DAYS}")
        print("---------------------------------------")

    return weights