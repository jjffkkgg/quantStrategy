# strategies/laaMA2F.py
"""
LAA_MA2F 전략 (LAA_MA2 + Re-entry Cooldown Filter)

중요한 목표:
- COOLDOWN_DAYS = 0 일 때는 LAA_MA2와 "완전히 동일"해야 한다.
- 즉, base 로직/임계값/리밸런싱 단위는 laaMA2.py와 1:1로 맞추고,
  QQQ ON/OFF 신호에만 '재진입 지연' 필터를 얹는다.

LAA_MA2 (baseline) 요약:
- IAU 25%, IEF 25%는 항상 보유
- QQQ 25% : QQQ MA 정배열 전략으로 ON/OFF (w>=0.5면 ON)
- IWD 25% : (불경기 & 하락장)에서만 OFF, 그 외 ON
- OFF된 슬롯은 SGOV로 이동

추가되는 Filter:
- QQQ가 OFF로 바뀐 날(SELL) 이후 N 거래일 동안은
  QQQ raw 신호가 ON으로 돌아와도 무시하고 OFF 유지
  (wash sale / 횡보장 잦은 왕복 완화 목적)
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

COOLDOWN_DAYS = 30  # 0이면 필터 OFF (LAA_MA2와 완전 동일)


# ----------------------------------------------------------------------
# 1) (LAA_MA2와 동일) 경기/추세 regime 시리즈 계산 (월말 기준 후 일별 ffill)
# ----------------------------------------------------------------------

def _compute_regime_flags(prices: pd.DataFrame) -> pd.DataFrame:
    """
    laaMA2.py와 동일한 구현.
    월말 기준으로 recession / uptrend 계산 후, 일별로 forward-fill.
    """
    prices = prices.sort_index()
    if "SPY" not in prices.columns:
        raise ValueError("LAA_MA2F 전략에는 'SPY' 가격 데이터가 필요합니다.")

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
# 2) QQQ ON/OFF 마스크에만 적용되는 "재진입 쿨다운" 필터
# ----------------------------------------------------------------------

def _apply_reentry_cooldown_mask(qqq_on: pd.Series, cooldown_days: int) -> pd.Series:
    """
    qqq_on: bool Series (True=QQQ ON, False=OFF), 일별 인덱스

    - cooldown_days <= 0 이면 "필터 OFF": 입력 그대로 반환
      => 이게 있어야 COOLDOWN=0에서 LAA_MA2와 결과가 완전히 동일해짐.

    필터 규칙:
    - ON -> OFF로 바뀌는 날을 SELL로 간주 (즉시 OFF)
    - SELL 이후 cooldown_days 거래일 동안은
      raw가 ON이어도 강제로 OFF 유지
    """
    if cooldown_days is None or cooldown_days <= 0:
        # 필터 완전 OFF: 입력 그대로
        return qqq_on.astype(bool)

    s = qqq_on.astype(bool).copy()
    if s.empty:
        return s

    on = bool(s.iloc[0])
    cd = cooldown_days  # 시작은 "충분히 지난 상태"로 두면 초기 BUY 제한이 생기지 않음

    out = pd.Series(index=s.index, dtype=bool)
    out.iloc[0] = on

    for i in range(1, len(s)):
        raw_on = bool(s.iloc[i])

        if on:
            # ON 상태
            if not raw_on:
                # SELL 발생
                on = False
                cd = 0
                out.iloc[i] = False
            else:
                out.iloc[i] = True
        else:
            # OFF 상태
            if cd < cooldown_days:
                cd += 1

            # 쿨다운이 끝나야만 재진입 허용
            if raw_on and cd >= cooldown_days:
                on = True
                out.iloc[i] = True
            else:
                out.iloc[i] = False

    return out


# ----------------------------------------------------------------------
# 3) 백테스트용 weight 시계열 (일별)  === laaMA2.py와 동일 구조 ===
# ----------------------------------------------------------------------

def _laa_ma2f_weights_timeseries(prices: pd.DataFrame) -> pd.DataFrame:
    """
    LAA_MA2F 전략의 일별 weight 시계열 생성.
    laaMA2.py와 동일하게 "일별 weight_df"를 만든다.
    """
    prices = prices.sort_index()
    cols = list(prices.columns)

    required = ["QQQ", "IWD", "IAU", "IEF", "SGOV", "SPY"]
    missing = [t for t in required if t not in cols]
    if missing:
        raise ValueError(f"LAA_MA2F 전략에 필요한 ETF 데이터가 없습니다: {missing}")

    idx = prices.index
    start_str = idx[0].strftime("%Y-%m-%d")

    # ------------------------------
    # 3-1) QQQ MA 기반 ON/OFF (laaMA2.py와 동일)
    # ------------------------------
    qqq_close = load_close_for_ma("QQQ", start=start_str)
    qqq_close = qqq_close.reindex(idx).ffill()

    w_qqq = _ma_alignment_weights(qqq_close)  # 0~1
    w_qqq = w_qqq.reindex(idx).ffill().fillna(0.0)

    # laaMA2.py와 동일: 0.5 이상이면 ON
    qqq_on_raw = (w_qqq >= 0.5)

    # ★ 여기만 추가: 재진입 쿨다운 필터 적용
    qqq_on = _apply_reentry_cooldown_mask(qqq_on_raw, cooldown_days=COOLDOWN_DAYS)

    # ------------------------------
    # 3-2) 경기/추세 regime 시리즈 (laaMA2.py와 동일)
    # ------------------------------
    regime = _compute_regime_flags(prices)
    rec = regime["recession"]
    up = regime["uptrend"]

    # IWD: recession & (not uptrend)일 때만 OFF
    iwd_on = ~(rec & (~up))

    # ------------------------------
    # 3-3) weight DataFrame 구성 (laaMA2.py와 동일)
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
    return _laa_ma2f_weights_timeseries(prices)


# ----------------------------------------------------------------------
# 4) 최신 시점 시그널 (runStrategy.py 용)
# ----------------------------------------------------------------------

def laa_ma2f_signal(prices: pd.DataFrame, verbose: bool = False) -> Dict[str, float]:
    """
    마지막 날짜 기준 LAA_MA2F 포트폴리오 비중 dict 반환.
    """
    prices = prices.sort_index()
    weight_df = _laa_ma2f_weights_timeseries(prices)

    last_date = prices.index[-1]
    last_w = weight_df.iloc[-1]
    weights = {k: float(v) for k, v in last_w.items() if abs(v) > 1e-12}

    if verbose:
        print("=== LAA_MA2F Signal (Latest) ===")
        print(f"Date : {last_date.date()}")
        for k, v in weights.items():
            print(f"  {k:5s}: {v*100:5.1f}%")
        print(f"COOLDOWN_DAYS = {COOLDOWN_DAYS}")
        print("---------------------------------------")

    return weights
