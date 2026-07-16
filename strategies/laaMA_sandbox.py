# strategies/laaMA_sandbox.py
"""
LAA_SANDBOX: 다양한 변형 전략을 테스트하기 위한 샌드박스

LAA_MA5 전략을 기반으로, 각 자산의 티커, 비중, 그리고 공격 자산의 매매 로직(타임프레임, 쿨다운) 등을
쉽게 변경하여 여러 아이디어를 빠르게 테스트할 수 있도록 설계된 유연한 전략입니다.

포트폴리오는 4개의 슬롯으로 구성되며, 각 슬롯의 자산과 비중을 자유롭게 설정할 수 있습니다.
"""

from __future__ import annotations
from typing import Dict, List, Literal
import pandas as pd
import numpy as np

from utils.data_loader import load_close_for_ma
from utils.macro_data import load_unemployment_rate
from strategies.laa import _is_recession, _is_market_uptrend

# ----------------------------------------------------------------------
# 설정 (여기서 값을 변경하여 테스트)
# ----------------------------------------------------------------------

# 1. 티커 설정
TICKER_AGGRESSIVE = "QQQ"    # 공격 자산 (MA 전략 적용)
TICKER_VALUE = "IWD"         # 가치 자산 (경기/추세 판단)
TICKER_GOLD = "IAU"          # 금 (수익률 모멘텀)
TICKER_BOND = "IEF"          # 채권 (고정 보유 또는 동적)
TICKER_CASH = "SGOV"         # 현금성 자산
TICKER_MARKET = "SPY"        # 시장 추세 판단용

# 2. 포트폴리오 비중 설정 (합계가 1.0이 되도록)
W_AGGRESSIVE = 0.25
W_VALUE = 0.25
W_GOLD = 0.25
W_BOND = 0.25

# 3. 공격 자산 (AGGRESSIVE) MA 전략 설정
AGGRESSIVE_TIMEFRAME: Literal['daily', 'weekly'] = 'daily' # 'daily' 또는 'weekly'
AGGRESSIVE_MA_PERIODS = [5, 10, 20, 60] # 사용할 이동평균 기간
AGGRESSIVE_COOLDOWN_DAYS = 0  # 재진입 쿨다운 (0이면 비활성화)

# 4. 금 (GOLD) 전략 설정
TICKER_GOLD_REF_BOND = "IEF" # 금 투자 판단시 참고할 채권 ETF
GOLD_LOOKBACK_YEARS = 1 # 1년 수익률
LOOKBACK_1Y = 252 * GOLD_LOOKBACK_YEARS

# 5. 가치 자산 (VALUE) 전략 설정
VALUE_STRATEGY: Literal['REGIME', 'MA', 'HOLD'] = 'REGIME' # 'REGIME', 'MA', 또는 'HOLD'
# 'REGIME': LAA와 동일한 (실업률 > 12M SMA) and (SPY < 200D MA) 로직 사용
# 'MA'    : AGGRESSIVE와 유사한 MA 정배열/역배열 로직 (아래 설정 적용)
# 'HOLD'  : 항상 보유 (전략 끄기)

# 5-1. VALUE 'MA' 전략 사용 시 설정
VALUE_MA_TIMEFRAME: Literal['daily', 'weekly'] = 'daily'
VALUE_MA_PERIODS = [5, 10, 20, 60]
VALUE_COOLDOWN_DAYS = 0

# ----------------------------------------------------------------------
# 내부 로직 (수정 불필요)
# ----------------------------------------------------------------------

def _ma_alignment_weights_configurable(price: pd.Series, periods: List[int]) -> pd.Series:
    """
    customMA._ma_alignment_weights의 설정 가능한 버전.
    주어진 기간(periods)에 따라 정배열/역배열을 판단.
    periods는 짧은 것부터 긴 순서로 정렬되어야 합니다 (예: [5, 10, 20, 60]).
    """
    price = price.astype(float)

    if len(periods) != 4:
        raise ValueError("MA 기간은 4개여야 합니다 (예: [5, 10, 20, 60]).")

    ma_s, ma_m1, ma_m2, ma_l = periods

    ma_df = pd.DataFrame(index=price.index)
    ma_df[f'ma{ma_s}'] = price.rolling(ma_s).mean()
    ma_df[f'ma{ma_m1}'] = price.rolling(ma_m1).mean()
    ma_df[f'ma{ma_m2}'] = price.rolling(ma_m2).mean()
    ma_df[f'ma{ma_l}'] = price.rolling(ma_l).mean()

    # 1) 완전 정배열 조건
    bullish_full = (
        (ma_df[f'ma{ma_s}'] > ma_df[f'ma{ma_m1}'])
        & (ma_df[f'ma{ma_m1}'] > ma_df[f'ma{ma_m2}'])
        & (ma_df[f'ma{ma_m2}'] > ma_df[f'ma{ma_l}'])
    )

    # 2) 역배열 "상태" (3개 조건 중 2개 이상 충족)
    rev_s_m1 = ma_df[f'ma{ma_s}'] < ma_df[f'ma{ma_m1}']
    rev_m1_m2 = ma_df[f'ma{ma_m1}'] < ma_df[f'ma{ma_m2}']
    rev_m2_l = ma_df[f'ma{ma_m2}'] < ma_df[f'ma{ma_l}']

    num_reversed = (
        rev_s_m1.astype(int)
        + rev_m1_m2.astype(int)
        + rev_m2_l.astype(int)
    )
    bearish_state = num_reversed >= 2

    # 3) 비중 생성
    weights = pd.Series(np.nan, index=price.index, dtype=float)
    if len(weights) == 0:
        return weights

    weights.iloc[0] = 0.0

    for i in range(1, len(price)):
        if bullish_full.iloc[i]:
            weights.iloc[i] = 1.0
        elif bearish_state.iloc[i]:
            weights.iloc[i] = 0.0
        else:
            weights.iloc[i] = weights.iloc[i - 1]

    insufficient_ma = ma_df[f'ma{ma_l}'].isna()
    weights[insufficient_ma] = 0.0

    weights = weights.ffill() # 시작 부분 NaN 채우기

    weights.name = "weight"
    return weights

def _compute_regime_flags(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index()
    if TICKER_MARKET not in prices.columns:
        raise ValueError(f"Sandbox 전략에는 '{TICKER_MARKET}' 가격 데이터가 필요합니다.")

    unrate_full = load_unemployment_rate().dropna()
    monthly_idx = prices.resample("ME").last().index

    rows = []
    idxs = []

    for dt in monthly_idx:
        unrate_sub = unrate_full[unrate_full.index <= dt]
        spy_sub = prices[TICKER_MARKET].loc[:dt].dropna()

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

def _apply_reentry_cooldown_mask(qqq_on: pd.Series, cooldown_days: int) -> pd.Series:
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

def _laa_sandbox_weights_timeseries(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index()
    idx = prices.index

    all_tickers = list(set([
        TICKER_AGGRESSIVE, TICKER_VALUE, TICKER_GOLD, TICKER_BOND,
        TICKER_CASH, TICKER_MARKET, TICKER_GOLD_REF_BOND
    ]))

    missing = [t for t in all_tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Sandbox 전략에 필요한 ETF 데이터가 없습니다: {missing}")

    start_str = idx[0].strftime("%Y-%m-%d")

    # 1. 공격 자산 (AGGRESSIVE) MA 기반 ON/OFF
    aggressive_close = load_close_for_ma(TICKER_AGGRESSIVE, start=start_str)

    if AGGRESSIVE_TIMEFRAME == 'weekly':
        resample_freq = 'W-FRI'
        price_for_ma = aggressive_close.resample(resample_freq).last().dropna()
    else: # daily
        price_for_ma = aggressive_close.dropna()

    w_aggressive_signal = _ma_alignment_weights_configurable(price_for_ma, periods=AGGRESSIVE_MA_PERIODS)
    w_aggressive_daily = w_aggressive_signal.reindex(idx, method='ffill').ffill().fillna(0.0)

    aggressive_on_raw = (w_aggressive_daily >= 0.5)
    aggressive_on = _apply_reentry_cooldown_mask(aggressive_on_raw, cooldown_days=AGGRESSIVE_COOLDOWN_DAYS)

    # 2. 가치 자산 (VALUE) ON/OFF
    if VALUE_STRATEGY == 'REGIME':
        # 경기/추세 기반 ON/OFF (LAA 로직)
        regime = _compute_regime_flags(prices)
        rec = regime["recession"]
        up = regime["uptrend"]
        value_on = ~(rec & (~up))
    elif VALUE_STRATEGY == 'MA':
        # MA 정배열/역배열 기반 ON/OFF
        value_close = load_close_for_ma(TICKER_VALUE, start=start_str)

        if VALUE_MA_TIMEFRAME == 'weekly':
            resample_freq = 'W-FRI'
            price_for_ma = value_close.resample(resample_freq).last().dropna()
        else: # daily
            price_for_ma = value_close.dropna()

        w_value_signal = _ma_alignment_weights_configurable(price_for_ma, periods=VALUE_MA_PERIODS)
        w_value_daily = w_value_signal.reindex(idx, method='ffill').ffill().fillna(0.0)
        value_on_raw = (w_value_daily >= 0.5)
        value_on = _apply_reentry_cooldown_mask(value_on_raw, cooldown_days=VALUE_COOLDOWN_DAYS)
    elif VALUE_STRATEGY == 'HOLD':
        # 항상 보유 (전략 끄기)
        value_on = pd.Series(True, index=idx)
    else:
        raise ValueError(f"지원하지 않는 VALUE_STRATEGY 입니다: {VALUE_STRATEGY}")

    # 3. 금 (GOLD) 1년 수익률 기반 ON/OFF
    ret_gold_1y = prices[TICKER_GOLD].pct_change(LOOKBACK_1Y, fill_method=None)
    ret_ref_bond_1y = prices[TICKER_GOLD_REF_BOND].pct_change(LOOKBACK_1Y, fill_method=None)

    gold_on_signal = (ret_gold_1y > 0) & (ret_ref_bond_1y > 0)
    gold_on_monthly = gold_on_signal.resample("ME").last()

    gold_on = gold_on_monthly.reindex(idx, method='ffill').fillna(False)

    # 4. weight DataFrame 구성
    weight_df = pd.DataFrame(0.0, index=idx, columns=prices.columns)

    weight_df[TICKER_BOND] = W_BOND

    weight_df.loc[aggressive_on, TICKER_AGGRESSIVE] = W_AGGRESSIVE
    weight_df.loc[~aggressive_on, TICKER_CASH] += W_AGGRESSIVE

    weight_df.loc[value_on, TICKER_VALUE] = W_VALUE
    weight_df.loc[~value_on, TICKER_CASH] += W_VALUE

    weight_df.loc[gold_on, TICKER_GOLD] = W_GOLD
    weight_df.loc[~gold_on, TICKER_CASH] += W_GOLD

    weight_df = weight_df.fillna(0.0)
    return weight_df


def get_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """runBacktest.py에서 사용하는 표준 인터페이스"""
    return _laa_sandbox_weights_timeseries(prices)

def laa_sandbox_signal(prices: pd.DataFrame, verbose: bool = False) -> Dict[str, float]:
    """마지막 날짜 기준 Sandbox 포트폴리오 비중 dict 반환."""
    prices = prices.sort_index()
    weight_df = _laa_sandbox_weights_timeseries(prices)

    last_date = prices.index[-1]
    last_w = weight_df.iloc[-1]
    weights = {k: float(v) for k, v in last_w.items() if abs(v) > 1e-12}

    if verbose:
        print("=== LAA_SANDBOX Signal (Latest) ===")
        print(f"Date : {last_date.date()}")
        print("--- Config ---")
        print(f"  Aggressive: {TICKER_AGGRESSIVE} ({W_AGGRESSIVE*100:.1f}%) | MA({AGGRESSIVE_TIMEFRAME}, cd={AGGRESSIVE_COOLDOWN_DAYS})")
        print(f"  Value     : {TICKER_VALUE} ({W_VALUE*100:.1f}%) | Strategy: {VALUE_STRATEGY}")
        if VALUE_STRATEGY == 'MA':
            print(f"    └ MA Config: {VALUE_MA_TIMEFRAME}, periods={VALUE_MA_PERIODS}, cd={VALUE_COOLDOWN_DAYS}")
        print(f"  Gold      : {TICKER_GOLD} ({W_GOLD*100:.1f}%) | 1Y Ret vs {TICKER_GOLD_REF_BOND}")
        print(f"  Bond      : {TICKER_BOND} ({W_BOND*100:.1f}%) | Fixed")
        print(f"  Cash      : {TICKER_CASH}")
        print("--- Final Weights ---")
        for k, v in weights.items():
            print(f"  {k:5s}: {v*100:5.1f}%")
        print("---------------------------------------")

    return weights