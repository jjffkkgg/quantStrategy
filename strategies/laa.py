# strategies/laa.py
# ================================================================
#   LAA (Growth-Trend) Strategy — D Version (with SGOV instead of SHY)
#
#   - Core 75% is always: IWD 25%, IAU 25%, IEF 25%
#   - Remaining 25% switches between:
#         QQQ  (Risk-On)
#         SGOV (Risk-Off)
#
#   - Switching rule (D-version GT matrix):
#       * Recession (UNRATE > 12M SMA)  AND
#         Bear Market (SPY < 200MA)      → SGOV 25%
#       * Otherwise                      → QQQ 25%
#
#   - Rebalanced monthly (month-end)
#   - USES:
#         • UNRATE (load_unemployment_rate)
#         • SPY price series
# ================================================================

from typing import Dict, Optional
import pandas as pd

from utils.macro_data import load_unemployment_rate


# ------------------------------------------------------------
# 1) 경기 판단 함수 (UNRATE > 12M SMA)
# ------------------------------------------------------------
def _is_recession(unrate: pd.Series) -> bool:
    """
    실업률 기반 경기 판단.

    Returns
    -------
    bool
        True  → 불경기 (Recession)
        False → 호경기 (Expansion)
    """
    unrate = unrate.dropna()
    if len(unrate) < 13:
        raise ValueError("실업률 데이터가 부족하여 12M SMA 계산 불가.")

    last_value = float(unrate.iloc[-1])
    sma12 = float(unrate.tail(12).mean())

    # 불경기 기준: 마지막 값이 12M 이동평균보다 높으면 Recession
    return last_value > sma12


# ------------------------------------------------------------
# 2) 시장 추세 판단 함수 (SPY > 200D MA)
# ------------------------------------------------------------
def _is_market_uptrend(spy: pd.Series) -> bool:
    """
    SPY의 200일 이동평균 기반 추세 판단.

    Returns
    -------
    bool
        True  → 상승 추세
        False → 하락 추세
    """
    spy = spy.dropna().astype(float)
    if len(spy) < 200:
        raise ValueError("SPY 데이터가 200일 이동평균을 계산하기에 부족합니다.")

    ma200 = spy.rolling(200).mean().iloc[-1]
    last_price = spy.iloc[-1]

    return last_price > ma200


# ------------------------------------------------------------
# 3) Single Moment LAA(GT) Signal
# ------------------------------------------------------------
def laa_signal(
    prices: pd.DataFrame,
    unrate: Optional[pd.Series] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    LAA(GT) — Growth/Trend 기반 LAA 전략 (D 버전, SHY 대신 SGOV)

    Parameters
    ----------
    prices : pd.DataFrame
        SPY / QQQ / SGOV / IWD / IAU / IEF 포함된 가격 데이터
    unrate : pd.Series
        실업률 데이터 (생략 시 자동 로드)
    verbose : bool
        True 시 디버깅 정보 출력

    Returns
    -------
    Dict[str, float] : 이 시점의 포트폴리오 비중
    """
    required = ["QQQ", "IWD", "IAU", "IEF", "SGOV", "SPY"]
    missing = [t for t in required if t not in prices.columns]
    if missing:
        raise ValueError(f"LAA(GT) 전략에 필요한 ETF 데이터가 없습니다: {missing}")

    prices = prices.sort_index()

    # -------------------------
    # 실업률 데이터 로드 & recession 판단
    # -------------------------
    if unrate is None:
        unrate = load_unemployment_rate()

    unrate = unrate.dropna()
    if len(unrate) < 13:
        raise ValueError("UNRATE 데이터가 부족합니다 (12M SMA 필요).")

    recession = _is_recession(unrate)

    # -------------------------
    # SPY 추세 판단
    # -------------------------
    spy = prices["SPY"]
    uptrend = _is_market_uptrend(spy)

    # -------------------------
    # GT 룰로 Growth 자산 선택
    # -------------------------
    # (불경기 AND 하락장) → SGOV
    # (나머지 모든 경우) → QQQ
    if recession and (not uptrend):
        growth_asset = "SGOV"
    else:
        growth_asset = "QQQ"

    # -------------------------
    # 디버깅 출력
    # -------------------------
    if verbose:
        last_unemp_date = unrate.index[-1]
        last_unemp_value = float(unrate.iloc[-1])
        sma12 = float(unrate.tail(12).mean())

        spy_last_date = spy.dropna().index[-1]
        spy_last_price = float(spy.dropna().iloc[-1])
        spy_ma200 = float(spy.dropna().rolling(200).mean().iloc[-1])

        print("=== LAA(GT) Signal Info ===")
        print(f"UNRATE latest date  : {last_unemp_date.date()}")
        print(f"UNRATE last value   : {last_unemp_value:.2f}%")
        print(f"UNRATE 12M SMA      : {sma12:.2f}%")
        print(f"Recession?          : {recession}")
        print(f"SPY last date       : {spy_last_date.date()}")
        print(f"SPY last price      : {spy_last_price:.2f}")
        print(f"SPY 200D MA         : {spy_ma200:.2f}")
        print(f"Uptrend?            : {uptrend}")
        print(f"Growth Asset Chosen : {growth_asset}")
        print("----------------------------------------")

    # -------------------------
    # 최종 포트 구성 (항상 4자산 × 25%)
    # -------------------------
    weights = {
        growth_asset: 0.25,
        "IWD": 0.25,
        "IAU": 0.25,
        "IEF": 0.25,
    }
    return weights


# ------------------------------------------------------------
# 4) 월말 weight 시계열 생성 (백테스트용)
# ------------------------------------------------------------
def _laa_weights_timeseries(prices: pd.DataFrame) -> pd.DataFrame:
    """
    LAA(GT) 전략의 월말 리밸런싱 weight 시계열 생성.

    - 각 월말 날짜 기준으로 laa_signal()을 호출
    - 해당 시점까지의 데이터만 사용 (룩어헤드 방지)
    - 초기 구간(UNRATE < 12M or SPY < 200일)은 자동 Skip
    """
    prices = prices.sort_index()
    cols = list(prices.columns)

    # 월말 인덱스 뽑기
    monthly_idx = prices.resample("ME").last().index

    # 실업률 전체 시계열
    unrate_full = load_unemployment_rate().dropna()

    weight_rows = []
    valid_dates = []

    for dt in monthly_idx:
        # 월말 시점까지의 정보만 사용
        price_sub = prices.loc[:dt]
        unrate_sub = unrate_full[unrate_full.index <= dt]

        try:
            w_dict = laa_signal(price_sub, unrate=unrate_sub, verbose=False)
        except ValueError:
            # 데이터 부족한 초기 구간은 Skip
            continue

        # weight row 구성
        row = {c: 0.0 for c in cols}
        for t, w in w_dict.items():
            if t in row:
                row[t] = w

        weight_rows.append(row)
        valid_dates.append(dt)

    if not weight_rows:
        raise ValueError("LAA(GT) weight 시계열 생성 실패. 데이터 범위를 확인하십시오.")

    weight_df = pd.DataFrame(weight_rows, index=pd.to_datetime(valid_dates))
    weight_df = weight_df.reindex(columns=cols)

    return weight_df


# ------------------------------------------------------------
# 5) 백테스트 엔진용 인터페이스
# ------------------------------------------------------------
def get_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """
    백테스트 엔진이 호출하는 표준 인터페이스.
    """
    return _laa_weights_timeseries(prices)


def debug_laa_states(prices: pd.DataFrame) -> pd.DataFrame:
    """
    디버그용:
    매 월말마다 (recession, uptrend, growth_asset)을 기록한 테이블 생성.
    """
    prices = prices.sort_index()
    unrate_full = load_unemployment_rate().dropna()

    monthly_idx = prices.resample("M").last().index

    rows = []
    idxs = []

    for dt in monthly_idx:
        price_sub = prices.loc[:dt]
        unrate_sub = unrate_full[unrate_full.index <= dt]

        try:
            # recession / uptrend 계산
            unrate_sub = unrate_sub.dropna()
            if len(unrate_sub) < 13:
                continue

            spy = price_sub["SPY"].dropna()
            if len(spy) < 200:
                continue

            rec = _is_recession(unrate_sub)
            up = _is_market_uptrend(spy)
            if rec and (not up):
                g = "SGOV"
            else:
                g = "QQQ"
        except Exception:
            continue

        rows.append(
            {
                "recession": rec,
                "uptrend": up,
                "growth_asset": g,
            }
        )
        idxs.append(dt)

    if not rows:
        return pd.DataFrame()

    state_df = pd.DataFrame(rows, index=pd.DatetimeIndex(idxs))
    return state_df
