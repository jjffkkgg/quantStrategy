# strategies/laa_dm.py

from typing import Dict
import pandas as pd

from strategies.adjDualMomentum import dual_momentum_signal


def laa_dm_signal(
    prices: pd.DataFrame,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    듀얼모멘텀 기반 LAA 전략 (LAA+DM)

    - 포트 구성은 기존 LAA와 동일:
        리스크온  → QQQ / IEF / IWD / IAU (각 25%)
        리스크오프 → SGOV / IEF / IWD / IAU (각 25%)

    - 리스크온/오프 판단은 듀얼모멘텀:
        DM 시그널이 {"SPY":1} 또는 {"EFA":1} → 리스크온
        그외(안전자산 믹스) → 리스크오프
    """
    required = ["QQQ", "IEF", "IWD", "IAU", "SGOV"]
    missing = [t for t in required if t not in prices.columns]
    if missing:
        raise ValueError(f"LAA+DM 전략에 필요한 ETF 데이터가 없습니다: {missing}")

    dm_sig = dual_momentum_signal(prices)

    # DM 공격모드 판정
    risk_on = False
    if isinstance(dm_sig, dict) and len(dm_sig) == 1:
        (tkr, w), = dm_sig.items()
        if tkr in ("SPY", "EFA") and abs(w - 1.0) < 1e-8:
            risk_on = True

    if verbose:
        print("=== LAA+DM Decision ===")
        print(f"DM raw signal : {dm_sig}")
        print(f"Risk ON?      : {risk_on}")
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


# ================= 백테스트용 weight 시계열 =================

def _laa_dm_weights_timeseries(prices: pd.DataFrame) -> pd.DataFrame:
    """
    듀얼모멘텀 기반 LAA 전략 월말 weight 테이블 생성.
    """
    prices = prices.sort_index()
    cols = list(prices.columns)

    monthly_idx = prices.resample("M").last().index

    rows = []
    idxs = []

    from strategies.laaDm import laa_dm_signal

    for dt in monthly_idx:
        price_sub = prices.loc[:dt]

        weights = laa_dm_signal(price_sub, verbose=False)

        row = {c: 0.0 for c in cols}
        for t, w in weights.items():
            if t in row:
                row[t] = w

        rows.append(row)
        idxs.append(dt)

    wdf = pd.DataFrame(rows, index=pd.to_datetime(idxs))
    return wdf.reindex(columns=cols)


def get_weights(prices: pd.DataFrame) -> pd.DataFrame:
    return _laa_dm_weights_timeseries(prices)
