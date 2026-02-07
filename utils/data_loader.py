# utils/data_loader.py

import yfinance as yf
import pandas as pd
import numpy as np

###############################################################################
# 가격 데이터 로더 (개선 버전)
###############################################################################

"""
전략에서는 항상 논리 티커 (QQQ, SGOV, IAU 등)만 사용하고,
실제 Yahoo ticker 매핑은 이 파일에서만 관리한다.

QQQ  -> ^NDX (proxy)
IAU  -> GC=F (Gold futures)
SGOV -> synthetic cash (3M T-Bill ^IRX 기반)
"""

REAL_TICKERS = {
    "QQQ": "^NDX",
    "SPY": "^GSPC",
    "IAU": "GC=F",
    "SGOV": None,
    "IEF": "IEF",
    "IWD": "IWD",
    # 나머지는 그대로 사용
}

CASH_PROXY_TICKER = "^IRX"


###############################################################################
# IRX 캐시 (속도 개선 핵심)
###############################################################################

_IRX_CACHE = None


def _load_irx_rate(start="1995-01-01") -> pd.Series:
    """^IRX 금리를 한 번만 다운로드해서 캐시에 저장."""
    global _IRX_CACHE

    if _IRX_CACHE is not None:
        return _IRX_CACHE

    raw = yf.download(CASH_PROXY_TICKER, start=start, auto_adjust=False)
    if raw.empty:
        raise ValueError("IRX 금리 데이터를 다운로드하지 못했습니다.")

    if "Adj Close" in raw.columns:
        rate = raw["Adj Close"]
    else:
        rate = raw.iloc[:, 0]

    if isinstance(rate, pd.DataFrame):
        rate = rate.iloc[:, 0]

    rate = rate.sort_index()
    _IRX_CACHE = rate

    return rate


###############################################################################
# SGOV synthetic cash
###############################################################################

def _build_sgov_series(index: pd.DatetimeIndex, start="1995-01-01") -> pd.Series:
    """IRX 금리 기반 synthetic SGOV price series 생성."""
    rate = _load_irx_rate(start=start)

    # 사용 날짜에 맞춰 정렬
    rate = rate.reindex(index).ffill().fillna(0.0)

    r = rate.astype(float)
    r = np.where(r > 50.0, r / 10000.0, r / 100.0)   # basis point & percent 방어

    daily_ret = r / 252.0
    daily_ret = pd.Series(daily_ret, index=index)

    price = (1.0 + daily_ret).cumprod()
    return price.rename("SGOV")


###############################################################################
# 가격 다운로드
###############################################################################

def load_prices(tickers, start: str = "1995-01-01") -> pd.DataFrame:
    """
    tickers 리스트를 받아서 Yahoo Finance 가격을 가져오고,
    SGOV 등은 synthetic cash로 처리한 뒤
    논리 티커 이름(예: "QQQ")을 컬럼명으로 유지한다.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    download_map = []   # (logical_name, real_ticker)
    cash_assets = []    # SGOV 등

    # 1) 티커 매핑 정리
    for t in tickers:
        real = REAL_TICKERS.get(t, t)
        if real is None:                     # SGOV
            cash_assets.append(t)
        else:
            download_map.append((t, real))

    real_list = [real for (_, real) in download_map]

    # 2) 가격 다운로드
    if real_list:
        raw = yf.download(real_list, start=start, auto_adjust=False)

        # 단일/복수 티커 구분
        if "Adj Close" in raw.columns:
            raw_close = raw["Adj Close"]
        else:
            raw_close = raw.iloc[:, 0]
    else:
        raw_close = pd.DataFrame()

    # 3) 단일/복수 케이스 처리
    if isinstance(raw_close, pd.Series):
        # Series → 단일 ticker
        orig = download_map[0][0]   # logical name
        data = raw_close.to_frame(name=orig)

    elif isinstance(raw_close, pd.DataFrame) and not raw_close.empty:
        # 여러 ticker
        mapping = {real: orig for (orig, real) in download_map}
        data = raw_close.rename(columns=mapping)

    else:
        data = pd.DataFrame()

    data = data.sort_index()

    # 4) SGOV synthetic 추가
    for c in cash_assets:
        if data.empty:
            # cash만 있을 경우 → IRX 기반 index 생성
            idx = _load_irx_rate(start=start).index
        else:
            idx = data.index

        sgov = _build_sgov_series(idx, start=start)

        if data.empty:
            data = pd.DataFrame(index=sgov.index)

        data[c] = sgov.reindex(data.index).ffill()

    # 5) 전체 정리
    data = data.dropna(how="all")

    return data

################################################################################
# MA 전략 전용 원시 Close 로더
################################################################################

def load_close_for_ma(ticker: str, start: str = "1995-01-01") -> pd.Series:
    """
    MA 전략에서만 사용할 '원시 Close' 시계열 로더.

    - 논리 티커(예: "QQQ")를 넣으면 REAL_TICKERS 매핑을 거쳐 실제 Yahoo 티커에서 Close를 가져옴.
    - Adj Close 기준 백테스트(price_df)와는 분리해서, MA 신호 전용으로만 사용.
    """
    # 논리 티커 -> 실제 티커 매핑 (QQQ -> ^IXIC 등)
    real = REAL_TICKERS.get(ticker, ticker)

    raw = yf.download(real, start=start, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"{real} 가격 데이터를 다운로드하지 못했습니다.")

    if "Close" in raw.columns:
        s = raw["Close"]
    elif "Adj Close" in raw.columns:
        s = raw["Adj Close"]
    else:
        s = raw.iloc[:, 0]

    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    s = s.sort_index().astype(float)
    s.name = ticker  # 나중에 디버깅 편하게

    return s
