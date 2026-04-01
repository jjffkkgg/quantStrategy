# runBacktest.py
"""
전략별 백테스트 실행 스크립트.

사용법 (터미널에서):

    python runBacktest.py LAA
    python runBacktest.py SP500_MA
    python runBacktest.py DM
    python runBacktest.py LAA_DM
    python runBacktest.py LAA_MA

- LAA     : 실업률 기반 LAA
- SP500_MA: S&P500 5/10/20/60일 MA 정배열/역배열 전략
- DM      : 듀얼모멘텀 (SPY/EFA + 안전자산)
- LAA_DM  : 듀얼모멘텀을 리스크온/오프 스위치로 사용하는 LAA 변형
- LAA_MA  : 실업률 대신 QQQ MA 로 리스크온/오프를 결정하는 LAA 변형
- LAA_MA2 : QQQ는 MA로, IWD는 경기/추세로 리스크온/오프를 결정하는 LAA 변형
- MA2     : QQQ, IWD 각자의 MA 전략으로 ON/OFF 하는 전략
- LAA_MA2F : QQQ + IWD MA 로 리스크온/오프를 결정하는 LAA 변형 (재진입 쿨다운 30일 적용)
- LAA_MA3 : LAA_MA2F에서 IWD 매도/재진입 로직을 매월에서 매일/3일연속으로 체크하는 버전
- LAA_MA4 : LAA_MA2F에서 IAU 매매 전략을 추가한 버전 (1년 수익률 기반)
- DM_RP   : 듀얼모멘텀 + 리스크패리티 오버레이 전략
"""

import os
import sys
from typing import Dict, List

import pandas as pd
import numpy as np

from utils.backtest import run_backtest
from utils.data_loader import load_prices
from utils.data_loader import load_close_for_ma
from utils.macro_data import load_unemployment_rate
from strategies.customMA import _ma_alignment_weights
from strategies.laa import laa_signal
from strategies.laa import get_weights as laa_get_weights
from strategies.laa2 import get_weights as laa2_get_weights
from strategies.laaDm import get_weights as laa_dm_get_weights
from strategies.laaMA import get_weights as laa_ma_get_weights
from strategies.laaMA2 import get_weights as laa_ma2_get_weights
from strategies.ma2 import get_weights as ma2_get_weights
from strategies.dm_rp import get_weights as dm_rp_get_weights
from strategies.laaMA3 import get_weights as laa_ma3_get_weights
from strategies.laaMA2F import get_weights as laa_ma2f_get_weights
from strategies.laaMA4 import get_weights as laa_ma4_get_weights


# ----------------------------------------------------------------------
# 1. 전략별로 필요한 티커들 정의
# ----------------------------------------------------------------------


def get_tickers_for_strategy(strategy_name: str) -> List[str]:
    """
    각 전략에서 사용하는 티커 리스트 정의.
    """
    name = strategy_name.upper()

    if name == "LAA":
        # 실업률 기반 LAA에서 쓰는 ETF들
        return ["QQQ", "IWD", "IAU", "IEF", "SGOV", "SPY"]
    
    if name == "LAA2":
        # LAA2: 기본 LAA + QQQ/IWD 리스크 블록 동시 온오프
        return ["QQQ", "IWD", "IAU", "IEF", "SGOV", "SPY"]

    if name in ("SP500_MA", "SP500", "SP500MA"):
        # S&P500 지수 + 캐시(SGOV)
        return ["^GSPC", "SGOV"]

    if name == "DM":
        # 듀얼모멘텀 (공격: SPY/EFA, 방어: 안전자산 + 현금)
        return ["SPY", "EFA", "SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "BWX", "EMB", "SGOV"]

    if name == "LAA_DM":
        # LAA + 듀얼모멘텀 (리스크판단용 DM 티커 포함)
        return ["QQQ", "IEF", "IWD", "IAU", "SGOV",
                "SPY", "EFA", "SHY", "TLT", "TIP", "LQD", "HYG", "BWX", "EMB"]

    if name == "LAA_MA":
        # LAA_MA: LAA 와 동일한 ETF 세트
        return ["QQQ", "IEF", "IWD", "IAU", "SGOV"]
    
    if name == "MA2":
        return ["QQQ", "IWD", "IAU", "IEF", "SGOV"]
    
    if name == "LAA_MA2":          # 🔹 새로 추가
        return ["QQQ", "IEF", "IWD", "IAU", "SGOV", "SPY"]
    
    if name == "LAA_MA2F":
        return ["QQQ", "IEF", "IWD", "IAU", "SGOV", "SPY"]
    
    if name == "LAA_MA3":
        return ["QQQ", "IEF", "IWD", "IAU", "SGOV", "SPY"]
    
    if name == "LAA_MA4":
        return ["QQQ", "IEF", "IWD", "IAU", "SGOV", "SPY"]
    
    if name == "DM_RP":
        # Dual Momentum + Risk Parity 전략에서 사용할 자산들
        return [
            "SPY", "QQQ", "IWD", "IWM", "EFA",   # 위험자산
            "SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "BWX", "EMB",  # 방어 ETF
            "SGOV",                             # 캐시 대체
        ]


    raise ValueError(f"지원하지 않는 전략 이름입니다: {strategy_name}")


# ----------------------------------------------------------------------
# 3. S&P500 MA: 일별 weight 시계열
# ----------------------------------------------------------------------


def build_sp500_ma_weights_timeseries(
    price_df: pd.DataFrame,
    sp500_ticker: str = "^GSPC",
    cash_ticker: str = "SGOV",
) -> pd.DataFrame:
    """
    S&P500 MA 전략 (customMA._ma_alignment_weights 사용):

    - sp500_ticker 에 대해 5/10/20/60 MA 정배열/역배열을 기준으로
      0~1 weight 시계열 생성.
    - 남는 비중은 cash_ticker (SGOV)에 배분 (1 - 주식 weight).
    """
    price_df = price_df.sort_index()
    cols = list(price_df.columns)

    if sp500_ticker not in price_df.columns:
        raise ValueError(f"prices에 '{sp500_ticker}' 컬럼이 없습니다.")

    # 1) MA 계산용 Close 시리즈 로드
    start_str = price_df.index[0].strftime("%Y-%m-%d")
    close_series = load_close_for_ma(sp500_ticker, start=start_str)

    # 2) 백테스트에서 쓰는 Adj Close 인덱스에 맞춰 정렬
    close_series = close_series.reindex(price_df.index).ffill()

    # 3) Close 기반으로 MA weight 계산
    w_stock = _ma_alignment_weights(close_series)

    # 전체 price_df 인덱스로 맞추고, NA는 직전값(또는 0)으로 처리
    w_stock = w_stock.reindex(price_df.index).ffill().fillna(0.0)

    # weight DataFrame 초기화
    weight_df = pd.DataFrame(0.0, index=price_df.index, columns=cols)

    # 주식 비중
    weight_df[sp500_ticker] = w_stock

    # 캐시 비중 (있을 경우에만)
    if cash_ticker in weight_df.columns:
        weight_df[cash_ticker] = 1.0 - w_stock

    return weight_df


# ----------------------------------------------------------------------
# 4. DM(듀얼모멘텀): 월별 weight 시계열 (내부 구현)
# ----------------------------------------------------------------------


def _dm_choose_cash_ticker(price_df: pd.DataFrame) -> str:
    """
    듀얼모멘텀에서 쓸 현금 대체 자산 선택:
    - SGOV 있으면 SGOV
    - 없으면 SHY
    - 둘 다 없으면 'CASH' (수익률 0으로 처리)
    """
    cols = price_df.columns
    if "SGOV" in cols:
        return "SGOV"
    if "SHY" in cols:
        return "SHY"
    return "CASH"


def _dm_defensive_weights(price_sub: pd.DataFrame) -> Dict[str, float]:
    """
    듀얼모멘텀 방어 모드 안전자산 배분:

    1) 안전자산 universe:
       SHY, IEF, TLT, TIP, LQD, HYG, BWX, EMB (존재하는 것만 사용)
    2) 최근 6개월(≈126 거래일) 수익률 기준 상위 3개 ETF 선택
    3) 각 1/3 비중
    4) 단, 선택된 ETF의 6개월 수익률이 0 이하이면 그 1/3은 투자 안 하고 현금 전환
       → 현금 비중은 0, 1/3, 2/3, 1.0
    """
    safe_all = ["SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "BWX", "EMB"]
    safe_universe = [t for t in safe_all if t in price_sub.columns]

    cash_ticker = _dm_choose_cash_ticker(price_sub)

    if len(safe_universe) == 0 or len(price_sub) < 30:
        return {cash_ticker: 1.0}

    lookback = 126  # ~ 6개월
    if len(price_sub) <= lookback:
        return {cash_ticker: 1.0}

    # FutureWarning 방지를 위해 fill_method=None 명시
    rets_6m = price_sub[safe_universe].pct_change(lookback, fill_method=None).iloc[-1]
    rets_6m = rets_6m.dropna()
    if rets_6m.empty:
        return {cash_ticker: 1.0}

    top = rets_6m.sort_values(ascending=False).head(3)
    selected = list(top.index)
    n_sel = len(selected)
    if n_sel == 0:
        return {cash_ticker: 1.0}

    base_w = 1.0 / n_sel
    weights: Dict[str, float] = {}
    cash_w = 0.0

    for t in selected:
        r = top[t]
        if r > 0:
            weights[t] = base_w
        else:
            cash_w += base_w

    if cash_w > 0:
        weights[cash_ticker] = weights.get(cash_ticker, 0.0) + cash_w

    # 혹시 전부 0 이하라서 다 현금으로 간 경우
    if all(w <= 0 for w in weights.values()):
        return {cash_ticker: 1.0}

    return weights


def _dm_signal(price_sub: pd.DataFrame) -> Dict[str, float]:
    """
    듀얼모멘텀 시그널 (공격/방어 모드 결정):

    - 공격 모드(absolute momentum OK):
        * 자산군: SPY vs EFA
        * 12개월 수익률이 더 큰 쪽에 100% 투자 → {"SPY":1.0} 또는 {"EFA":1.0}
        * 단, SPY/EFA 모두 12개월 수익률 < 0 이면 방어 모드로 진입

    - 방어 모드(absolute momentum FAIL):
        * _dm_defensive_weights() 규칙 적용
    """
    cols = price_sub.columns
    if "SPY" not in cols or "EFA" not in cols:
        # 데이터가 없으면 방어 모드
        return _dm_defensive_weights(price_sub)

    lookback_12m = 252
    if len(price_sub) <= lookback_12m:
        return _dm_defensive_weights(price_sub)

    mom_12m = price_sub[["SPY", "EFA"]].pct_change(lookback_12m, fill_method=None).iloc[-1]

    # 절대 모멘텀: 둘 다 < 0 이면 방어 모드
    if (mom_12m < 0).all():
        return _dm_defensive_weights(price_sub)

    # 상대 모멘텀: 더 좋은 쪽 100%
    best = mom_12m.idxmax()
    return {best: 1.0}


def build_dm_weights_timeseries(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    듀얼모멘텀(DM)의 월별 weight 시계열 생성.
    """
    price_df = price_df.sort_index()
    cols = list(price_df.columns)

    monthly_idx = price_df.resample("ME").last().index

    weights_list: List[Dict[str, float]] = []
    idx_list: List[pd.Timestamp] = []

    for dt in monthly_idx:
        price_sub = price_df.loc[:dt]

        w_dict = _dm_signal(price_sub)

        row = {c: 0.0 for c in cols}
        for t, w in w_dict.items():
            if t == "CASH":
                continue
            if t in row:
                row[t] = w

        weights_list.append(row)
        idx_list.append(dt)

    if not weights_list:
        raise ValueError("DM weight 시계열이 비어 있습니다. 가격 데이터 기간을 확인하세요.")

    weight_df = pd.DataFrame(weights_list, index=pd.to_datetime(idx_list))
    weight_df = weight_df.reindex(columns=cols)

    return weight_df


# ----------------------------------------------------------------------
# 5. LAA_DM / LAA_MA: 외부 전략 모듈을 통한 weight 생성
# ----------------------------------------------------------------------


def build_laa_dm_weights_timeseries(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    LAA_DM 전략:
    - 실제 리밸런싱 로직은 strategies.laa_dm.get_weights 에 있으며,
      여기서는 단순히 그 함수를 호출해서 weight_df 를 받아온다.
    """
    return laa_dm_get_weights(price_df)


def build_laa_ma_weights_timeseries(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    LAA_MA 전략:
    - 실제 리밸런싱 로직은 strategies.laa_ma.get_weights 에 있으며,
      여기서는 단순히 그 함수를 호출해서 weight_df 를 받아온다.
    """
    return laa_ma_get_weights(price_df)


# ----------------------------------------------------------------------
# 6. 전략 이름 -> weight_df 매핑
# ----------------------------------------------------------------------


def get_strategy_weights(strategy_name: str, price_df: pd.DataFrame) -> pd.DataFrame:

    name = strategy_name.upper()

    if name == "LAA":
        # 최신 laa.py 의 get_weights() 사용
        return laa_get_weights(price_df)
    
    if name == "LAA2":
        return laa2_get_weights(price_df)

    if name in ("SP500_MA", "SP500", "SP500MA"):
        return build_sp500_ma_weights_timeseries(price_df, sp500_ticker="^GSPC", cash_ticker="SGOV")

    if name == "DM":
        return build_dm_weights_timeseries(price_df)

    if name == "LAA_DM":
        return laa_dm_get_weights(price_df)

    if name == "LAA_MA":
        return laa_ma_get_weights(price_df)
    
    if name == "MA2":
        return ma2_get_weights(price_df)
    
    if name == "LAA_MA2":
        return laa_ma2_get_weights(price_df)
    
    if name == "LAA_MA2F":
        return laa_ma2f_get_weights(price_df)
    
    if name == "LAA_MA3":
        return laa_ma3_get_weights(price_df)
    
    if name == "LAA_MA4":
        return laa_ma4_get_weights(price_df)
    
    if name == "DM_RP":
        return dm_rp_get_weights(price_df)


    raise ValueError(f"지원하지 않는 전략입니다: {strategy_name}")



# ----------------------------------------------------------------------
# 7. 메인 실행
# ----------------------------------------------------------------------


def main():
    if len(sys.argv) < 2:
        print("사용법: python runBacktest.py <전략이름>")
        print("예시:  python runBacktest.py LAA")
        print("       python runBacktest.py SP500_MA")
        print("       python runBacktest.py DM")
        print("       python runBacktest.py LAA_DM")
        print("       python runBacktest.py LAA_MA")
        print("       python runBacktest.py MA2")
        print("       python runBacktest.py LAA_MA3")
        print("       python runBacktest.py LAA_MA4")
        sys.exit(1)
        
    strategy_name = sys.argv[1].upper()
    print(f"[INFO] 선택한 전략: {strategy_name}")

    # 1) 전략별 티커 리스트
    tickers = get_tickers_for_strategy(strategy_name)
    print(f"[INFO] 사용 티커: {tickers}")

    # 2) 가격 로딩
    # 백테스트 시작일 이전에 데이터가 없는 ETF들을 위한 역사적 프록시 정의
    # data_loader.py에서 이미 QQQ->^NDX, SPY->^GSPC, IAU->GC=F, SGOV->^IRX 매핑을 처리합니다.
    # 여기서는 IWD, IEF와 같이 직접 다운로드되지만 역사가 짧은 티커들을 대상으로 프록시를 추가합니다.
    historical_fill_proxies = {
        "IWD": "VWNDX", # iShares Russell 1000 Value ETF (시작: 2000년) -> Vanguard Developed Markets Index Fund Admiral Shares
        "IEF": "^TNX",  # iShares 7-10 Year Treasury Bond ETF (시작: 2002년) -> 10-Year Treasury Yield
    }

    # 전략 티커 목록에 프록시 티커 추가 (다운로드를 위해)
    tickers_to_download = list(tickers) # 원본 리스트 복사
    for target_ticker, proxy_ticker in historical_fill_proxies.items():
        if target_ticker in tickers_to_download and proxy_ticker not in tickers_to_download:
            tickers_to_download.append(proxy_ticker)
            print(f"[INFO] '{target_ticker}'의 역사적 데이터 보충을 위해 프록시 티커 '{proxy_ticker}'를 다운로드 목록에 추가합니다.")

    # 2) 가격 로딩 (확장된 티커 목록 사용)
    print("[INFO] 가격 데이터 로드 중...")
    price_df = load_prices(tickers=tickers_to_download, start="1985-01-01")

    # 다운로드된 프록시 티커로 원본 티커의 초기 누락 데이터 채우기
    for target_ticker, proxy_ticker in historical_fill_proxies.items():
        if target_ticker in price_df.columns and proxy_ticker in price_df.columns:
            # 타겟 티커의 첫 유효 데이터 인덱스 찾기
            first_valid_target_idx = price_df[target_ticker].first_valid_index()

            if first_valid_target_idx is None:
                print(f"[INFO] '{target_ticker}'에 유효한 데이터가 없어 프록시 채우기를 건너킵니다.")
                continue

            # 타겟 티커의 첫 유효 데이터 시점의 값
            target_value_at_start = price_df.loc[first_valid_target_idx, target_ticker]
            
            # 프록시 티커의 해당 시점 값
            proxy_value_at_start = price_df.loc[first_valid_target_idx, proxy_ticker]

            if pd.isna(target_value_at_start) or pd.isna(proxy_value_at_start) or proxy_value_at_start == 0:
                print(f"[WARNING] '{target_ticker}' 또는 '{proxy_ticker}'의 시작점 데이터가 유효하지 않아 프록시 채우기를 건너킵니다.")
                continue

            # 타겟 티커가 NaN이고 프록시 티커가 유효한 기간 마스크
            missing_period_mask = price_df[target_ticker].isna() & price_df[proxy_ticker].notna()
            
            if missing_period_mask.sum() == 0:
                print(f"[INFO] '{target_ticker}'에 채울 누락 데이터가 없거나 '{proxy_ticker}' 데이터가 부족합니다.")
                continue

            # 전체 프록시 시리즈
            proxy_series = price_df[proxy_ticker]

            synthetic_series_part = pd.Series(np.nan, index=price_df.index)

            if target_ticker == "IWD": # 가격-대-가격 프록시 (정비례 관계)
                scaling_factor = target_value_at_start / proxy_value_at_start
                synthetic_series_part.loc[missing_period_mask] = proxy_series.loc[missing_period_mask] * scaling_factor
                print(f"[INFO] '{target_ticker}' (IWD)의 초기 누락 데이터를 '{proxy_ticker}' (VWNDX)로 스케일링하여 채웁니다.")
            elif target_ticker == "IEF": # 가격-대-금리 프록시 (역비례 관계)
                # P_IEF_t = P_IEF_start * (Y_TNX_start / Y_TNX_t)
                # 프록시 시리즈(금리)에 0 값이 없는지 확인
                if (proxy_series.loc[missing_period_mask] == 0).any():
                    print(f"[WARNING] '{proxy_ticker}' (YIELD)에 0 값이 있어 '{target_ticker}' (IEF) 프록시 채우기를 건너킵니다.")
                    continue
                synthetic_series_part.loc[missing_period_mask] = target_value_at_start * (proxy_value_at_start / proxy_series.loc[missing_period_mask])
                print(f"[INFO] '{target_ticker}' (IEF)의 초기 누락 데이터를 '{proxy_ticker}' (^TNX)의 역비례 관계로 스케일링하여 채웁니다.")
            else:
                # 새로운 프록시 타입이 추가되었을 때를 위한 기본 combine_first 폴백
                synthetic_series_part.loc[missing_period_mask] = price_df.loc[missing_period_mask, proxy_ticker]
                print(f"[INFO] '{target_ticker}'의 초기 누락 데이터를 '{proxy_ticker}' 데이터로 채웁니다 (기본 combine_first).")

            original_nan_count = price_df[target_ticker].isna().sum()
            price_df[target_ticker] = price_df[target_ticker].combine_first(synthetic_series_part)
            filled_nan_count = price_df[target_ticker].isna().sum()
            
            if original_nan_count > filled_nan_count:
                print(f"[INFO] '{target_ticker}'의 초기 누락 데이터 {original_nan_count - filled_nan_count}개를 '{proxy_ticker}' 데이터로 채웠습니다.")
            elif filled_nan_count == original_nan_count:
                print(f"[INFO] '{target_ticker}'의 누락 데이터를 '{proxy_ticker}'로 채우지 못했습니다 (프록시 데이터도 누락되었거나 충분하지 않음).")
        else:
            print(f"[INFO] '{target_ticker}' 또는 '{proxy_ticker}' 컬럼이 price_df에 없어 프록시 채우기를 건너킵니다.")

    if not isinstance(price_df.index, pd.DatetimeIndex):
        raise ValueError("load_prices 결과의 index가 DatetimeIndex가 아닙니다.")

    price_df = price_df.sort_index()
    print(f"[INFO] 가격 기간: {price_df.index[0].date()} ~ {price_df.index[-1].date()}")

    # 3) 전략별 weight 시계열 생성
    print("[INFO] 전략 weight 시계열 생성 중...")
    weight_df = get_strategy_weights(strategy_name, price_df)

    # 최종 price_df에서 전략에 필요한 티커만 남기고, 추가 다운로드된 프록시 티커는 제거
    price_df = price_df[list(tickers)]

    print(f"[INFO] 리밸런싱/적용 기간: {weight_df.index[0].date()} ~ {weight_df.index[-1].date()}")

    # 4) 백테스트 실행
    print("[INFO] 백테스트 실행 중...")
    result = run_backtest(
        price_df=price_df,
        weight_df=weight_df,
        initial_capital=1_000_000.0,
        shift_weight=True,  # 룩어헤드 방지
    )

    print("\n=== 최근 20개 매매 내역 (Most Recent 20 Trades) ===")
    if result.trade_log.empty:
        print("매매 내역이 없습니다 (포지션 변화 없음).")
    else:
        print(result.trade_log.tail(20))

    os.makedirs("outputs", exist_ok=True)
    result.trade_log.to_csv(f"outputs/trades_{strategy_name}.csv")

    print("\n=== Backtest Result ===")
    print(f"Strategy : {strategy_name}")
    print(f"CAGR     : {result.cagr * 100:.2f}%")
    print(f"MDD      : {result.mdd * 100:.2f}%")
    print(f"Sharpe   : {result.sharpe:.2f}")

    # 5) equity curve 저장
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"equity_{strategy_name}.csv")
    result.equity_curve.to_csv(out_path)
    print(f"[INFO] equity curve 저장: {out_path}")

    # runBacktest.py (LAA 돌린 뒤)
    equity = result.equity_curve

    from utils.backtest import _calc_mdd

    print("Daily MDD :", _calc_mdd(equity))

    equity_m = equity.resample("ME").last()
    print("Monthly MDD:", _calc_mdd(equity_m))



if __name__ == "__main__":
    main()
