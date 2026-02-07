# utils/backtest.py

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

TRADING_DAYS = 252  # 연 환산용


@dataclass
class BacktestResult:
    equity_curve: pd.Series          # 포트 가치 시계열 (명목 기준)
    daily_returns: pd.Series         # 포트 일간 수익률
    cagr: float                      # 연환산 수익률 (명목, Nominal)
    mdd: float                       # 최대 낙폭 (음수)
    sharpe: float                    # 샤프지수 (Rf=0 가정)
    trade_log: pd.DataFrame          # 매매 내역 (포지션 변경 시점)
    real_cagr: Optional[float] = None  # CPI 기준 실질 연환산 수익률 (Real CAGR)


# ----------------------------------------------------------------------
# 기본 지표 계산 함수들
# ----------------------------------------------------------------------

def _calc_cagr(equity: pd.Series) -> float:
    """연환산 수익률(CAGR) 계산 (명목 기준)."""
    if equity.empty:
        return np.nan

    start_value = float(equity.iloc[0])
    end_value = float(equity.iloc[-1])
    if start_value <= 0:
        return np.nan

    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return np.nan

    years = days / 365.25
    return (end_value / start_value) ** (1.0 / years) - 1.0


def _calc_mdd(equity: pd.Series) -> float:
    """
    최대 낙폭(MDD) 계산. 결과는 음수값 (ex: -0.25 == -25%).
    """
    if equity.empty:
        return np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _calc_sharpe(daily_ret: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """
    샤프지수 계산 (Rf=0).
    """
    if daily_ret.empty or daily_ret.std() == 0:
        return np.nan

    return float(daily_ret.mean() / daily_ret.std() * np.sqrt(trading_days))


# ----------------------------------------------------------------------
# CPI(인플레이션) 로딩 & Real CAGR 계산
# ----------------------------------------------------------------------

def _load_cpi_series(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """
    FRED에서 CPIAUCSL (미국 CPI, 1982-84=100) 시리즈를 받아온 뒤,
    [start, end] 구간으로 잘라서 반환.

    - 월별 데이터이므로 이후 일별로 reindex & ffill 해서 사용.
    - FRED CSV 직접 읽기 (pandas_datareader 안 씀).
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"

    # DATE, CPIAUCSL 컬럼 포함된 CSV
    df = pd.read_csv(url, parse_dates=["DATE"])
    df = df.rename(columns={"DATE": "date", "CPIAUCSL": "cpi"})
    df = df.set_index("date").sort_index()

    # 필요한 구간으로 슬라이싱
    df = df.loc[(df.index >= start) & (df.index <= end)]
    cpi = df["cpi"].astype(float)

    return cpi


def _calc_real_cagr(equity: pd.Series) -> float:
    """
    CPI 기반 Real CAGR 계산.

    순서:
      1) equity.index 기간에 맞는 CPI 시리즈 로딩
      2) CPI를 일별로 reindex + ffill
      3) 실질 포트 가치 = equity / (CPI / 초기 CPI)
      4) 그 실질 포트 가치에 대해 _calc_cagr() 재사용
    """
    if equity.empty:
        return np.nan

    start = equity.index[0]
    end = equity.index[-1]

    if not isinstance(start, pd.Timestamp):
        start = pd.to_datetime(start)
    if not isinstance(end, pd.Timestamp):
        end = pd.to_datetime(end)

    try:
        cpi = _load_cpi_series(start, end)
    except Exception:
        # CPI 다운로드 실패 시 Real CAGR 계산 불가 → NaN
        return np.nan

    if cpi.empty:
        return np.nan

    # equity index(일별)로 CPI 맞추고 ffill
    cpi = cpi.reindex(equity.index).ffill()

    # 기준 CPI (처음 값)
    base_cpi = float(cpi.iloc[0])
    if base_cpi <= 0:
        return np.nan

    # 실질 포트 가치 시계열
    real_equity = equity / (cpi / base_cpi)

    return _calc_cagr(real_equity)


# ----------------------------------------------------------------------
# 백테스트 본체
# ----------------------------------------------------------------------

def run_backtest(
    price_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    shift_weight: bool = True,
) -> BacktestResult:
    """
    단일 전략 백테스트.

    Parameters
    ----------
    price_df : pd.DataFrame
        자산별 가격 시계열 (columns = 티커, index = DatetimeIndex, Adj Close 기준)
    weight_df : pd.DataFrame
        동일한 티커 컬럼을 가진 포트 비중 시계열.
        보통 월말/월초/일별 리밸런싱 weight.
    initial_capital : float
        초기 자본.
    shift_weight : bool
        True면 weight를 하루 뒤로 shift 해서 룩어헤드 방지.
        (오늘 결정한 weight를 내일 수익률에 적용)
    """

    # 1) 가격 -> 일간 수익률
    price_df = price_df.sort_index()

    # FutureWarning 방지를 위해 fill_method=None 명시
    daily_ret_assets = price_df.pct_change(fill_method=None).fillna(0.0)

    # 2) weight를 가격 인덱스로 맞추고, 리밸런싱 구간 동안 forward fill
    weight_df = weight_df.sort_index()
    aligned_weights = (
        weight_df.reindex(price_df.index)   # 같은 인덱스로 확장
                .ffill()                    # 마지막 weight 유지
                .fillna(0.0)
    )

    # 3) 룩어헤드 방지: weight를 하루 뒤로 밀기
    if shift_weight:
        aligned_weights = aligned_weights.shift(1).fillna(0.0)

    # ------------------------------------------------------------------
    # 매매 히스토리(trade_log) 생성 (티커별 BUY/SELL 로그)
    # ------------------------------------------------------------------
    trade_rows = []
    if not aligned_weights.empty:
        prev_w = aligned_weights.iloc[0]
        dates = aligned_weights.index

        for i in range(1, len(aligned_weights)):
            curr_w = aligned_weights.iloc[i]

            # 각 티커별 weight 차이
            diff = curr_w - prev_w
            changed = diff[diff.abs() > 1e-9]  # 사실상 0 아닌 것만

            if not changed.empty:
                for ticker, d in changed.items():
                    old = float(prev_w[ticker])
                    new = float(curr_w[ticker])
                    delta = float(d)

                    action = "BUY" if delta > 0 else "SELL"

                    trade_rows.append(
                        {
                            "date": dates[i],
                            "ticker": ticker,
                            "old_w": old,
                            "new_w": new,
                            "delta": delta,
                            "action": action,
                        }
                    )

                prev_w = curr_w

    if trade_rows:
        trade_log_df = pd.DataFrame(trade_rows).set_index("date")
    else:
        trade_log_df = pd.DataFrame(
            columns=["ticker", "old_w", "new_w", "delta", "action"]
        )
        trade_log_df.index.name = "date"

    # 4) 포트 일간 수익률
    daily_port_ret = (aligned_weights * daily_ret_assets).sum(axis=1)

    # 5) 포트 가치 시계열 (명목 equity)
    equity_curve = (1.0 + daily_port_ret).cumprod() * initial_capital

    # 6) 성과 지표 계산
    cagr = _calc_cagr(equity_curve)          # Nominal CAGR
    mdd = _calc_mdd(equity_curve)
    sharpe = _calc_sharpe(daily_port_ret)

    # 7) Real CAGR (CPI 기준 인플레 차감)
    real_cagr = _calc_real_cagr(equity_curve)

    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_port_ret,
        cagr=cagr,
        mdd=mdd,
        sharpe=sharpe,
        trade_log=trade_log_df,
        real_cagr=real_cagr,
    )


# ----------------------------------------------------------------------
# 여러 전략 비교용 유틸
# ----------------------------------------------------------------------

def compare_strategies(
    price_df: pd.DataFrame,
    weight_dict: Dict[str, pd.DataFrame],
    initial_capital: float = 1_000_000.0,
    shift_weight: bool = True,
) -> (pd.DataFrame, Dict[str, BacktestResult]):
    """
    여러 전략을 한 번에 백테스트 해서 성과지표 비교.

    Parameters
    ----------
    price_df : pd.DataFrame
        공통 가격 시계열
    weight_dict : Dict[str, pd.DataFrame]
        {전략이름: weight_df}
    """
    summary_rows = []
    result_dict: Dict[str, BacktestResult] = {}

    for name, wdf in weight_dict.items():
        res = run_backtest(
            price_df=price_df,
            weight_df=wdf,
            initial_capital=initial_capital,
            shift_weight=shift_weight,
        )
        result_dict[name] = res
        summary_rows.append(
            {
                "Strategy": name,
                "CAGR": res.cagr,
                "RealCAGR": res.real_cagr,
                "MDD": res.mdd,
                "Sharpe": res.sharpe,
            }
        )

    summary_df = pd.DataFrame(summary_rows).set_index("Strategy")

    return summary_df, result_dict
