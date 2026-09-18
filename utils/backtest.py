# utils/backtest.py

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional
from utils.trading_calendar import completed_month_ends

TRADING_DAYS = 252  # 연 환산용


def defer_pre_history_allocations(
    price_df: pd.DataFrame, weight_df: pd.DataFrame, cash_ticker: str = 'SGOV'
) -> pd.DataFrame:
    """Park allocations in cash until an asset has an observed price.

    Only the leading unavailable history is handled. Later missing marks are
    left for the engine to validate. The original strategy targets are not
    mutated, and becoming available generates a normal delayed trade signal.
    """
    prices = price_df.sort_index()
    # Monthly strategies supply sparse targets. Expand before checking history
    # so an asset can become available between two signal dates.
    weights = weight_df.sort_index()
    weights = weights.reindex(weights.index.union(prices.index)).ffill()
    weights = weights.reindex(prices.index).fillna(0.0).copy()
    if cash_ticker not in prices or cash_ticker not in weights:
        raise ValueError(f'Pre-history allocation requires {cash_ticker}.')
    for ticker in weights.columns:
        if ticker == cash_ticker or not weights[ticker].gt(0).any():
            continue
        if ticker not in prices:
            raise ValueError(f'Missing price column: {ticker}')
        valid = prices[ticker].gt(0) & np.isfinite(prices[ticker])
        if not valid.any():
            raise ValueError(f'No valid price history for allocated asset {ticker}; check download.')
        deferred = ~valid.cummax() & weights[ticker].gt(0)
        if deferred.any():
            weights.loc[deferred, cash_ticker] += weights.loc[deferred, ticker]
            weights.loc[deferred, ticker] = 0.
            print(f'[INFO] {ticker}: {int(deferred.sum())} pre-history signal days '
                  f'allocated to {cash_ticker}; first valid price {prices.index[valid][0].date()}.')
    return weights


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
    rebalance: str = "monthly_and_changes",
    calculate_real_cagr: bool = True,
    missing_execution: str = 'raise',
    signal_only_asset: Optional[str] = None,
    signal_only_weight: float = 0.25,
    cash_ticker: str = 'SGOV',
) -> BacktestResult:
    """Self-financing holdings with zero costs and explicit close execution.

    With shift_weight=True, a signal at t close executes at t+1 close. Old
    holdings earn the entire t+1 return; new holdings first earn t+2 returns.
    False allows same-close execution (optimistic if signals use that close).
    Rebalance on target changes plus month-end decisions by default. Between
    events, holdings drift. `changes` and `daily` are also supported.
    Unallocated capital is zero-interest cash; SGOV earns its supplied return.
    """
    if signal_only_asset is not None:
        return _run_separate_sleeves(
            price_df, weight_df, initial_capital, shift_weight,
            calculate_real_cagr, missing_execution,
            signal_only_asset, signal_only_weight, cash_ticker)
    if rebalance not in {"monthly_and_changes", "monthly", "changes", "daily"}:
        raise ValueError("Unknown rebalance policy.")
    if missing_execution not in {'raise', 'defer'}:
        raise ValueError('Unknown missing execution policy.')
    price_df = price_df.sort_index().astype(float)
    weight_df = weight_df.sort_index()
    if price_df.empty or price_df.index.has_duplicates or weight_df.index.has_duplicates:
        raise ValueError("Prices must be nonempty and indexes must be unique.")
    if initial_capital <= 0 or not np.isfinite(initial_capital):
        raise ValueError("Initial capital must be positive and finite.")
    extra = weight_df.columns.difference(price_df.columns)
    if len(extra) and weight_df[extra].fillna(0).abs().to_numpy().any():
        raise ValueError("Weights contain assets without prices.")
    targets = weight_df.reindex(columns=price_df.columns, fill_value=0)
    targets = targets.reindex(targets.index.union(price_df.index)).ffill()
    targets = targets.reindex(price_df.index).fillna(0).astype(float)
    if (not np.isfinite(targets.to_numpy()).all() or (targets < -1e-12).any().any()
            or (targets.sum(axis=1) > 1 + 1e-9).any()):
        raise ValueError("Only finite, long-only, unleveraged targets are supported.")
    changed = targets.diff().abs().gt(1e-10).any(axis=1)
    changed.iloc[0] = targets.iloc[0].abs().sum() > 0
    decisions = changed.copy()
    if rebalance == 'monthly':
        decisions[:] = False
        decisions.iloc[0] = changed.iloc[0]
        decisions.loc[completed_month_ends(price_df.index)] = True
    elif rebalance == "monthly_and_changes":
        decisions.loc[completed_month_ends(price_df.index)] = True
    elif rebalance == "daily":
        decisions[:] = True
    events = targets.loc[decisions].reindex(price_df.index)
    if shift_weight:
        events = events.shift(1)
    marked = price_df.ffill()
    returns = marked.pct_change(fill_method=None)
    holdings = np.zeros(len(price_df.columns))
    cash = float(initial_capital)
    equity_values = []
    trade_rows = []
    pending_target = None
    deferred_days = 0
    for i, date in enumerate(price_df.index):
        r = returns.iloc[i].to_numpy()
        if i and np.any((holdings > 1e-9) & ~np.isfinite(r)):
            raise ValueError(f"Missing held-asset return at {date.date()}")
        holdings *= 1 + np.nan_to_num(r, nan=0.0)
        nav = float(holdings.sum() + cash)
        event = events.iloc[i]
        if event.notna().all():
            # The latest executable signal supersedes an older pending order.
            pending_target = event
        if pending_target is not None:
            target = pending_target
            desired = nav * target.to_numpy()
            delta = desired - holdings
            raw = price_df.iloc[i].to_numpy()
            traded = np.abs(delta) > max(nav * 1e-10, 1e-9)
            if np.any(traded & (~np.isfinite(raw) | (raw <= 0))):
                invalid = traded & (~np.isfinite(raw) | (raw <= 0))
                tickers = ', '.join(price_df.columns[invalid])
                if missing_execution == 'defer':
                    deferred_days += 1
                    if deferred_days <= 5:
                        print(f'[INFO] Rebalance deferred at {date.date()}: missing price for {tickers}.')
                    equity_values.append(nav)
                    continue
                raise ValueError(f"Cannot execute with missing/invalid price at {date.date()}: {tickers}")
            for j in np.flatnonzero(traded):
                trade_rows.append(dict(
                    date=date, ticker=price_df.columns[j],
                    old_w=holdings[j]/nav, new_w=desired[j]/nav,
                    delta=delta[j]/nav, amount=delta[j],
                    action="BUY" if delta[j] > 0 else "SELL",
                    execution="close"))
            holdings = desired
            cash = nav - float(holdings.sum())
            pending_target = None
        equity_values.append(nav)
    if deferred_days:
        print(f'[INFO] Rebalances deferred on {deferred_days} days due to unavailable execution prices.')
    if pending_target is not None:
        print('[INFO] Final rebalance remains pending; no fill was assumed.')
    equity_curve = pd.Series(equity_values, index=price_df.index)
    daily_port_ret = equity_curve.pct_change().fillna(0.0)
    trade_log_df = pd.DataFrame(trade_rows, columns=[
        "date", "ticker", "old_w", "new_w", "delta", "amount", "action", "execution"
    ]).set_index("date")

    # 6) 성과 지표 계산
    cagr = _calc_cagr(equity_curve)          # Nominal CAGR
    mdd = _calc_mdd(equity_curve)
    sharpe = _calc_sharpe(daily_port_ret)

    # 7) Real CAGR (CPI 기준 인플레 차감)
    real_cagr = _calc_real_cagr(equity_curve) if calculate_real_cagr else None

    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_port_ret,
        cagr=cagr,
        mdd=mdd,
        sharpe=sharpe,
        trade_log=trade_log_df,
        real_cagr=real_cagr,
    )


def _run_separate_sleeves(prices, weights, capital, shift, real_cagr,
                          missing_execution, asset, fraction, cash_ticker):
    """Independent signal-only asset/cash capital and monthly core capital.

    No capital transfers between sleeves, even while the signal sleeve is in
    cash. Thus a sale and later re-entry use the whole sleeve's current value,
    not a fresh 25% of total NAV. The core rebalances its own remaining capital.
    """
    if not 0 < fraction < 1 or asset == cash_ticker:
        raise ValueError('Invalid signal-only sleeve configuration.')
    if asset not in weights or cash_ticker not in weights:
        raise ValueError('Signal-only asset and cash must be present in weights.')
    allocation = weights[asset]
    if not (np.isclose(allocation, 0) | np.isclose(allocation, fraction)).all():
        raise ValueError('Signal-only sleeve requires binary on/off target weights.')
    signal_weights = pd.DataFrame({asset: allocation / fraction,
                                   cash_ticker: 1 - allocation / fraction}, index=weights.index)
    core_weights = weights.drop(columns=asset).copy()
    core_weights[cash_ticker] -= fraction - allocation
    if (core_weights[cash_ticker] < -1e-9).any():
        raise ValueError('Cash target does not cover the inactive signal sleeve.')
    core_weights[cash_ticker] = core_weights[cash_ticker].clip(lower=0)
    core_weights /= 1 - fraction
    common = dict(shift_weight=shift, calculate_real_cagr=False,
                  missing_execution=missing_execution)
    signal = run_backtest(prices[[asset, cash_ticker]], signal_weights,
                          initial_capital=capital * fraction, rebalance='changes', **common)
    core = run_backtest(prices.drop(columns=asset), core_weights,
                        initial_capital=capital * (1-fraction), rebalance='monthly', **common)
    equity = signal.equity_curve + core.equity_curve
    returns = equity.pct_change().fillna(0.)
    logs = []
    for label, result in [('signal', signal), ('monthly_core', core)]:
        log = result.trade_log.copy()
        scale = (result.equity_curve / equity).reindex(log.index).to_numpy()
        for col in ['old_w', 'new_w', 'delta']:
            log[col] *= scale
        log['sleeve'] = label
        logs.append(log)
    trades = pd.concat(logs).sort_index(kind='stable')
    return BacktestResult(equity, returns, _calc_cagr(equity), _calc_mdd(equity),
                          _calc_sharpe(returns), trades,
                          _calc_real_cagr(equity) if real_cagr else None)


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
