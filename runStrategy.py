# main.py
"""
현재 시점 시그널 체크용 스크립트.

- LAA      : 실업률 기반 LAA
- S&P500_MA: 5/10/20/60 MA 정배열/역배열
- DM       : 듀얼모멘텀 (adjDualMomentum 모듈)
- LAA_DM   : DM 을 리스크판단에 사용하는 LAA 변형
- LAA_MA   : QQQ MA 를 리스크판단에 사용하는 LAA 변형
"""

from config import START_DATE, TICKERS
from utils.data_loader import load_prices

from strategies.laa import laa_signal
from strategies.customMA import sp500_ma_signal
from strategies.adjDualMomentum import dual_momentum_signal
from strategies.laaDm import laa_dm_signal
from strategies.laaMA import laa_ma_signal
from strategies.laaMA2 import laa_ma2_signal
from strategies.dm_rp import dm_rp_signal

def print_weight_result(name: str, result):
    """
    전략 결과가 dict이면 포트 비중 형식으로, 아니면 그대로 출력.
    """
    if isinstance(result, dict):
        print(f"{name:<10} -> 포트폴리오 비중")
        for ticker, w in result.items():
            # ✅ ticker 를 무조건 문자열로 변환해서 출력
            print(f"   {str(ticker):5s} : {w*100:5.1f}%")
    else:
        print(f"{name:<10} -> {result}")



def main():

    # ✅ 각 전략에 필요한 티커들을 모두 합친 리스트
    tickers_for_signals = [
        # LAA / LAA_MA / LAA_MA2 공통
        "QQQ", "IWD", "IAU", "IEF", "SGOV", "SPY",
        # DM (듀얼모멘텀)
        "EFA", "SHY", "TLT", "TIP", "LQD", "HYG", "BWX", "EMB",
        # S&P500 MA
        "^GSPC",
        # DM_RP에 쓰일 수 있는 애들 (이미 위에 대부분 포함이지만 그냥 한 번 더)
        "IWM",
    ]
    # 중복 제거
    tickers_for_signals = sorted(set(tickers_for_signals))

    print("=== Quant Strategy Signal Checker ===")
    print(f"Start date: {START_DATE}")
    print(f"Tickers  : {', '.join(TICKERS)}")
    print("Downloading price data from Yahoo Finance...")

    prices = load_prices(TICKERS, start=START_DATE)

    alias_map = {
        "QQQ": "^NDX",   # QQQ 대신 ^NDX 로딩된 경우
        "SPY": "^GSPC",  # SPY 대신 ^GSPC 로딩된 경우
    }

    for etf, idx_ticker in alias_map.items():
        if etf not in prices.columns and idx_ticker in prices.columns:
            prices[etf] = prices[idx_ticker]
            print(f"[INFO] Alias column 생성: {idx_ticker} -> {etf}")

    last_date = prices.index[-1].date()
    print(f"\nLatest data date: {last_date}")

    # ---------- LAA (실업률 기반) ----------
    try:
        # verbose=True → 최신 실업률 날짜/값/12M 평균 출력
        laa = laa_signal(prices, verbose=True)
    except Exception as e:
        laa = f"Error: {e}"

    # ---------- S&P500 MA ----------
    try:
        spx_ma = sp500_ma_signal(prices, sp500_ticker="^GSPC", verbose=True)
    except Exception as e:
        spx_ma = f"Error: {e}"

    # ---------- 듀얼모멘텀 ----------
    try:
        dm = dual_momentum_signal(prices)
    except Exception as e:
        dm = f"Error: {e}"

    # ---------- LAA_DM ----------
    try:
        laa_dm = laa_dm_signal(prices, verbose=True)
    except Exception as e:
        laa_dm = f"Error: {e}"

    # ---------- LAA_MA (QQQ MA 기반 LAA) ----------
    try:
        laa_ma = laa_ma_signal(prices, verbose=True)
    except Exception as e:
        laa_ma = f"Error: {e}"

    # ---------- LAA_MA2 (QQQ + IWD MA 기반 LAA) ----------
    try:
        laa_ma2 = laa_ma2_signal(prices, verbose=True)
    except Exception as e:
        laa_ma2 = f"Error: {e}"

    # ---------- DM_RP (Dual Momentum + Risk Parity) ----------
    try:
        dm_rp = dm_rp_signal(prices, verbose=True)
    except Exception as e:
        dm_rp = f"Error: {e}"


    print("\n=== Signals ===")
    print_weight_result("LAA", laa)
    print(f"S&P MA    -> {spx_ma}")
    print_weight_result("DM", dm)
    print_weight_result("LAA_DM", laa_dm)
    print_weight_result("LAA_MA", laa_ma)
    print_weight_result("LAA_MA2", laa_ma2)
    print_weight_result("DM_RP", dm_rp)



if __name__ == "__main__":
    main()
