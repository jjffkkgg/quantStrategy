# config.py

# 사용할 ETF 티커들

TICKERS = [
    # LAA / LAA_MA / LAA_DM 에 필요한 것들
    "QQQ", "IEF", "IWD", "IAU", "SGOV", "^NDX",

    # 듀얼모멘텀(DM) 및 기타
    "SPY", "EFA", "SHY", "TLT", "TIP", "LQD", "HYG", "BWX", "EMB",

    # 기타 전략용 (원하면 유지)
    "AGG", "DBC", "EEM", "VNQ",

    # S&P500 MA 전략
    "^GSPC",
]

# 가격 데이터 시작일
START_DATE = "1970-01-01"

# 한 달을 몇 거래일로 볼지 (모멘텀 계산용)
TRADING_DAYS_PER_MONTH = 21

# 모멘텀 룩백 (개월 단위)
MOMENTUM_MONTHS = [3, 6, 12]
