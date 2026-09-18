"""NYSE month-end decisions, including weekends and exchange holidays."""
import pandas as pd
import pandas_market_calendars as mcal


def completed_month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.empty:
        return pd.DatetimeIndex([])
    start = index.min().to_period('M').start_time
    end = index.max().to_period('M').end_time.normalize()
    sessions = mcal.get_calendar('NYSE').schedule(start_date=start, end_date=end).index
    ends = pd.Series(sessions, index=sessions).resample('ME').last().dropna()
    return pd.DatetimeIndex(ends[ends <= index.max()]).intersection(index)
