# utils/macro_data.py

import io
import pandas as pd
import requests
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def load_unemployment_vintages() -> pd.DataFrame:
    """UNRATE values indexed by observation and actual availability dates.

    UNRATE_VINTAGES_CSV may point to an ALFRED-format CSV containing date,
    realtime_start, value. Otherwise FRED_API_KEY is required. Never fall back
    to revised current data, which would silently reintroduce look-ahead.
    """
    path = os.environ.get('UNRATE_VINTAGES_CSV')
    if path:
        data = pd.read_csv(path)
    else:
        key = os.environ.get('FRED_API_KEY')
        if not key:
            raise RuntimeError('Point-in-time UNRATE requires FRED_API_KEY or '
                               'UNRATE_VINTAGES_CSV (date,realtime_start,value).')
        rows = []
        offset = 0
        while True:
            try:
                response = requests.get(
                    'https://api.stlouisfed.org/fred/series/observations',
                    params=dict(series_id='UNRATE', api_key=key, file_type='json',
                                realtime_start='1776-07-04', realtime_end='9999-12-31',
                                output_type=1, limit=100000, offset=offset), timeout=60)
            except requests.RequestException:
                # Request exceptions can include the complete URL and API key.
                raise RuntimeError('ALFRED connection failed. Check network access and retry.') from None
            # Do not include the request URL (which contains the key) in errors.
            if response.status_code != 200:
                raise RuntimeError(f'ALFRED request failed: HTTP {response.status_code}')
            payload = response.json()
            batch = payload.get('observations', [])
            rows.extend(batch)
            offset += len(batch)
            if offset >= int(payload.get('count', 0)):
                break
            if not batch:
                raise RuntimeError('Incomplete ALFRED response.')
        data = pd.DataFrame(rows)
    required = ['date', 'realtime_start', 'value']
    if not set(required).issubset(data.columns):
        raise ValueError('UNRATE vintage data requires date,realtime_start,value.')
    data = data[required].copy()
    for col in required[:2]:
        data[col] = pd.to_datetime(data[col], errors='raise')
    data['value'] = pd.to_numeric(data['value'], errors='coerce')
    if data.empty:
        raise ValueError('UNRATE vintage history is empty.')
    return data.sort_values(['realtime_start', 'date'])


def unemployment_asof(vintages: pd.DataFrame, decision_date) -> pd.Series:
    """Reconstruct the latest available revision for each observation month."""
    date = pd.Timestamp(decision_date)
    known = vintages.loc[(vintages.realtime_start <= date) & (vintages.date <= date)]
    latest = known.sort_values('realtime_start').drop_duplicates('date', keep='last')
    return latest.set_index('date')['value'].sort_index().dropna().rename('UNRATE')


def load_unemployment_rate(start: str = "1950-01-01") -> pd.Series:
    """
    FRED에서 미국 실업률(UNRATE)을 CSV로 직접 다운로드해서 로드.
    - pandas_datareader 없이 동작
    - 컬럼 이름이 살짝 바뀌어도 견딜 수 있도록 느슨하게 파싱
    """

    csv_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"

    r = requests.get(csv_url)
    if r.status_code != 200:
        raise RuntimeError(f"FRED 다운로드 실패: status {r.status_code}")

    # 1) 먼저 그냥 읽는다 (parse_dates, index_col 지정 X)
    df = pd.read_csv(io.StringIO(r.text))

    if df.empty:
        raise RuntimeError("FRED에서 받아온 UNRATE CSV가 비어있습니다.")

    # 2) 날짜 컬럼 추론
    #    - 'DATE'라는 이름이 있으면 그걸 사용
    #    - 아니면 첫 번째 컬럼을 날짜로 간주
    if "DATE" in df.columns:
        date_col = "DATE"
    else:
        date_col = df.columns[0]

    # 날짜 변환
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col)

    # 3) 값 컬럼 추론
    #    - 'UNRATE' 컬럼이 있으면 그걸 사용
    #    - 아니면 날짜 컬럼을 제외한 첫 번째 컬럼을 값으로 사용
    if "UNRATE" in df.columns:
        val_col = "UNRATE"
    else:
        # 날짜 인덱스 제외하고 남은 컬럼들 중 하나 선택
        value_candidates = [c for c in df.columns]
        if not value_candidates:
            raise RuntimeError("UNRATE 값 컬럼을 찾을 수 없습니다.")
        val_col = value_candidates[0]

    s = pd.to_numeric(df[val_col], errors="coerce").dropna()
    s.name = "UNRATE"

    # 4) 시작 날짜 이후만 사용
    s = s[s.index >= pd.to_datetime(start)]

    if s.empty:
        raise RuntimeError("시작 날짜 이후의 UNRATE 데이터가 없습니다.")

    return s
