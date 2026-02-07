# utils/macro_data.py

import io
import pandas as pd
import requests


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
