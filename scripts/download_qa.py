"""
LEMONADE VQA 표(메타데이터)만 내려받는 스크립트.
비디오(약 61GB)는 받지 않고, 질문/보기/정답 표(약 1.4MB parquet)만 저장한다.

출력: data/raw/lemonade_qa.parquet  (36,521행)
"""
import os
import urllib.request

# HuggingFace가 자동 변환해 둔 parquet (비디오 미포함, 표 데이터만)
PARQUET_URL = (
    "https://huggingface.co/datasets/amathislab/LEMONADE/"
    "resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "raw", "lemonade_qa.parquet")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"다운로드 중: {PARQUET_URL}")
    req = urllib.request.Request(PARQUET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(OUT, "wb") as f:
        f.write(r.read())
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"저장 완료: {OUT} ({size_mb:.2f} MB)")

    # 간단 검증
    import pandas as pd

    df = pd.read_parquet(OUT)
    print(f"행 수: {len(df):,}, 열: {list(df.columns)}")


if __name__ == "__main__":
    main()
