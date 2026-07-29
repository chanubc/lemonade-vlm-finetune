"""
참가자(Participant) 단위로 train/val/test를 나눈다.

왜 참가자 단위인가:
  같은 사람의 영상이 train과 test에 함께 들어가면, 모델이 "이 사람 = 나이 40" 같은
  개인 특성을 외워서 점수가 부풀려진다(데이터 누수). 참가자 자체를 통째로 한 쪽에만
  넣어야 "처음 보는 사람"에 대한 진짜 일반화 성능을 측정할 수 있다.

방법:
  참가자를 QA 개수 내림차순으로 정렬한 뒤, 매번 "목표 대비 가장 부족한 split"에
  통째로 배정하는 그리디(greedy) 방식. 결정론적이라 매번 같은 결과가 나온다.

출력:
  data/splits/train.parquet, val.parquet, test.parquet
  data/splits/split_manifest.json  (어느 참가자가 어느 split인지 기록)
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(HERE, "data", "raw", "lemonade_qa.parquet")
OUT_DIR = os.path.join(HERE, "data", "splits")

# 목표 비율 (QA 개수 기준). 합이 1이 되도록.
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def assign_participants(counts: dict, ratios: dict, total: int) -> dict:
    """참가자를 split에 통째로 배정. 반환: {participant: split}."""
    targets = {s: r * total for s, r in ratios.items()}
    current = {s: 0 for s in ratios}
    assignment = {}
    # 큰 참가자부터 배정해야 목표에 잘 수렴한다.
    for participant, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        # "목표까지 남은 여유(target - current)"가 가장 큰 split을 고른다.
        best = max(ratios, key=lambda s: targets[s] - current[s])
        assignment[participant] = best
        current[best] += n
    return assignment


def main():
    df = pd.read_parquet(IN)
    df["participant"] = df["Clip"].str.split("_").str[0]
    total = len(df)

    counts = df["participant"].value_counts().to_dict()
    assignment = assign_participants(counts, RATIOS, total)
    df["split"] = df["participant"].map(assignment)

    os.makedirs(OUT_DIR, exist_ok=True)

    # split별 저장 + 리포트
    manifest = {"ratios_target": RATIOS, "splits": {}}
    print(f"{'split':6} {'#QA':>7} {'비율':>7}  참가자")
    print("-" * 70)
    for s in RATIOS:
        sub = df[df["split"] == s].drop(columns=["split"])
        sub.to_parquet(os.path.join(OUT_DIR, f"{s}.parquet"), index=False)
        parts = sorted(sub["participant"].unique())
        manifest["splits"][s] = {
            "n_qa": int(len(sub)),
            "ratio": round(len(sub) / total, 4),
            "participants": parts,
        }
        print(f"{s:6} {len(sub):>7,} {len(sub)/total:>6.1%}  {', '.join(parts)}")

    # 누수 검증: 참가자가 두 split에 걸치면 안 됨
    seen = {}
    for s, info in manifest["splits"].items():
        for p in info["participants"]:
            assert p not in seen, f"누수! {p} in {seen[p]} and {s}"
            seen[p] = s
    print("-" * 70)
    print("누수 검증 통과: 모든 참가자가 정확히 한 split에만 속함.")

    # 각 split에 6개 하위카테고리가 모두 있는지 확인
    print("\n=== split별 하위카테고리 분포 ===")
    pv = pd.crosstab(df["split"], df["Subcategory"])
    print(pv.to_string())

    with open(os.path.join(OUT_DIR, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n매니페스트 저장: {os.path.join(OUT_DIR, 'split_manifest.json')}")


if __name__ == "__main__":
    main()
