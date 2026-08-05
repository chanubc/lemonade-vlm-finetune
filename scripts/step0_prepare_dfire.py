"""
STEP 0 데이터 준비: D-Fire(공개 실화재셋)에서 baseline 측정용 소량 샘플을 만든다.

무엇을 하나:
  - HuggingFace 미러(badsaarow/d-fire, 원본 D-Fire를 parquet로 재배포)의 test 샤드 1개만 받아
  - 화재(fire) 이미지 N장 + 정상(normal) 이미지 N장을 결정론적으로 샘플해 jpg로 저장한다.

클래스 정의 (D-Fire YOLO 라벨 기준, 실측으로 확인함):
  - class 1 = fire(불), class 0 = smoke(연기)
  - fire 양성  = 라벨에 class 1(불)이 있는 이미지         → data/dfire_step0/fire/
  - normal 음성 = 라벨이 완전히 비어 있음(불·연기 다 없음)  → data/dfire_step0/normal/
  - 연기만 있는 이미지(class 0만)는 "불이냐"는 질문을 흐리므로 baseline에서 제외한다.

출력:
  - data/dfire_step0/fire/*.jpg, data/dfire_step0/normal/*.jpg
  - data/dfire_step0/manifest.jsonl  (각 줄: {"path","label"})  label ∈ {"fire","normal"}

사용:
  uv run python scripts/step0_prepare_dfire.py            # 클래스당 100장
  uv run python scripts/step0_prepare_dfire.py --per-class 150
"""
import argparse
import io
import json
import os
import random

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "data", "dfire_step0")
REPO = "badsaarow/d-fire"
# test 샤드 하나에 화재 ~249장, 정상 ~992장 → 클래스당 100~150장에 충분
SHARD = "data/test-00000-of-00003.parquet"


def label_classes(lbl: str):
    """YOLO 라벨 문자열 → 등장하는 class id 집합."""
    out = set()
    for line in lbl.strip().splitlines():
        parts = line.strip().split()
        if parts:
            out.add(parts[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=100, help="화재/정상 각 장수")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    import pandas as pd
    from PIL import Image

    print(f"D-Fire test 샤드 다운로드(캐시 사용): {REPO}/{SHARD}")
    path = hf_hub_download(REPO, SHARD, repo_type="dataset")
    df = pd.read_parquet(path)  # 컬럼: image, label, filename
    print(f"샤드 이미지 수: {len(df)}")

    cls = df["label"].apply(label_classes)
    is_fire = cls.apply(lambda s: "1" in s)          # 불 존재
    is_normal = df["label"].str.strip() == ""        # 불·연기 모두 없음

    fire_idx = list(df.index[is_fire])
    normal_idx = list(df.index[is_normal])
    print(f"후보 — 화재: {len(fire_idx)}장, 정상: {len(normal_idx)}장")

    n = args.per_class
    if len(fire_idx) < n or len(normal_idx) < n:
        n = min(len(fire_idx), len(normal_idx), n)
        print(f"[주의] 후보 부족 → 클래스당 {n}장으로 축소")

    rng = random.Random(args.seed)
    fire_pick = rng.sample(fire_idx, n)
    normal_pick = rng.sample(normal_idx, n)

    fire_dir = os.path.join(OUT_DIR, "fire")
    normal_dir = os.path.join(OUT_DIR, "normal")
    os.makedirs(fire_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    manifest = []

    def save(idx_list, label, out_dir):
        for i in idx_list:
            row = df.loc[i]
            img_field = row["image"]
            data = img_field["bytes"] if isinstance(img_field, dict) else img_field
            img = Image.open(io.BytesIO(data)).convert("RGB")
            fname = str(row["filename"])
            fpath = os.path.join(out_dir, fname)
            img.save(fpath, "JPEG", quality=95)
            manifest.append({"path": os.path.relpath(fpath, HERE).replace("\\", "/"),
                             "label": label})

    print(f"이미지 저장 중... (클래스당 {n}장)")
    save(fire_pick, "fire", fire_dir)
    save(normal_pick, "normal", normal_dir)

    mpath = os.path.join(OUT_DIR, "manifest.jsonl")
    with open(mpath, "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"완료: 화재 {n} + 정상 {n} = {len(manifest)}장")
    print(f"  이미지: {os.path.relpath(OUT_DIR, HERE)}/(fire|normal)/")
    print(f"  매니페스트: {os.path.relpath(mpath, HERE)}")


if __name__ == "__main__":
    main()
