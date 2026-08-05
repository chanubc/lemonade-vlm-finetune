"""
STEP 2b — v3 하이브리드 학습셋: 실 조리화재 영상(정상+화재) + 합성 발연.

v2 실패 교훈(results/step4_big_holdout.md): 합성 불꽃을 조리배경에 얹은 학습이
"스토브=위험" 편향을 키워 정상 오탐을 76%→95%로 악화. 필요한 건 **실 스토브 정상**.

접근:
  - 학습 영상(split=train: 소방서/데모 조리화재)의 프레임을 **불꽃 검출 휴리스틱**으로 자동 라벨.
    · 휴리스틱 = 밝고 따뜻한(R-B) 픽셀 비율(flame_area). VLM 안 씀(모델 편향 라벨 방지).
    · flame_area > THR_FIRE → fire, < THR_NORM → normal(=실 스토브 정상!), 중간 → 스킵.
  - smoke는 실데이터 부족 → 합성 발연(LEMONADE 배경, 홀드아웃 세그 제외).
  - 영상 단위 split: 여기 쓰는 영상은 홀드아웃(NIST cornoil/DVIDS/northants)과 겹치지 않음.

출력: data/synth/{normal,fire,smoke}/ (덮어씀) + labels.jsonl + _v3_review.jpg
사용: uv run python scripts/step2b_build_real_train.py
"""
import csv
import glob
import json
import os
import random
import shutil

import numpy as np
import cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_DIR = os.path.join(HERE, "data", "holdout_real", "frames")
SOURCES = os.path.join(HERE, "data", "holdout_real", "sources.json")
OUT = os.path.join(HERE, "data", "synth")

from step2_composite import alpha_from_rgb, add_smoke, sample_bgs, holdout_segments, randomize

THR_FIRE = 0.08    # flame_area > → fire (0.08로 상향: 텍스트/진행자 오검출 배제, 진짜 불꽃만)


def flame_area(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return float((alpha_from_rgb(rgb) > 0.15).mean())


def train_videos():
    """[(id, force_label|None)] — split=train & enabled."""
    src = json.load(open(SOURCES, encoding="utf-8"))
    return [(s["id"], s.get("force_label"))
            for s in src if s.get("split") == "train" and s.get("enabled", True)]


def montage(items, out_path, cols=8, thumb=180):
    from PIL import Image, ImageDraw
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    c = Image.new("RGB", (cols * thumb, rows * thumb), (18, 18, 18))
    d = ImageDraw.Draw(c)
    col = {"normal": (120, 230, 120), "fire": (250, 90, 70), "smoke": (240, 210, 90)}
    for i, (p, lab, fa) in enumerate(items):
        im = Image.open(p).convert("RGB"); im.thumbnail((thumb, thumb))
        c.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
        d.text(((i % cols) * thumb + 3, (i // cols) * thumb + 3),
               f"{lab} {fa:.3f}", fill=col.get(lab, (255, 255, 255)))
    c.save(out_path, "JPEG", quality=88)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-normal", type=int, default=170)
    ap.add_argument("--cap-fire", type=int, default=130)
    ap.add_argument("--smoke", type=int, default=90)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    rng = random.Random(args.seed); np.random.seed(args.seed)

    vids = train_videos()
    print(f"학습 영상(split=train): {vids}")
    normal_src, fire_src, review = [], [], []
    for vid, force in vids:
        for f in sorted(glob.glob(os.path.join(FRAME_DIR, vid, "*.jpg"))):
            if force == "normal":
                # 정상 조리 영상: 모든 프레임 normal (화재 없음이 확실)
                normal_src.append((f, 0.0))
                continue
            # 화재 데모 영상: 불꽃 프레임만 fire로(진행자·텍스트카드·정상은 버림)
            fa = flame_area(f)
            if fa is not None and fa > THR_FIRE:
                fire_src.append((f, fa))
    print(f"라벨 — 실 normal(조리영상): {len(normal_src)}, 실 fire(데모 불꽃): {len(fire_src)}")

    rng.shuffle(normal_src); rng.shuffle(fire_src)
    normal_src = normal_src[: args.cap_normal]
    fire_src = fire_src[: args.cap_fire]

    for cls in ("normal", "fire", "smoke"):
        d = os.path.join(OUT, cls)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
    if os.path.isdir(os.path.join(OUT, "real_fire")):
        shutil.rmtree(os.path.join(OUT, "real_fire"))

    rows = []
    from PIL import Image

    def dump(src, cls):
        for i, (f, fa) in enumerate(src):
            dst = os.path.join(OUT, cls, f"{cls}_{i:04d}.jpg")
            Image.open(f).convert("RGB").save(dst, "JPEG", quality=95)
            rows.append({"path": os.path.relpath(dst, HERE).replace("\\", "/"),
                         "label": cls, "src": "real_train"})
        for f, fa in src[:16]:
            review.append((f, cls, fa))

    dump(normal_src, "normal")
    dump(fire_src, "fire")

    # 합성 발연 (LEMONADE 배경, 홀드아웃 세그 제외)
    excl = holdout_segments()
    bgs = sample_bgs(args.smoke, args.seed, exclude=excl)
    for i, bgp in enumerate(bgs):
        img = cv2.cvtColor(cv2.imread(bgp), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = randomize(add_smoke(img, rng), rng)
        out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        dst = os.path.join(OUT, "smoke", f"smoke_{i:04d}.jpg")
        cv2.imwrite(dst, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        rows.append({"path": os.path.relpath(dst, HERE).replace("\\", "/"),
                     "label": "smoke", "src": "synth_smoke"})
        if i < 16:
            review.append((dst, "smoke", 0.0))

    with open(os.path.join(OUT, "labels.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    montage(review, os.path.join(OUT, "_v3_review.jpg"))

    from collections import Counter
    print(f"\n===== v3 학습셋 =====")
    print(f"총 {len(rows)}  {dict(Counter(r['label'] for r in rows))}  "
          f"{dict(Counter(r['src'] for r in rows))}")
    print("검수 몽타주: data/synth/_v3_review.jpg (라벨 정확한지 확인)")


if __name__ == "__main__":
    main()
