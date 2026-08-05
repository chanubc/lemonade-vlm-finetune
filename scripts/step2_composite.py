"""
STEP 2 (v2) — 실 조리 프레임 위 불꽃/발연 합성 + 하이브리드 실화재.

v1(합성-only) 실패 교훈 반영(results/step4_comparison.md):
  실패는 "불 못 잡음"이 아니라 오탐 증가. 원인 겨냥:
  (1) 붙여넣은 불꽃이 주변을 안 밝힘 → "합성 티" → 불빛 glow 추가.
  (2) 조리배경에 위험이 정상의 2배 → "조리배경=위험" 편향 → 정상 비중↑ 리밸런스.
  (3) 불꽃 다양성 부족 → NIST + D-Fire 실화재 컷아웃 혼합.
  (4) 실화재 학습 0장 → 하이브리드: D-Fire 원본 실화재 일부 직접 투입.
  (5) 누수 차단 → 홀드아웃 정상 세그먼트를 학습 배경에서 제외.

출력: data/synth/{normal,smoke,fire}/ + data/synth/real_fire/ + labels.jsonl
사용: uv run python scripts/step2_composite.py --normal 180 --smoke 120 --fire 120 --real-fire 40
"""
import argparse
import csv
import glob
import io
import json
import os
import random

import numpy as np
import cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEMONADE = os.path.join(HERE, "data", "frames")
MANIFEST = os.path.join(HERE, "data", "holdout_real", "holdout_v0_manifest.csv")
OUT = os.path.join(HERE, "data", "synth")


# ---------- 불꽃 컷아웃 ----------
def alpha_from_rgb(rgb):
    """RGB float(0..255) → alpha(0..1): 밝고(휘도) 따뜻한(R-B) 픽셀 = 불꽃. 흰선 배제."""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * R + 0.587 * G + 0.114 * B
    warm = R - B
    a = np.clip((lum - 55) / 150.0, 0, 1) * np.clip((warm - 12) / 60.0, 0, 1)
    a[a < 0.08] = 0.0
    return a


def crop_flame(rgb, a):
    ys, xs = np.where(a > 0)
    if len(xs) < 30:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    a_c = cv2.GaussianBlur(a[y0:y1 + 1, x0:x1 + 1], (0, 0), 1.2)
    return rgb[y0:y1 + 1, x0:x1 + 1].astype(np.uint8), np.clip(a_c, 0, 1)


def load_nist_flames():
    paths = []
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["label_suggested"] == "fire":
                    p = os.path.join(HERE, row["frame"])
                    if os.path.exists(p):
                        paths.append(p)
    out = []
    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        c = crop_flame(rgb, alpha_from_rgb(rgb))
        if c:
            out.append(c)
    return out


def load_dfire(n_flames, n_raw, seed):
    """캐시된 D-Fire test 샤드에서 실화재(class1) → 불꽃 컷아웃 + 원본 일부."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    from PIL import Image
    try:
        p = hf_hub_download("badsaarow/d-fire", "data/test-00000-of-00003.parquet",
                            repo_type="dataset")
    except Exception as e:
        print(f"[D-Fire 스킵] {e}")
        return [], []
    df = pd.read_parquet(p)
    fire_idx = [i for i, l in enumerate(df["label"])
                if any(ln.strip().split() and ln.strip().split()[0] == "1"
                       for ln in str(l).splitlines())]
    rng = random.Random(seed)
    rng.shuffle(fire_idx)
    flames, raw_dir, raws = [], os.path.join(OUT, "real_fire"), []
    os.makedirs(raw_dir, exist_ok=True)
    for i in fire_idx:
        if len(flames) >= n_flames and len(raws) >= n_raw:
            break
        img_field = df.iloc[i]["image"]
        data = img_field["bytes"] if isinstance(img_field, dict) else img_field
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(pil).astype(np.float32)
        if len(flames) < n_flames:
            c = crop_flame(arr, alpha_from_rgb(arr))
            if c and c[0].shape[0] > 24 and c[0].shape[1] > 24:
                flames.append(c)
        if len(raws) < n_raw:
            fp = os.path.join(raw_dir, f"dfire_fire_{len(raws):03d}.jpg")
            pil.save(fp, "JPEG", quality=92)
            raws.append(os.path.relpath(fp, HERE).replace("\\", "/"))
    return flames, raws


# ---------- 합성 연산 ----------
def composite_flame(bg, flame, rng):
    rgb_c, a_c = flame
    H, W = bg.shape[:2]
    fh0, fw0 = rgb_c.shape[:2]
    s = rng.uniform(0.30, 0.62) * H / fh0
    if fw0 * s > 0.95 * W:
        s = 0.95 * W / fw0
    nw = max(4, min(W, int(fw0 * s))); nh = max(4, min(H, int(fh0 * s)))
    fr = cv2.resize(rgb_c, (nw, nh)).astype(np.float32) / 255.0
    fa = cv2.resize(a_c, (nw, nh))
    if rng.random() < 0.5:
        fr = fr[:, ::-1]; fa = fa[:, ::-1]
    fr = np.clip(fr * rng.uniform(0.85, 1.15), 0, 1)
    cx = int(W * rng.uniform(0.35, 0.65)); cy = int(H * rng.uniform(0.42, 0.72))
    x0 = int(np.clip(cx - nw // 2, 0, W - nw)); y0 = int(np.clip(cy - nh // 2, 0, H - nh))
    # (a) 주변 불빛 glow: 따뜻한 halo를 장면에 더함 → "합성 티" 완화
    yy, xx = np.mgrid[0:H, 0:W]
    base_y = y0 + int(nh * 0.75)
    sig = max(8.0, 0.55 * nh)
    halo = np.exp(-(((xx - cx) ** 2 + (yy - base_y) ** 2) / (2 * sig * sig))).astype(np.float32)
    warm = np.array([1.0, 0.45, 0.12], np.float32)
    bg = np.clip(bg + rng.uniform(0.15, 0.30) * halo[..., None] * warm, 0, 1)
    # (b) 불꽃 알파-오버 + 발광
    roi = bg[y0:y0 + nh, x0:x0 + nw]; a3 = fa[..., None]
    over = roi * (1 - a3) + fr * a3
    bg[y0:y0 + nh, x0:x0 + nw] = np.clip(over + 0.25 * fr * a3, 0, 1)
    return bg


def add_smoke(bg, rng, gray=None):
    H, W = bg.shape[:2]
    mask = np.zeros((H, W), np.float32)
    cx = int(W * rng.uniform(0.4, 0.6)); base = int(H * rng.uniform(0.55, 0.72))
    ph = int(H * rng.uniform(0.35, 0.6)); n = rng.randint(14, 26)
    for i in range(n):
        t = i / n
        y = int(base - t * ph + rng.uniform(-4, 4))
        x = int(cx + np.sin(t * 6 + rng.random() * 6) * W * 0.05 * t + rng.uniform(-5, 5))
        r = int((0.04 + 0.10 * t) * W + rng.uniform(0, 6))
        cv2.circle(mask, (x, y), max(2, r), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(5, W * 0.022))
    mask = mask / (mask.max() + 1e-6) * rng.uniform(0.30, 0.58)
    g = gray if gray is not None else rng.uniform(0.62, 0.82)
    a3 = mask[..., None]
    return bg * (1 - a3) + np.ones_like(bg) * g * a3


def randomize(bg, rng):
    bg = bg * rng.uniform(0.82, 1.15)
    bg[..., 0] *= rng.uniform(0.95, 1.10); bg[..., 2] *= rng.uniform(0.90, 1.06)
    bg = np.clip(bg, 0, 1)
    br = rng.uniform(0, 1.3)
    if br > 0.3:
        bg = cv2.GaussianBlur(bg, (0, 0), br)
    ns = rng.uniform(0, 0.03)
    if ns > 0.005:
        bg = bg + np.random.normal(0, ns, bg.shape)
    return np.clip(bg, 0, 1)


# ---------- 홀드아웃 세그먼트 제외 (누수 차단) ----------
def holdout_segments():
    """step0c의 정상 샘플링(seed 0 random40 + seed1 pool180)을 재현해 제외 대상 세그먼트 집합."""
    segs = sorted(glob.glob(os.path.join(LEMONADE, "*", "")))
    excl = set()
    r0 = random.Random(0).sample(segs, min(40, len(segs)))
    excl.update(r0)
    pool_src = [s for s in segs if s not in set(r0)]
    r1 = random.Random(1).sample(pool_src, min(180, len(pool_src)))
    excl.update(r1)
    return excl


def sample_bgs(n, seed, exclude):
    segs = [s for s in sorted(glob.glob(os.path.join(LEMONADE, "*", ""))) if s not in exclude]
    pick = random.Random(seed).sample(segs, min(n, len(segs)))
    out = []
    for s in pick:
        fs = sorted(glob.glob(os.path.join(s, "*.jpg")))
        if fs:
            out.append(fs[len(fs) // 2])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", type=int, default=180)
    ap.add_argument("--smoke", type=int, default=120)
    ap.add_argument("--fire", type=int, default=120)
    ap.add_argument("--real-fire", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed); np.random.seed(args.seed)

    flames = load_nist_flames()
    print(f"불꽃 소스 — NIST: {len(flames)}")
    df_flames, raw_fires = load_dfire(n_flames=40, n_raw=args.real_fire, seed=args.seed)
    flames += df_flames
    print(f"불꽃 소스 — +D-Fire: {len(df_flames)} → 총 {len(flames)}, 원본 실화재 {len(raw_fires)}")
    if not flames:
        raise SystemExit("불꽃 컷아웃 없음.")

    excl = holdout_segments()
    n_bg = max(args.normal, args.smoke, args.fire)
    bgs = sample_bgs(n_bg, args.seed, exclude=excl)
    print(f"배경(LEMONADE, 홀드아웃 {len(excl)}세그 제외): {len(bgs)}장")

    for cls in ("normal", "smoke", "fire"):
        os.makedirs(os.path.join(OUT, cls), exist_ok=True)
    rows = []

    def bg_at(i):
        return cv2.cvtColor(cv2.imread(bgs[i % len(bgs)]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    for cls, cnt in (("normal", args.normal), ("smoke", args.smoke), ("fire", args.fire)):
        for i in range(cnt):
            img = bg_at(i)
            if cls == "smoke":
                img = add_smoke(img, rng)
            elif cls == "fire":
                img = composite_flame(img, rng.choice(flames), rng)
                if rng.random() < 0.5:
                    img = add_smoke(img, rng, gray=rng.uniform(0.3, 0.5))
            img = randomize(img, rng)
            out_rgb = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            path = os.path.join(OUT, cls, f"{cls}_{i:04d}.jpg")
            cv2.imwrite(path, cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
            rows.append({"path": os.path.relpath(path, HERE).replace("\\", "/"),
                         "label": cls, "src": "synth"})

    for rf in raw_fires:  # 하이브리드: 원본 실화재
        rows.append({"path": rf, "label": "fire", "src": "real_dfire"})

    with open(os.path.join(OUT, "labels.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"\n===== STEP 2 v2 완료 =====")
    print(f"총 {len(rows)}장  라벨 {dict(Counter(r['label'] for r in rows))}  "
          f"소스 {dict(Counter(r['src'] for r in rows))}")


if __name__ == "__main__":
    main()
