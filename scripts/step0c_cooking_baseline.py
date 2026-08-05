"""
STEP 0-C — 조리 도메인 홀드아웃에서 기본 Qwen2.5-VL-3B 재측정.

왜: 원래 STEP 0은 야외 D-Fire로 쟀다(F1 0.819, 오탐 40%). 하지만 이 프로젝트 도메인은
    "주방"이다. 조리 도메인 실데이터(NIST cornoil 화재 + LEMONADE 정상 조리)로 다시 재서
    "기본 모델이 조리 맥락에서 실제로 얼마나 하나"를 본다.

누수 방지 설계(중요):
    위험을 전부 NIST, 정상을 전부 LEMONADE로 섞어 한 F1로 내면 모델이 "불"이 아니라
    "배경 도메인"으로 가른다 → 결과 뻥튀기. 그래서 3개를 분리 측정한다.
      (1) cornoil 내부: 불꽃 vs 빈 팬 (배경 동일) → 순수 화재 인식력
      (2) 랜덤 조리 정상(LEMONADE) → 대표 오탐률
      (3) hard 조리 정상(김·튀김 선별) → 스트레스 오탐률

A) hard-정상 만들기:
    LEMONADE 프레임에서 "김/연기/튀김이 보이나?"(yes/no) 질문으로 선별 → hard 정상.
    ※ 선별 질문은 '화재' 질문과 다르다 → 화재 baseline을 인위로 낮추지 않게.
    랜덤 정상은 그냥 무작위 조리 프레임(대표 분포).
    라벨은 사전 확정: LEMONADE는 실제 화재가 없는 정상 조리 → 전부 'normal'(모델 무관).

사용:
  uv run --group train python scripts/step0c_cooking_baseline.py
"""
import argparse
import glob
import json
import os
import random

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(HERE, "data", "holdout_real")
HOLDOUT = os.path.join(BASE, "holdout_v0")
COOK = os.path.join(BASE, "cooking_normals")
LEMONADE = os.path.join(HERE, "data", "frames")
OUT = os.path.join(HERE, "results", "step0c_cooking_f1.json")

FIRE_PROMPT = "Does this image show a fire? Answer only 'fire' or 'normal'."
STEAM_PROMPT = ("Does this kitchen image show visible steam, boiling, or active "
                "frying/sizzling? Answer only 'yes' or 'no'.")


def parse_fire(text):
    t = text.strip().lower()
    if not t:
        return "normal"
    if "no fire" in t or t.startswith("normal") or ("not" in t and "fire" in t):
        return "normal"
    fi, ni = t.find("fire"), t.find("normal")
    if fi == -1 and ni == -1:
        return "normal"
    if ni == -1:
        return "fire"
    if fi == -1:
        return "normal"
    return "fire" if fi < ni else "normal"


def parse_yes(text):
    return text.strip().lower().startswith("y")


def sample_segments(n, seed, exclude):
    segs = sorted(glob.glob(os.path.join(LEMONADE, "*", "")))
    segs = [s for s in segs if s not in exclude]
    rng = random.Random(seed)
    pick = rng.sample(segs, min(n, len(segs)))
    out = []
    for s in pick:
        fs = sorted(glob.glob(os.path.join(s, "*.jpg")))
        if fs:
            out.append((s, fs[len(fs) // 2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=40)
    ap.add_argument("--n-hard", type=int, default=40)
    ap.add_argument("--pool", type=int, default=180, help="hard 선별용 후보 풀")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm
    import shutil

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", max_pixels=1003520)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto")
    model.eval()

    def ask(img_path, prompt, max_new=16):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": "file://" + img_path},
            {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inp = processor(text=[text], images=imgs, videos=vids,
                        padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        return processor.batch_decode(gen[:, inp.input_ids.shape[1]:],
                                      skip_special_tokens=True)[0]

    # ---------- A. hard-정상 + 랜덤 정상 만들기 ----------
    rand_dir = os.path.join(COOK, "random")
    hard_dir = os.path.join(COOK, "hard")
    for d in (rand_dir, hard_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    print(f"[A] 랜덤 조리 정상 {args.n_random}장 샘플링...")
    rand_segs = sample_segments(args.n_random, args.seed, exclude=set())
    rand_paths = []
    for i, (seg, f) in enumerate(rand_segs):
        dst = os.path.join(rand_dir, f"rand_{i:03d}.jpg")
        shutil.copy(f, dst)
        rand_paths.append(dst)

    print(f"[A] hard 정상 선별(김/튀김) 풀 {args.pool}장에서 최대 {args.n_hard}장...")
    exclude = {s for s, _ in rand_segs}
    pool = sample_segments(args.pool, args.seed + 1, exclude=exclude)
    hard_paths = []
    for seg, f in tqdm(pool, desc="steam 선별"):
        if len(hard_paths) >= args.n_hard:
            break
        if parse_yes(ask(f, STEAM_PROMPT, max_new=6)):
            dst = os.path.join(hard_dir, f"hard_{len(hard_paths):03d}.jpg")
            shutil.copy(f, dst)
            hard_paths.append(dst)
    print(f"[A] 완료: 랜덤 {len(rand_paths)}장, hard {len(hard_paths)}장")

    # ---------- C. 재측정 ----------
    fire_frames = sorted(glob.glob(os.path.join(HOLDOUT, "fire", "*.jpg")))
    cornoil_norm = sorted(glob.glob(os.path.join(HOLDOUT, "normal", "*.jpg")))
    smoke_frames = sorted(glob.glob(os.path.join(HOLDOUT, "smoke", "*.jpg")))

    def run_fire(paths, desc):
        preds = []
        for p in tqdm(paths, desc=desc):
            preds.append(parse_fire(ask(p, FIRE_PROMPT)))
        return preds

    # (1) cornoil 내부: 불꽃(양성) vs 빈 팬(음성)
    fp_pred = run_fire(fire_frames, "1) cornoil fire")
    cn_pred = run_fire(cornoil_norm, "1) cornoil normal")
    tp = sum(p == "fire" for p in fp_pred); fn = len(fp_pred) - tp
    fp = sum(p == "fire" for p in cn_pred); tn = len(cn_pred) - fp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    # (2)(3) 조리 정상 오탐률
    rand_pred = run_fire(rand_paths, "2) random normal")
    hard_pred = run_fire(hard_paths, "3) hard normal")
    smoke_pred = run_fire(smoke_frames, "smoke frames")
    rand_fpr = sum(p == "fire" for p in rand_pred) / len(rand_pred) if rand_pred else 0.0
    hard_fpr = sum(p == "fire" for p in hard_pred) / len(hard_pred) if hard_pred else 0.0
    smoke_as_fire = sum(p == "fire" for p in smoke_pred) / len(smoke_pred) if smoke_pred else 0.0

    print("\n===== STEP 0-C: 조리 도메인 재측정 (기본 Qwen2.5-VL-3B) =====")
    print(f"(1) cornoil 내부 [불꽃 {len(fire_frames)} vs 빈팬 {len(cornoil_norm)}] — 배경 동일, 순수 화재인식")
    print(f"    Acc {acc:.3f} | Prec {prec:.3f} | Recall {rec:.3f} | F1 {f1:.3f}")
    print(f"    혼동: TP {tp} FN {fn} / FP {fp} TN {tn}")
    print(f"(2) 랜덤 조리 정상 {len(rand_paths)}장 → 오탐률(fire로 오답) {rand_fpr:.1%}")
    print(f"(3) hard 조리 정상 {len(hard_paths)}장(김/튀김) → 오탐률 {hard_fpr:.1%}")
    print(f"(+) 발연(smoke) {len(smoke_frames)}장 → 'fire'로 답한 비율 {smoke_as_fire:.1%}")
    print(f"\n[비교] 원래 D-Fire(야외): F1 0.819, Recall 0.970, 오탐 40%")

    summary = {
        "within_cornoil": {"n_fire": len(fire_frames), "n_normal": len(cornoil_norm),
                            "acc": round(acc, 4), "precision": round(prec, 4),
                            "recall": round(rec, 4), "f1": round(f1, 4),
                            "tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "random_normal_fp_rate": round(rand_fpr, 4), "n_random": len(rand_paths),
        "hard_normal_fp_rate": round(hard_fpr, 4), "n_hard": len(hard_paths),
        "smoke_called_fire_rate": round(smoke_as_fire, 4), "n_smoke": len(smoke_frames),
        "reference_dfire": {"f1": 0.819, "recall": 0.970, "fp_rate": 0.40},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(summary, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: {os.path.relpath(OUT, HERE)}")
    print(f"조리 정상 프레임: {os.path.relpath(COOK, HERE)}/(random|hard)/")


if __name__ == "__main__":
    main()
