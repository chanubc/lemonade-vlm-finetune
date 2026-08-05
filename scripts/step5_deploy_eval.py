"""
STEP 5 — 배포 지표 평가 (홀드아웃 정상을 의미 있는 범주로 분해).

기존 "스토브 정상 오탐 52%"는 홀드아웃 정상이 적대적 프레임에 편중돼 실제보다 나쁘게 보였다.
범주를 나눠 진짜 배포 지표를 잡는다:
  - fire        : 실제 불꽃 (재현율이 중요)
  - everyday    : 일상 스토브 조리(불 안 남) = holdout 예약 조리영상 → **핵심 오탐 지표**
  - borderline  : 발화 직전 팬(dvids/northants 정상) = "곧 위험" 회색지대 → 참고
  - labrig      : cornoil 어두운 랩 리그 = 비대표적 → 참고
  - smoke       : 발연

사용:
  uv run --group train python scripts/step5_deploy_eval.py                    # base
  uv run --group train python scripts/step5_deploy_eval.py --adapter out/qwen2p5vl-3b-cooking-qlora
"""
import argparse
import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = os.path.join(HERE, "data", "holdout_real", "holdout_v0")
CN = os.path.join(HERE, "data", "holdout_real", "cooking_normals")
PROMPT = ("You are a kitchen safety monitor watching a stovetop. "
          "Classify the scene with exactly one word: "
          "'normal' (safe cooking; steam, frying, and food are fine), "
          "'smoke' (dangerous smoke such as overheating oil), or "
          "'fire' (visible flames). Answer:")


def parse3(t):
    t = t.strip().lower()
    for k in ("fire", "smoke", "normal"):
        if k in t:
            return k
    return "normal"


def cat_of_normal(path):
    b = os.path.basename(path)
    if b.startswith("nist_cornoil"):
        return "labrig(랩,비대표)"
    if b.startswith("dvids") or b.startswith("northants"):
        return "borderline(발화직전)"
    return None  # normal_everyday 폴더는 따로 처리


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm

    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", max_pixels=1003520)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    tag = "finetuned" if args.adapter else "base"

    def ask(p):
        m = [{"role": "user", "content": [
            {"type": "image", "image": "file://" + p}, {"type": "text", "text": PROMPT}]}]
        tx = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        im, vi = process_vision_info(m)
        inp = proc(text=[tx], images=im, videos=vi, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=8, do_sample=False)
        return parse3(proc.batch_decode(g[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0])

    def danger(x):
        return x in ("fire", "smoke")

    # 범주별 프레임 모으기
    groups = {
        "everyday(일상,핵심)": glob.glob(os.path.join(H, "normal_everyday", "*.jpg")),
        "everyday(LEMONADE)": glob.glob(os.path.join(CN, "random", "*.jpg")) +
                               glob.glob(os.path.join(CN, "hard", "*.jpg")),
    }
    norm_mixed = glob.glob(os.path.join(H, "normal", "*.jpg"))
    for p in norm_mixed:
        c = cat_of_normal(p)
        if c:
            groups.setdefault(c, []).append(p)
    fire = glob.glob(os.path.join(H, "fire", "*.jpg"))
    smoke = glob.glob(os.path.join(H, "smoke", "*.jpg"))

    print(f"\n===== STEP 5 배포 지표 [{tag}] =====")
    # 정상 범주: 오탐률(위험이라 답한 비율)
    out = {"tag": tag}
    print("[정상 범주별 오탐률 (낮을수록 좋음)]")
    for name, paths in groups.items():
        if not paths:
            continue
        fp = sum(danger(ask(p)) for p in tqdm(paths, desc=name, leave=False))
        rate = fp / len(paths)
        out[name] = {"fp": fp, "n": len(paths), "rate": round(rate, 3)}
        print(f"  {name:22} {fp:>3}/{len(paths):<3} = {rate:.0%}")
    # 화재/발연 재현율
    fr = sum(danger(ask(p)) for p in tqdm(fire, desc="fire", leave=False))
    sr = sum(danger(ask(p)) for p in tqdm(smoke, desc="smoke", leave=False))
    out["fire_recall"] = {"hit": fr, "n": len(fire)}
    out["smoke_recall"] = {"hit": sr, "n": len(smoke)}
    print(f"[위험 재현율 (높을수록 좋음)]")
    print(f"  fire  {fr}/{len(fire)}")
    print(f"  smoke {sr}/{len(smoke)}")

    op = os.path.join(HERE, "results", f"step5_deploy_{tag}.json")
    json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"저장: {os.path.relpath(op, HERE)}")


if __name__ == "__main__":
    main()
