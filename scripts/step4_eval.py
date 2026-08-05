"""
STEP 4-c — 파인튜닝 전/후를 실 홀드아웃으로 평가 (조리 안전 3분류).

평가셋(전부 실데이터, 학습 미사용):
  - 위험: holdout_v0/fire (cornoil 불꽃), holdout_v0/smoke (발연)
  - 정상(cornoil 리그): holdout_v0/normal
  - 정상(실조리): cooking_normals/random, cooking_normals/hard

측정: 3분류 정확도 + "위험(smoke|fire) vs 정상" 이진 + 조리 정상 오탐률.
비교: base(어댑터 없음) vs finetuned(--adapter).

사용:
  uv run --group train python scripts/step4_eval.py                 # base
  uv run --group train python scripts/step4_eval.py --adapter out/qwen2p5vl-3b-cooking-qlora
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(HERE, "data", "holdout_real")
PROMPT = ("You are a kitchen safety monitor watching a stovetop. "
          "Classify the scene with exactly one word: "
          "'normal' (safe cooking; steam, frying, and food are fine), "
          "'smoke' (dangerous smoke such as overheating oil), or "
          "'fire' (visible flames). Answer:")


def parse3(text):
    t = text.strip().lower()
    for k in ("fire", "smoke", "normal"):
        if k in t:
            return k
    return "normal"


def collect():
    g = lambda *p: sorted(glob.glob(os.path.join(BASE, *p)))
    return {
        "fire":        [(p, "fire")   for p in g("holdout_v0", "fire", "*.jpg")],
        "smoke":       [(p, "smoke")  for p in g("holdout_v0", "smoke", "*.jpg")],
        "normal_rig":  [(p, "normal") for p in g("holdout_v0", "normal", "*.jpg")],
        "normal_rand": [(p, "normal") for p in g("cooking_normals", "random", "*.jpg")],
        "normal_hard": [(p, "normal") for p in g("cooking_normals", "hard", "*.jpg")],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", max_pixels=1003520)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    tag = "finetuned" if args.adapter else "base"

    def ask(p):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": "file://" + p}, {"type": "text", "text": PROMPT}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inp = processor(text=[text], images=imgs, videos=vids, padding=True,
                        return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=8, do_sample=False)
        return parse3(processor.batch_decode(gen[:, inp.input_ids.shape[1]:],
                                             skip_special_tokens=True)[0])

    groups = collect()
    results = {}
    for name, items in groups.items():
        preds = [ask(p) for p, _ in tqdm(items, desc=f"{tag}:{name}")]
        results[name] = list(zip([g for _, g in items], preds))

    # 집계
    def danger(x): return x in ("fire", "smoke")
    # 위험 재현율
    fire_hit = sum(danger(p) for _, p in results["fire"])
    smoke_hit = sum(danger(p) for _, p in results["smoke"])
    n_fire, n_smoke = len(results["fire"]), len(results["smoke"])
    # 조리 정상 오탐(위험이라고 답한 비율)
    def fp_rate(name):
        r = results[name]
        return (sum(danger(p) for _, p in r) / len(r)) if r else 0.0
    # 3분류 정확도(smoke를 fire로 답한 것도 위험은 맞지만 3분류는 틀림)
    def acc3(name):
        r = results[name]
        return (sum(g == p for g, p in r) / len(r)) if r else 0.0

    print(f"\n===== STEP 4 평가 [{tag}] =====")
    print(f"위험 재현율: fire {fire_hit}/{n_fire}, smoke(발연) {smoke_hit}/{n_smoke} (위험으로 감지)")
    print(f"발연 3분류 정확도(정확히 'smoke'): {acc3('smoke'):.1%}")
    print(f"조리 정상 오탐률: 랜덤 {fp_rate('normal_rand'):.1%} | hard {fp_rate('normal_hard'):.1%}")
    print(f"cornoil 빈팬 오탐률: {fp_rate('normal_rig'):.1%}")

    summary = {
        "tag": tag, "adapter": args.adapter,
        "fire_recall": f"{fire_hit}/{n_fire}", "smoke_detected_as_danger": f"{smoke_hit}/{n_smoke}",
        "smoke_exact_acc": round(acc3("smoke"), 4),
        "fp_random": round(fp_rate("normal_rand"), 4),
        "fp_hard": round(fp_rate("normal_hard"), 4),
        "fp_cornoil_rig": round(fp_rate("normal_rig"), 4),
        "n": {k: len(v) for k, v in results.items()},
    }
    out = args.out or os.path.join(HERE, "results", f"step4_eval_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"저장: {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
