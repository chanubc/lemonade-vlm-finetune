"""
STEP 0 — 파인튜닝 안 한 기본 Qwen2.5-VL-3B의 "화재 인식" 성능을 측정한다.

이 숫자가 프로젝트 방향을 가른다 (HANDOFF.md 5절):
  - baseline F1이 높으면(예: 0.85+)  → 단순 화재감지는 약함 → 어려운 태스크로 전환.
  - baseline F1이 낮으면              → 파인튜닝 가치 있음 → 그대로 진행.

방식:
  - 각 이미지에 VQA 프롬프트로 질의: "Does this image show a fire? Answer only 'fire' or 'normal'."
  - greedy 디코딩(T=0), 출력에서 fire/normal 판정.
  - 양성 클래스 = fire. precision/recall/F1 + 정확도 + 혼동행렬 집계.

선행:
  - uv run python scripts/step0_prepare_dfire.py   (data/dfire_step0/manifest.jsonl 생성)
  - uv sync --group train

사용:
  uv run --group train python scripts/step0_baseline.py
  uv run --group train python scripts/step0_baseline.py --limit 20   # 빠른 점검
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(HERE, "data", "dfire_step0", "manifest.jsonl")
DEFAULT_OUT = os.path.join(HERE, "results", "step0_baseline_f1.json")
PROMPT = "Does this image show a fire? Answer only 'fire' or 'normal'."


def parse_pred(text: str):
    """모델 출력 → 'fire' / 'normal' / None(판독불가).

    프롬프트가 한 단어를 강제하므로 대부분 정확히 'fire'|'normal'.
    'no fire' 같은 부정 표현은 normal로 처리한다.
    """
    t = text.strip().lower()
    if not t:
        return None
    # 부정 표현 우선 처리 ('no fire', 'not a fire' → normal)
    if "no fire" in t or "not " in t or t.startswith("normal") or "no fire" in t:
        return "normal"
    # 첫 등장 단어 기준
    fi = t.find("fire")
    ni = t.find("normal")
    if fi == -1 and ni == -1:
        return None
    if ni == -1:
        return "fire"
    if fi == -1:
        return "normal"
    return "fire" if fi < ni else "normal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--limit", type=int, default=None, help="앞 N개만(빠른 점검)")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--max-pixels", type=int, default=1003520,
                    help="이미지 최대 픽셀(VRAM 절약). 기본 ~1280*784")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm

    records = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit:
        records = records[: args.limit]
    print(f"평가 대상: {len(records)}장  (fire {sum(r['label']=='fire' for r in records)} / "
          f"normal {sum(r['label']=='normal' for r in records)})")

    print(f"모델 로드: {args.model} (파인튜닝 없음, base)")
    proc_kwargs = {}
    if args.max_pixels:
        proc_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model, **proc_kwargs)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto")
    model.eval()

    # 혼동행렬 (양성 = fire)
    tp = fp = fn = tn = 0
    unparsed = 0
    results = []

    for r in tqdm(records, desc="STEP0 baseline"):
        img_path = os.path.join(HERE, r["path"])
        messages = [{"role": "user", "content": [
            {"type": "image", "image": "file://" + img_path},
            {"type": "text", "text": PROMPT},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos,
                           padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        out_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        pred = parse_pred(out_text)
        if pred is None:
            unparsed += 1
            pred = "normal"   # 판독불가 → 음성으로 보수적 처리
        gold = r["label"]

        if gold == "fire" and pred == "fire":
            tp += 1
        elif gold == "normal" and pred == "fire":
            fp += 1
        elif gold == "fire" and pred == "normal":
            fn += 1
        else:
            tn += 1
        results.append({"path": r["path"], "gold": gold, "pred": pred,
                        "raw": out_text.strip()})

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    print("\n===== STEP 0: 기본 Qwen2.5-VL-3B 화재 인식 (양성=fire) =====")
    print(f"이미지 수      : {total}   (판독불가 {unparsed} → normal 처리)")
    print(f"정확도(Acc)    : {accuracy:.3f}")
    print(f"정밀도(Prec)   : {precision:.3f}")
    print(f"재현율(Recall) : {recall:.3f}")
    print(f"F1             : {f1:.3f}")
    print("\n[혼동행렬]           pred_fire  pred_normal")
    print(f"  gold_fire            {tp:>6}      {fn:>6}")
    print(f"  gold_normal          {fp:>6}      {tn:>6}")

    verdict = ("HIGH → 단순 화재감지는 약함. HANDOFF 5절 A/B(어려운 태스크·distillation)로 전환 검토."
               if f1 >= 0.85 else
               "LOW/MID → 파인튜닝 가치 있음. STEP 1 진행 가능.")
    print(f"\n판단 기준(F1 0.85): {verdict}")

    summary = {
        "model": args.model, "n": total, "positive_class": "fire",
        "accuracy": round(accuracy, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "f1": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "unparsed": unparsed, "prompt": PROMPT,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f,
                  ensure_ascii=False, indent=1)
    print(f"\n저장: {os.path.relpath(args.out, HERE)}")


if __name__ == "__main__":
    main()
