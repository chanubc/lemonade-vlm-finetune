"""
Kanana-1.5-v-3b 를 LEMONADE(Perception/Reasoning) test로 평가한다.
evaluate.py(Qwen용)의 카나나 버전 — 데이터·채점 로직은 동일, 모델 로딩/추론만 카나나 API로.

카나나는 커스텀 아키텍처라 로딩 방식이 다르다:
  AutoModelForVision2Seq + AutoProcessor (trust_remote_code=True)
  입력: {"image":[PIL...], "conv":[{"user","<image>*N"},{"user","질문"}]}
       -> processor.batch_encode_collate([sample], padding_side="left", add_generation_prompt=True)

실행(모델 ~6-7GB 최초 다운로드 + GPU 필요):
  uv run python scripts/evaluate_kanana.py --split test --out out/eval_kanana_before.json
  uv run python scripts/evaluate_kanana.py --split test --limit 20   # 빠른 스모크
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "kakaocorp/kanana-1.5-v-3b-instruct"
LETTERS = ["A", "B", "C", "D"]


def parse_choice(text: str) -> str:
    m = re.search(r"[ABCD]", text.strip().upper())
    return m.group(0) if m else "A"


def frames_exist(record) -> bool:
    return all(os.path.exists(os.path.join(HERE, p)) for p in record["images"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--adapter", default=None, help="LoRA 어댑터(파인튜닝 후)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from PIL import Image
    from tqdm import tqdm

    records = json.load(open(os.path.join(HERE, "data", "converted", f"{args.split}.json"), encoding="utf-8"))
    before = len(records)
    records = [r for r in records if frames_exist(r)]
    if len(records) < before:
        print(f"프레임 없는 {before-len(records)}개 제외 → {len(records)}개")
    if args.limit:
        records = records[: args.limit]

    print(f"모델 로드: {args.model} (trust_remote_code)")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def acc(d):
        return d["correct"] / d["total"] if d["total"] else 0.0

    overall = {"correct": 0, "total": 0}
    by_sub, by_diff, results = {}, {}, []

    for r in tqdm(records, desc="kanana 평가"):
        imgs = [Image.open(os.path.join(HERE, p)).convert("RGB") for p in r["images"]]
        prompt = r["messages"][0]["content"].replace("<image>", "").strip()
        sample = {
            "image": imgs,
            "conv": [
                {"role": "user", "content": " ".join(["<image>"] * len(imgs))},
                {"role": "user", "content": prompt},
            ],
        }
        inputs = processor.batch_encode_collate(
            [sample], padding_side="left", add_generation_prompt=True)
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            gens = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        in_len = inputs["input_ids"].shape[1]
        out_text = processor.tokenizer.batch_decode(gens[:, in_len:], skip_special_tokens=True)[0]

        pred = parse_choice(out_text)
        gold = r["messages"][1]["content"].strip().upper()
        ok = int(pred == gold)
        sub, diff = r["meta"]["subcategory"], r["meta"]["difficulty"]
        overall["total"] += 1; overall["correct"] += ok
        by_sub.setdefault(sub, {"correct": 0, "total": 0}); by_diff.setdefault(diff, {"correct": 0, "total": 0})
        by_sub[sub]["correct"] += ok; by_sub[sub]["total"] += 1
        by_diff[diff]["correct"] += ok; by_diff[diff]["total"] += 1
        results.append({"sample_id": r["meta"]["sample_id"], "pred": pred, "gold": gold, "ok": ok, "raw": out_text})

    tag = "kanana-v-3b" + ("+adapter" if args.adapter else "(기본)")
    print(f"\n===== 결과 [{tag}] (찍기=25%) =====")
    print(f"전체: {acc(overall):.1%}  ({overall['correct']}/{overall['total']})")
    print("[하위카테고리]")
    for k in sorted(by_sub):
        print(f"  {k:12} {acc(by_sub[k]):.1%}")
    print("[난이도]")
    for k in ["easy", "medium", "hard"]:
        if k in by_diff:
            print(f"  {k:8} {acc(by_diff[k]):.1%}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"summary": {"tag": tag, "overall": acc(overall),
                               "by_subcategory": {k: acc(v) for k, v in by_sub.items()},
                               "by_difficulty": {k: acc(v) for k, v in by_diff.items()},
                               "n": overall["total"]},
                   "results": results},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
