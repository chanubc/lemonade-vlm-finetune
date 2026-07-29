"""
Qwen2.5-VL을 LEMONADE(Perception/Reasoning) test 세트로 평가한다.

논문과 동일한 방식:
  - 프레임 + 논문 프롬프트 입력, greedy 디코딩(T=0)
  - 출력에서 A/B/C/D 한 글자 추출 (못 찾으면 "A"로 처리 — 논문 규칙)
  - 정확도(accuracy)를 전체 / 하위카테고리 / 난이도별로 집계

before/after 비교:
  - before(기본 모델):   uv run python scripts/evaluate.py
  - after(파인튜닝):     uv run python scripts/evaluate.py --adapter out/qwen2p5vl-3b-lemonade-lora
  - V가 쓰이는지 대조군:  uv run python scripts/evaluate.py --text-only

주의: 실행하려면 학습 패키지가 필요하다 → `uv sync --group train`
      프레임이 먼저 추출돼 있어야 한다 → `uv run python scripts/extract_frames.py`
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LETTERS = ["A", "B", "C", "D"]


def parse_choice(text: str) -> str:
    """모델 출력에서 첫 A/B/C/D를 뽑는다. 없으면 'A'(논문 규칙, 무작위와 동등)."""
    m = re.search(r"[ABCD]", text.strip().upper())
    return m.group(0) if m else "A"


def strip_image_tokens(content: str) -> str:
    return content.replace("<image>", "")


def build_messages(record, text_only: bool):
    """변환 JSON 레코드 → Qwen 채팅 형식으로 재구성."""
    prompt = strip_image_tokens(record["messages"][0]["content"])
    if text_only:
        content = [{"type": "text", "text": prompt}]
    else:
        content = [{"type": "image", "image": "file://" + os.path.join(HERE, p)}
                   for p in record["images"]]
        content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def frames_exist(record) -> bool:
    return all(os.path.exists(os.path.join(HERE, p)) for p in record["images"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA 어댑터 경로(파인튜닝 후)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="앞 N개만(빠른 점검)")
    ap.add_argument("--text-only", action="store_true",
                    help="이미지 없이 텍스트만 → V가 쓰이는지 확인용 대조군")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-pixels", type=int, default=None,
                    help="이미지 최대 픽셀(VRAM 절약, 예: 200704 = 448*448)")
    ap.add_argument("--out", default=None, help="문항별 결과 저장 경로(json)")
    args = ap.parse_args()

    # 무거운 import는 여기서 (스크립트 열람만 할 땐 torch 없어도 됨)
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm

    data_path = os.path.join(HERE, "data", "converted", f"{args.split}.json")
    records = json.load(open(data_path, encoding="utf-8"))

    if not args.text_only:
        before = len(records)
        records = [r for r in records if frames_exist(r)]
        if len(records) < before:
            print(f"프레임 없는 {before - len(records)}개 제외 → 평가 대상 {len(records)}개")
    if args.limit:
        records = records[: args.limit]

    print(f"모델 로드: {args.model}" + (f" + LoRA({args.adapter})" if args.adapter else ""))
    proc_kwargs = {}
    if args.max_pixels:
        proc_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model, **proc_kwargs)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # 집계용 카운터
    def acc(d):  # {"correct":x,"total":y} -> 비율
        return d["correct"] / d["total"] if d["total"] else 0.0

    overall = {"correct": 0, "total": 0}
    by_sub, by_diff = {}, {}
    results = []

    for r in tqdm(records, desc="평가"):
        messages = build_messages(r, args.text_only)
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        images, videos = (None, None) if args.text_only else process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos,
                           padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        out_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        pred = parse_choice(out_text)
        gold = r["messages"][1]["content"].strip().upper()
        ok = int(pred == gold)

        sub = r["meta"]["subcategory"]
        diff = r["meta"]["difficulty"]
        overall["total"] += 1; overall["correct"] += ok
        by_sub.setdefault(sub, {"correct": 0, "total": 0})
        by_diff.setdefault(diff, {"correct": 0, "total": 0})
        by_sub[sub]["correct"] += ok; by_sub[sub]["total"] += 1
        by_diff[diff]["correct"] += ok; by_diff[diff]["total"] += 1
        results.append({"sample_id": r["meta"]["sample_id"], "pred": pred,
                        "gold": gold, "ok": ok, "raw": out_text})

    # 리포트
    tag = "TEXT-ONLY(대조군)" if args.text_only else ("파인튜닝" if args.adapter else "기본모델")
    print(f"\n===== 결과 [{tag}] (찍기=25%) =====")
    print(f"전체 정확도: {acc(overall):.1%}  ({overall['correct']}/{overall['total']})")
    print("\n[하위카테고리별]")
    for k in sorted(by_sub):
        print(f"  {k:12} {acc(by_sub[k]):.1%}  ({by_sub[k]['correct']}/{by_sub[k]['total']})")
    print("\n[난이도별]")
    for k in ["easy", "medium", "hard"]:
        if k in by_diff:
            print(f"  {k:8} {acc(by_diff[k]):.1%}  ({by_diff[k]['correct']}/{by_diff[k]['total']})")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        summary = {"tag": tag, "overall": acc(overall),
                   "by_subcategory": {k: acc(v) for k, v in by_sub.items()},
                   "by_difficulty": {k: acc(v) for k, v in by_diff.items()},
                   "n": overall["total"]}
        json.dump({"summary": summary, "results": results},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n상세 결과 저장: {args.out}")


if __name__ == "__main__":
    main()
