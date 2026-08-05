"""
STEP 4-a — STEP 2 합성 이미지(360장)를 LLaMA-Factory VQA(sharegpt) 학습 포맷으로 변환.

- 태스크: 조리 장면 3분류 (normal / smoke / fire).
- 이미지 1장/샘플, 프롬프트는 평가와 동일하게 쓴다(학습·평가 프롬프트 일치가 중요).
- 평가는 합성이 아니라 실 홀드아웃(holdout_v0)으로 → 합성은 전부 학습에 사용.

출력: data/synth_vqa/train.json + dataset_info.json (dataset 이름: cooking_synth_train)
사용: uv run python scripts/step4_make_vqa.py
"""
import json
import os
import random

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH = os.path.join(HERE, "data", "synth", "labels.jsonl")
OUT_DIR = os.path.join(HERE, "data", "synth_vqa")

PROMPT = ("You are a kitchen safety monitor watching a stovetop. "
          "Classify the scene with exactly one word: "
          "'normal' (safe cooking; steam, frying, and food are fine), "
          "'smoke' (dangerous smoke such as overheating oil), or "
          "'fire' (visible flames). Answer:")


def main():
    recs = []
    with open(SYNTH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            recs.append({
                "messages": [
                    {"role": "user", "content": "<image>" + PROMPT},
                    {"role": "assistant", "content": d["label"]},
                ],
                "images": [d["path"]],
            })
    random.Random(0).shuffle(recs)

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(recs, open(os.path.join(OUT_DIR, "train.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    dataset_info = {
        "cooking_synth_train": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant"},
        }
    }
    json.dump(dataset_info, open(os.path.join(OUT_DIR, "dataset_info.json"), "w",
                                 encoding="utf-8"), ensure_ascii=False, indent=2)

    from collections import Counter
    c = Counter(r["messages"][1]["content"] for r in recs)
    print(f"변환 완료: {len(recs)}개  {dict(c)}")
    print(f"출력: {os.path.relpath(OUT_DIR, HERE)}/train.json + dataset_info.json")
    print(f"프롬프트: {PROMPT[:60]}...")


if __name__ == "__main__":
    main()
