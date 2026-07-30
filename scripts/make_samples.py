"""
데모용 샘플 세트를 samples/ 에 만든다 (git에 커밋할 소량).

전체 프레임(수 GB)은 .gitignore 되지만, 이 스크립트가 고른 몇 개 문항의
프레임(8장)과 질문/정답을 samples/ 에 복사·기록해, clone 후 바로 데모를 돌려볼 수 있게 한다.

실행: uv run python scripts/make_samples.py
출력: samples/frames/<sid>/frame_00..07.jpg,  samples/README.md
"""
import ast
import json
import os
import shutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_PER_SUB = 3  # 하위카테고리별 몇 개


def main():
    test = json.load(open(os.path.join(HERE, "data", "converted", "test.json"), encoding="utf-8"))
    out_root = os.path.join(HERE, "samples")
    frames_out = os.path.join(out_root, "frames")
    os.makedirs(frames_out, exist_ok=True)

    picked, per_sub = [], {}
    for r in test:
        sub = r["meta"]["subcategory"]
        if per_sub.get(sub, 0) >= N_PER_SUB:
            continue
        # 프레임이 실제로 있어야 함
        if not all(os.path.exists(os.path.join(HERE, p)) for p in r["images"]):
            continue
        picked.append(r)
        per_sub[sub] = per_sub.get(sub, 0) + 1
        if all(per_sub.get(s, 0) >= N_PER_SUB for s in ["Perception", "Reasoning"]):
            break

    lines = [
        "# 데모 샘플 (git 포함)",
        "",
        "clone 후 바로 open-webui에서 테스트할 수 있는 소량 예제.",
        "각 예제는 `samples/frames/<폴더>/` 의 **8장을 모두 첨부** + 아래 질문을 넣고,",
        "모델이 **정답 글자**를 맞히는지 확인한다. (데모 실행법: `docs/DEMO.md`)",
        "",
        "> 어댑터는 git에 없다 → 서버 실행 시 HF에서 받는다:",
        "> `uv run python scripts/serve_openai.py --adapter chanubc/Qwen2.5-VL-3B-LEMONADE-LoRA`",
        "",
    ]
    for i, r in enumerate(picked, 1):
        sid = r["meta"]["sample_id"]
        sub = r["meta"]["subcategory"]
        diff = r["meta"]["difficulty"]
        gold = r["messages"][1]["content"]
        prompt = r["messages"][0]["content"].replace("<image>", "")
        # 프레임 복사
        dst = os.path.join(frames_out, sid)
        os.makedirs(dst, exist_ok=True)
        for p in r["images"]:
            shutil.copy2(os.path.join(HERE, p), os.path.join(dst, os.path.basename(p)))
        lines += [
            f"## 예제 {i} — {sub} ({diff}) · 정답: **{gold}**",
            f"이미지: `samples/frames/{sid}/` (8장)",
            "",
            "```",
            prompt,
            "```",
            "",
        ]

    with open(os.path.join(out_root, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"샘플 {len(picked)}개 생성 → {out_root}")
    print("하위카테고리별:", per_sub)


if __name__ == "__main__":
    main()
