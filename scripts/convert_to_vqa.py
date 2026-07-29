"""
LEMONADE split(parquet) → Qwen2.5-VL 파인튜닝용 JSON 변환.

이번 범위: Perception / Reasoning 두 하위카테고리만.
  (나이·속도·각도처럼 프레임 몇 장으로 원리적으로 못 푸는 유형은 제외)

출력 형식: LLaMA-Factory의 멀티모달 sharegpt 포맷.
  각 레코드 = {"messages": [...], "images": [프레임 경로들]}
  본문 앞에 이미지 수만큼 "<image>" 토큰을 붙인다(토큰 수 == images 길이).

주의: 이 단계에서는 프레임(이미지)이 아직 없어도 된다.
  각 QA가 사용할 프레임 "경로 규칙"만 기록해 둔다.
  실제 이미지는 다음 단계(비디오 다운로드 → 프레임 추출)에서 생성한다.

프레임 경로 규칙:
  data/frames/{sample_id}/frame_{i:02d}.jpg
  sample_id = "{Clip}_{int(Start)}_{int(End)}"   (한 QA의 문맥 구간을 유일하게 식별)
"""
import argparse
import ast
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROMPT_TEMPLATE = (
    "Answer the following multiple-choice question using the given images.\n"
    "Question: {question}\n"
    "Choices:\n"
    "A. {a}\n"
    "B. {b}\n"
    "C. {c}\n"
    "D. {d}\n"
    "Respond only with the letter of the correct answer."
)

LETTERS = ["A", "B", "C", "D"]


def sample_id(clip: str, start: float, end: float) -> str:
    return f"{clip}_{int(start)}_{int(end)}"


def frame_paths(sid: str, n_frames: int, frames_root: str) -> list:
    """학습 config에서 참조할 상대 경로(POSIX 슬래시)."""
    d = f"{frames_root}/{sid}"
    return [f"{d}/frame_{i:02d}.jpg" for i in range(n_frames)]


def build_record(row, n_frames: int, frames_root: str, require_frames: bool):
    # 1) 보기 파싱: "['Hold','Read',...]" (문자열) → 실제 리스트
    try:
        choices = ast.literal_eval(row["Answers"])
    except (ValueError, SyntaxError):
        return None, "answers_parse_error"
    if not isinstance(choices, list) or len(choices) != 4:
        return None, "not_4_choices"

    # 2) 정답 글자 검증 (Answers 순서에 A/B/C/D가 대응)
    correct = str(row["Correct Answer"]).strip()
    if correct not in LETTERS:
        return None, "bad_correct_answer"

    sid = sample_id(row["Clip"], row["Start"], row["End"])
    imgs = frame_paths(sid, n_frames, frames_root)

    # 프레임 실제 존재를 요구하는 경우(프레임 추출 후 사용)
    if require_frames:
        abs_imgs = [os.path.join(HERE, p) for p in imgs]
        if not all(os.path.exists(p) for p in abs_imgs):
            return None, "frames_missing"

    text = "".join(["<image>"] * n_frames) + PROMPT_TEMPLATE.format(
        question=str(row["Question"]).strip(),
        a=choices[0], b=choices[1], c=choices[2], d=choices[3],
    )

    record = {
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": correct},
        ],
        "images": imgs,
        # 평가/디버깅용 메타 (LLaMA-Factory는 무시함)
        "meta": {
            "sample_id": sid,
            "subcategory": row["Subcategory"],
            "difficulty": row["Difficulty"],
            "clip": row["Clip"],
            "start": int(row["Start"]),
            "end": int(row["End"]),
        },
    }
    return record, "ok"


def convert_split(split: str, args):
    in_path = os.path.join(HERE, "data", "splits", f"{split}.parquet")
    df = pd.read_parquet(in_path)
    df = df[df["Subcategory"].isin(args.subcats)].reset_index(drop=True)

    records, reasons = [], {}
    for _, row in df.iterrows():
        rec, why = build_record(row, args.frames, args.frames_root, args.require_frames)
        reasons[why] = reasons.get(why, 0) + 1
        if rec is not None:
            records.append(rec)

    out_dir = os.path.join(HERE, "data", "converted")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)

    print(f"[{split}] 후보 {len(df):,}개 중 변환 {len(records):,}개 → {out_path}")
    skipped = {k: v for k, v in reasons.items() if k != "ok"}
    if skipped:
        print(f"        건너뜀: {skipped}")
    return len(records)


def write_dataset_info(args):
    """LLaMA-Factory가 데이터셋을 인식하도록 dataset_info.json 생성."""
    info = {}
    for split in args.splits:
        info[f"lemonade_{split}"] = {
            "file_name": f"{split}.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    path = os.path.join(HERE, "data", "converted", "dataset_info.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"\nLLaMA-Factory용 dataset_info.json 저장: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--subcats", nargs="+", default=["Perception", "Reasoning"])
    ap.add_argument("--frames", type=int, default=8,
                    help="QA당 프레임 수 (12GB VRAM이면 4 권장)")
    ap.add_argument("--frames-root", default="data/frames",
                    help="프레임 이미지가 저장될 루트(상대경로)")
    ap.add_argument("--require-frames", action="store_true",
                    help="프레임이 실제로 있는 QA만 남긴다(프레임 추출 후 사용)")
    args = ap.parse_args()

    print(f"대상 하위카테고리: {args.subcats}, 프레임 수: {args.frames}")
    total = 0
    for split in args.splits:
        total += convert_split(split, args)
    write_dataset_info(args)
    print(f"\n총 변환: {total:,}개")


if __name__ == "__main__":
    main()
