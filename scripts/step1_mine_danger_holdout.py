"""
STEP 1 — 위험 쪽 홀드아웃(평가용) 후보를 공개 실화재셋에서 채굴한다.

목적: "조리 위험" 홀드아웃에 쓸 실데이터 후보를 최대한 조리/실내 도메인에 가깝게 뽑기.
      (D-Fire는 대부분 야외 산불이므로, 근접·실내 화재만 골라 도메인 간극을 줄인다.)

파이프라인:
  1) D-Fire test 스플릿에서 화재(class 1) 이미지만 추출
     - test만 사용 → 나중에 train을 학습에 써도 홀드아웃 누수 없음
  2) 불꽃 박스 면적으로 랭킹: 큰 불 = 근접(주방 화재 유사), 작은 불 = 먼 산불(버림)
  3) 상위 pre개를 VLM(기본 Qwen2.5-VL-3B)으로 indoor/outdoor 태깅
  4) 후보 이미지 + manifest + 몽타주 저장 (사람이 최종 큐레이션하도록)

주의: 이건 "후보"다. 홀드아웃은 반드시 사람이 눈으로 확인해 확정해야 한다.
      D-Fire엔 진짜 "주방" 화재가 거의 없다 → 이건 도메인 근사치이고,
      진짜 주방 화재(소방서 실연영상·발연 촬영 등)로 보강해야 한다.

사용:
  uv run --group train python scripts/step1_mine_danger_holdout.py
  uv run --group train python scripts/step1_mine_danger_holdout.py --pre 250 --no-vlm
"""
import argparse
import io
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "badsaarow/d-fire"
TEST_SHARDS = [
    "data/test-00000-of-00003.parquet",
    "data/test-00001-of-00003.parquet",
    "data/test-00002-of-00003.parquet",
]
OUT_DIR = os.path.join(HERE, "data", "holdout_real", "danger_candidates")
SCENE_PROMPT = ("Is this fire indoors (kitchen, stove, room, building interior) "
                "or outdoors (forest, hill, street, landscape)? "
                "Answer only 'indoor' or 'outdoor'.")


def max_fire_area(label: str) -> float:
    """YOLO 라벨에서 class 1(불) 박스의 최대 면적(정규화 w*h). 없으면 0."""
    best = 0.0
    for line in label.strip().splitlines():
        p = line.strip().split()
        if len(p) == 5 and p[0] == "1":
            try:
                best = max(best, float(p[3]) * float(p[4]))
            except ValueError:
                pass
    return best


def parse_scene(text: str) -> str:
    t = text.strip().lower()
    ii, oi = t.find("indoor"), t.find("outdoor")
    if ii == -1 and oi == -1:
        return "unknown"
    if oi == -1:
        return "indoor"
    if ii == -1:
        return "outdoor"
    return "indoor" if ii < oi else "outdoor"


def make_montage(paths, out_path, cols=6, thumb=256):
    from PIL import Image
    if not paths:
        return
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb, rows * thumb), (20, 20, 20))
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        im.thumbnail((thumb, thumb))
        x = (i % cols) * thumb + (thumb - im.width) // 2
        y = (i // cols) * thumb + (thumb - im.height) // 2
        canvas.paste(im, (x, y))
    canvas.save(out_path, "JPEG", quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", type=int, default=200,
                    help="박스 크기 상위 몇 장을 VLM 태깅 대상으로 삼을지")
    ap.add_argument("--shards", type=int, default=3, help="test 샤드 수(1~3)")
    ap.add_argument("--no-vlm", action="store_true", help="VLM 태깅 생략(박스크기 랭킹만)")
    ap.add_argument("--montage-n", type=int, default=24)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    import pandas as pd
    from PIL import Image

    # 1) 샤드 로드 + 화재 이미지 추출
    frames = []
    for shard in TEST_SHARDS[: args.shards]:
        print(f"다운로드(캐시): {shard}")
        path = hf_hub_download(REPO, shard, repo_type="dataset")
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    print(f"test 총 이미지: {len(df)}")

    df["fire_area"] = df["label"].apply(max_fire_area)
    fire = df[df["fire_area"] > 0].copy()
    fire = fire.sort_values("fire_area", ascending=False).reset_index(drop=True)
    print(f"화재 이미지: {len(fire)}  (근접 화재 우선 랭킹)")

    pre = fire.head(args.pre).reset_index(drop=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 이미지 저장
    print(f"상위 {len(pre)}장 저장 중...")
    saved = []
    for _, row in pre.iterrows():
        img_field = row["image"]
        data = img_field["bytes"] if isinstance(img_field, dict) else img_field
        img = Image.open(io.BytesIO(data)).convert("RGB")
        fname = str(row["filename"])
        fpath = os.path.join(OUT_DIR, fname)
        img.save(fpath, "JPEG", quality=95)
        saved.append({"path": os.path.relpath(fpath, HERE).replace("\\", "/"),
                      "filename": fname, "fire_area": round(float(row["fire_area"]), 5),
                      "scene": None})

    # 3) VLM indoor/outdoor 태깅
    if not args.no_vlm:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        from tqdm import tqdm

        print("VLM 로드(scene 태깅용): Qwen2.5-VL-3B")
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", max_pixels=1003520)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto")
        model.eval()

        for rec in tqdm(saved, desc="scene 태깅"):
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": "file://" + os.path.join(HERE, rec["path"])},
                {"type": "text", "text": SCENE_PROMPT}]}]
            text = processor.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            inp = processor(text=[text], images=imgs, videos=vids,
                            padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=8, do_sample=False)
            out = processor.batch_decode(gen[:, inp.input_ids.shape[1]:],
                                         skip_special_tokens=True)[0]
            rec["scene"] = parse_scene(out)

    # 매니페스트
    mpath = os.path.join(OUT_DIR, "manifest.jsonl")
    with open(mpath, "w", encoding="utf-8") as f:
        for rec in saved:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 요약 + 몽타주(실내 우선)
    indoor = [r for r in saved if r["scene"] == "indoor"]
    outdoor = [r for r in saved if r["scene"] == "outdoor"]
    print("\n===== 위험 홀드아웃 후보 채굴 결과 =====")
    print(f"후보 총: {len(saved)}장  (박스크기 상위 = 근접 화재)")
    if not args.no_vlm:
        print(f"  indoor(주방 유사): {len(indoor)}장")
        print(f"  outdoor(먼 화재) : {len(outdoor)}장")
        print(f"  unknown          : {len(saved)-len(indoor)-len(outdoor)}장")

    # 몽타주: 실내 후보 우선, 부족하면 근접 화재로 채움
    montage_pool = (indoor or saved)[: args.montage_n]
    mont_paths = [os.path.join(HERE, r["path"]) for r in montage_pool]
    mont_out = os.path.join(OUT_DIR, "montage.jpg")
    make_montage(mont_paths, mont_out, cols=6)
    print(f"\n저장:")
    print(f"  이미지     : {os.path.relpath(OUT_DIR, HERE)}/*.jpg ({len(saved)}장)")
    print(f"  매니페스트 : {os.path.relpath(mpath, HERE)} (scene/fire_area 포함)")
    print(f"  몽타주     : {os.path.relpath(mont_out, HERE)} (후보 {len(montage_pool)}장 미리보기)")
    print("\n다음: 몽타주/이미지를 사람이 확인 → 진짜 위험(주방 유사)만 홀드아웃으로 확정.")
    print("      D-Fire엔 실주방 화재가 드묾 → 소방서 실연영상·발연 촬영으로 보강 필요.")


if __name__ == "__main__":
    main()
