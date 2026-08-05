"""
STEP 1b — 오버레이 없는 실주방 화재 영상을 홀드아웃에 라벨 추가.

NIST(오버레이 크롭 필요)와 달리, 소방서/DVIDS 영상은 오버레이가 없어 크롭 불필요.
시간구간 기반 어시스트 라벨(사람 검수) → holdout_v0/{label}/에 vid-접두 파일명으로 추가.
step4_eval.py는 holdout_v0/{fire,smoke,normal}/*.jpg를 glob하므로 자동으로 평가에 포함됨.

깨끗한 스토브탑 영상만 자동 라벨(dvids, northants).
grease_fire_demo(진행자 프레임)·chip_pan_facts(애니메이션)는 혼합이라 제외 → 수동 검토 대상.

사용: uv run python scripts/step1b_label_flat.py
"""
import csv
import glob
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_DIR = os.path.join(HERE, "data", "holdout_real", "frames")
HOLDOUT = os.path.join(HERE, "data", "holdout_real", "holdout_v0")
MANIFEST = os.path.join(HERE, "data", "holdout_real", "holdout_v1_manifest.csv")

# 시간구간(초) 라벨 — 몽타주 검수 기반, 사람이 label_confirmed로 확정.
# 타이틀/엔드카드/애니메이션 구간은 라벨 dict에서 제외해 자동으로 버려짐.
LABELS = {
    "dvids_grease_fire": [(5, 43, "normal"), (44, 107, "fire")],           # >=108 엔드카드 제외
    "northants_chip_pan_guidance": [(0, 13, "normal"), (14, 30, "fire"), (31, 33, "smoke")],
}
LICENSE = {
    "dvids_grease_fire": "US government (DVIDS) - public domain",
    "northants_chip_pan_guidance": "소방서 공익 공개영상 - 연구/평가 내부용",
}


def frame_t(path):
    return int(os.path.basename(path).split("_t")[1].split("_")[0].split(".")[0])


def label_of(vid, t):
    for a, b, lab in LABELS[vid]:
        if a <= t <= b:
            return lab
    return None  # 구간 밖(타이틀 등) → 버림


def main():
    from PIL import Image
    rows = []
    counts = {}
    for vid, ranges in LABELS.items():
        files = sorted(glob.glob(os.path.join(FRAME_DIR, vid, "*.jpg")))
        for f in files:
            t = frame_t(f)
            lab = label_of(vid, t)
            if lab is None:
                continue
            out_dir = os.path.join(HOLDOUT, lab)
            os.makedirs(out_dir, exist_ok=True)
            dst = os.path.join(out_dir, f"{vid}_t{t:04d}.jpg")
            Image.open(f).convert("RGB").save(dst, "JPEG", quality=95)
            rows.append({"video_id": vid, "t_sec": t,
                         "frame": os.path.relpath(dst, HERE).replace("\\", "/"),
                         "label_suggested": lab, "label_confirmed": "",
                         "eval_holdout_ok": True, "license": LICENSE[vid]})
            counts[(vid, lab)] = counts.get((vid, lab), 0) + 1

    cols = ["video_id", "t_sec", "frame", "label_suggested", "label_confirmed",
            "eval_holdout_ok", "license"]
    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    print("===== 오버레이 없는 영상 라벨 추가 =====")
    for (vid, lab), n in sorted(counts.items()):
        print(f"  {vid:32} {lab:7} {n}장")
    print(f"총 {len(rows)}장 → holdout_v0/에 추가")
    print(f"매니페스트: {os.path.relpath(MANIFEST, HERE)}")
    print("검수: data/holdout_real/_new_sources_overview.jpg 참고, label_confirmed 채우기.")
    print("주의: grease_fire_demo(진행자)·chip_pan_facts(애니메이션)는 혼합 → 수동 검토 후 추가.")


if __name__ == "__main__":
    main()
