"""
STEP 1 마무리 — NIST 프레임 크롭 + 어시스트 라벨링 → 위험 홀드아웃 v0.

배경(실측으로 확인함):
  NIST FCD 영상은 HRR(열방출률) 곡선이 화면 전체에 겹쳐 그려진다.
  - 정지된 격자·제목·로고·축 라벨 = 모든 프레임 동일 → 라벨과 무관(무해), 잘라내면 더 깨끗.
  - 불날 때만 튀는 "현재시각 스파이크" = 화재와 상관 → 누수 위험.
    · cornoil 2종: 스파이크가 오른쪽 끝 → 오른쪽 크롭으로 제거됨 → 평가 홀드아웃으로 OK.
    · stovetop/kitchen 3종: 스파이크가 화면 중앙(불과 겹침) → 제거 불가 → 평가 부적합.
      → 이들은 "화재 참고(reference)"로만 분리 보관(학습 후보/외형 참고), 평가엔 미사용.

어시스트 라벨:
  각 영상의 정상/발연(smoke)/발화(fire) 구간을 시각 검수로 잡아 "제안 라벨"을 부여.
  사람은 montage를 보고 label_confirmed 칸만 채우면 됨(제안이 맞으면 그대로 복사).
  ※ 모델로 라벨하지 않는다(홀드아웃 라벨은 사람이 확정 — 순환/누수 방지).

입력: data/holdout_real/frames/<vid>/*.jpg (1fps 프레임, 이미 추출됨)
출력:
  data/holdout_real/holdout_v0/{normal,smoke,fire}/*.jpg   (cornoil, 평가용)
  data/holdout_real/fire_reference/<vid>/*.jpg              (나머지, 참고용)
  data/holdout_real/holdout_v0_manifest.csv                 (제안 라벨 + 확정 칸)
  data/holdout_real/review_<vid>.jpg                        (제안 라벨 붙은 검수용 몽타주)

사용: uv run python scripts/step1_crop_label_holdout.py
"""
import csv
import glob
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_DIR = os.path.join(HERE, "data", "holdout_real", "frames")
BASE = os.path.join(HERE, "data", "holdout_real")

# 영상별 크롭 박스 (제목·y축라벨·x축·로고 제거; cornoil은 오른쪽 스파이크까지 제거)
CROP = {
    "nist_cornoil_alumipan":     (145, 95, 1560, 1000),   # 1920x1080
    "nist_cornoil_calphalon":    (145, 95, 1560, 1000),
    "nist_stovetop_massloss13":  (110, 45, 1180, 620),    # 1280x720
    "nist_stovetop_massloss14c": (110, 45, 1180, 620),
    "nist_fcd_kitchen_room_fire":(150, 95, 1770, 1015),   # 1920x1080
}

# 평가 홀드아웃으로 써도 되는 영상(스파이크 제거됨) vs 참고용
EVAL_OK = {
    "nist_cornoil_alumipan": True,
    "nist_cornoil_calphalon": True,
    "nist_stovetop_massloss13": False,
    "nist_stovetop_massloss14c": False,
    "nist_fcd_kitchen_room_fire": False,
}

# 제안 라벨 구간: (시작초, 끝초 포함, 라벨). 시각 검수 기반, 사람이 확정.
LABELS = {
    "nist_cornoil_alumipan":     [(0, 6, "normal"), (7, 9, "smoke"), (10, 13, "fire"), (14, 99, "normal")],
    "nist_cornoil_calphalon":    [(0, 6, "normal"), (7, 7, "smoke"), (8, 13, "fire"), (14, 99, "normal")],
    "nist_stovetop_massloss13":  [(0, 14, "normal"), (15, 22, "fire"), (23, 99, "normal")],
    "nist_stovetop_massloss14c": [(0, 10, "normal"), (11, 12, "smoke"), (13, 20, "fire"), (21, 99, "normal")],
    "nist_fcd_kitchen_room_fire":[(0, 3, "normal"), (4, 11, "fire"), (12, 99, "smoke")],
}

LICENSE = "US government (NIST NFRL FCD) - public domain"


def suggest_label(vid, t):
    for a, b, lab in LABELS[vid]:
        if a <= t <= b:
            return lab
    return "normal"


def frame_t(path):
    # "..._t0008.jpg" 또는 "..._t0008_000123.jpg" 둘 다 처리
    return int(os.path.basename(path).split("_t")[1].split("_")[0].split(".")[0])


def review_montage(items, out_path, cols=6, thumb=260):
    from PIL import Image, ImageDraw
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb, rows * thumb), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    color = {"normal": (120, 230, 120), "smoke": (240, 210, 90), "fire": (250, 90, 70)}
    for i, (img, t, lab) in enumerate(items):
        im = img.copy(); im.thumbnail((thumb, thumb))
        x = (i % cols) * thumb + (thumb - im.width) // 2
        y = (i // cols) * thumb + (thumb - im.height) // 2
        canvas.paste(im, (x, y))
        d.text(((i % cols) * thumb + 4, (i // cols) * thumb + 4),
               f"{t}s {lab}", fill=color.get(lab, (255, 255, 255)))
    canvas.save(out_path, "JPEG", quality=90)


def main():
    from PIL import Image
    rows = []
    counts = {}
    for vid, box in CROP.items():
        files = sorted(f for f in glob.glob(os.path.join(FRAME_DIR, vid, "*.jpg"))
                       if not os.path.basename(f).startswith("_"))
        if not files:
            print(f"[건너뜀] 프레임 없음: {vid}")
            continue
        eval_ok = EVAL_OK[vid]
        review = []
        for f in files:
            t = frame_t(f)
            lab = suggest_label(vid, t)
            img = Image.open(f).convert("RGB").crop(box)
            if eval_ok:
                out_dir = os.path.join(BASE, "holdout_v0", lab)
                dset = "holdout_v0"
            else:
                out_dir = os.path.join(BASE, "fire_reference", vid)
                dset = "reference"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{vid}_t{t:04d}.jpg")
            img.save(out_path, "JPEG", quality=95)
            rows.append({
                "set": dset, "video_id": vid, "t_sec": t,
                "frame": os.path.relpath(out_path, HERE).replace("\\", "/"),
                "label_suggested": lab, "label_confirmed": "",
                "eval_holdout_ok": eval_ok, "license": LICENSE,
            })
            counts.setdefault((dset, lab), 0)
            counts[(dset, lab)] += 1
            review.append((img, t, lab))
        review_montage(review, os.path.join(BASE, f"review_{vid}.jpg"))

    cols = ["set", "video_id", "t_sec", "frame", "label_suggested",
            "label_confirmed", "eval_holdout_ok", "license"]
    with open(os.path.join(BASE, "holdout_v0_manifest.csv"), "w",
              encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols); w.writeheader(); w.writerows(rows)

    print("===== 크롭 + 어시스트 라벨링 완료 =====")
    print("[holdout_v0 — cornoil, 평가용]")
    for (dset, lab), n in sorted(counts.items()):
        if dset == "holdout_v0":
            print(f"  {lab:7} {n}장")
    print("[fire_reference — 나머지, 참고용(평가 미사용)]")
    for (dset, lab), n in sorted(counts.items()):
        if dset == "reference":
            print(f"  {lab:7} {n}장")
    print(f"\n총 {len(rows)}장")
    print(f"매니페스트: {os.path.relpath(os.path.join(BASE,'holdout_v0_manifest.csv'), HERE)}")
    print("검수: data/holdout_real/review_<vid>.jpg 를 보고 label_confirmed 칸을 채우세요.")


if __name__ == "__main__":
    main()
