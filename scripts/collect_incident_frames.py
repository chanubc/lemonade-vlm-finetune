"""
조리사고/화재 영상 → 프레임 → 라벨 템플릿  (위험 홀드아웃 실물화 스캐폴드)

무엇을 하나:
  data/holdout_real/sources.json 에 적힌 영상들을 받아서(yt-dlp 또는 직접 다운로드),
  cv2로 프레임을 뽑고(ffmpeg 불필요), 사람이 라벨만 채우면 되는 매니페스트를 만든다.

핵심 규칙(HANDOFF 7절):
  - 홀드아웃 = 평가용 = 실데이터. 여기서 만든 건 "위험 쪽 후보".
  - 누수 방지: 분할은 "영상 단위". 한 영상은 통째로 holdout 또는 train 중 하나로만 간다
    (같은 영상 프레임이 학습·평가 양쪽에 들어가지 않게). split 필드로 명시.
  - 라벨은 사람이 확정: label 칸은 비워두고, label_hint(제작자 의도)만 참고로 채운다.

의존성: yt-dlp, imageio-ffmpeg (uv add --group collect 로 설치됨), opencv(이미 있음)

사용:
  uv run --group train --group collect python scripts/collect_incident_frames.py
  uv run --group train --group collect python scripts/collect_incident_frames.py --fps 2 --only nist_fcd_kitchen
"""
import argparse
import csv
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(HERE, "data", "holdout_real")
SOURCES = os.path.join(BASE, "sources.json")
VIDEO_DIR = os.path.join(BASE, "videos")
FRAME_DIR = os.path.join(BASE, "frames")
LABEL_CSV = os.path.join(BASE, "label_manifest.csv")
LABEL_JSONL = os.path.join(BASE, "label_manifest.jsonl")


def ffmpeg_dir():
    """imageio-ffmpeg가 동봉한 정적 ffmpeg 바이너리 폴더 (yt-dlp가 스트림 병합에 사용)."""
    try:
        import imageio_ffmpeg
        return os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None


def download_ytdlp(url, out_base):
    """yt-dlp로 진행형(progressive) mp4 우선 다운로드. 반환: 실제 저장 경로 or None."""
    import yt_dlp
    opts = {
        "outtmpl": out_base + ".%(ext)s",
        # 프레임 추출만 할 거라 화질은 720p 이하 단일 파일 우선(병합 최소화)
        "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    fd = ffmpeg_dir()
    if fd:
        opts["ffmpeg_location"] = fd
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def download_direct(url, out_path):
    """직접 mp4 URL 다운로드(스트리밍)."""
    import requests
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return out_path


def extract_frames(video_path, out_dir, vid, fps, max_frames):
    """cv2로 fps 간격 프레임 추출. 반환: [(frame_path, t_sec), ...]."""
    import cv2
    import math
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(src_fps / fps)))
    # 발화가 영상 중·후반에 나올 수 있으므로, max_frames로 잘릴 상황이면
    # 앞부분만 뽑지 말고 전체에 고르게 퍼지도록 간격을 키운다.
    if max_frames and total > 0 and (total / step) > max_frames:
        step = max(step, math.ceil(total / max_frames))
    saved = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            t = i / src_fps
            # 파일명에 원본 프레임 index를 붙여 충돌 방지(fps>1이라도 덮어쓰기 없음)
            fpath = os.path.join(out_dir, f"{vid}_t{int(round(t)):04d}_{i:06d}.jpg")
            cv2.imwrite(fpath, frame)
            saved.append((fpath, round(t, 2)))
            if max_frames and len(saved) >= max_frames:
                break
        i += 1
    cap.release()
    return saved, src_fps, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=1.0, help="초당 뽑을 프레임 수")
    ap.add_argument("--max-frames", type=int, default=120, help="영상당 최대 프레임")
    ap.add_argument("--only", default=None, help="특정 source id 하나만")
    ap.add_argument("--sources", default=SOURCES)
    args = ap.parse_args()

    if not os.path.exists(args.sources):
        raise SystemExit(f"소스 목록이 없습니다: {args.sources}\n"
                         "  → data/holdout_real/sources.json 을 먼저 만드세요.")
    sources = json.load(open(args.sources, encoding="utf-8"))
    os.makedirs(VIDEO_DIR, exist_ok=True)

    rows = []
    for s in sources:
        if not s.get("enabled", True):
            continue
        if args.only and s["id"] != args.only:
            continue
        vid = s["id"]
        print(f"\n=== {vid} ({s.get('label_hint','?')}, split={s.get('split','?')}) ===")
        out_base = os.path.join(VIDEO_DIR, vid)
        try:
            existing = [p for p in (out_base + ".mp4", out_base + ".webm", out_base + ".mkv")
                        if os.path.exists(p)]
            if existing:
                vpath = existing[0]
                print(f"  이미 받음: {os.path.basename(vpath)}")
            elif s.get("kind") == "direct":
                print(f"  직접 다운로드: {s['url']}")
                vpath = download_direct(s["url"], out_base + ".mp4")
            else:
                print(f"  yt-dlp 다운로드: {s['url']}")
                vpath = download_ytdlp(s["url"], out_base)
        except Exception as e:
            print(f"  [실패] 다운로드 오류: {e}")
            continue

        try:
            frames, src_fps, total = extract_frames(
                vpath, os.path.join(FRAME_DIR, vid), vid, args.fps, args.max_frames)
        except Exception as e:
            print(f"  [실패] 프레임 추출 오류: {e}")
            continue
        print(f"  원본 {src_fps:.1f}fps, 총 {total}프레임 → {len(frames)}장 추출")

        for fpath, t in frames:
            rows.append({
                "video_id": vid,
                "frame": os.path.relpath(fpath, HERE).replace("\\", "/"),
                "t_sec": t,
                "split": s.get("split", "holdout"),
                "label_hint": s.get("label_hint", ""),
                "label": "",  # 사람이 채움: normal / smoke / fire
                "license": s.get("license", ""),
                "source_url": s.get("url", ""),
            })

    if not rows:
        print("\n추출된 프레임이 없습니다. sources.json의 enabled/url을 확인하세요.")
        return

    os.makedirs(BASE, exist_ok=True)
    cols = ["video_id", "frame", "t_sec", "split", "label_hint", "label",
            "license", "source_url"]
    with open(LABEL_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(LABEL_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n===== 완료 =====")
    print(f"총 {len(rows)}프레임 (영상 {len(set(r['video_id'] for r in rows))}개)")
    print(f"라벨 템플릿: {os.path.relpath(LABEL_CSV, HERE)}  (label 칸을 normal/smoke/fire로 채우세요)")
    print(f"프레임: {os.path.relpath(FRAME_DIR, HERE)}/<video_id>/")


if __name__ == "__main__":
    main()
