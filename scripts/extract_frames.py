"""
LEMONADE 비디오에서 QA별 프레임을 추출한다.

핵심 아이디어(61GB 통째 다운로드 회피):
  비디오는 HuggingFace에 5개의 큰 zip으로 묶여 있다(videos_batch_0~4.zip).
  각 zip의 "목차(중앙 디렉터리)"만 HTTP range로 읽어 어떤 영상이 어느 zip에
  있는지 색인을 만든 뒤, 필요한 mp4만 골라 임시로 내려받아 프레임을 뽑고 삭제한다.
  → 디스크 사용량은 "한 번에 영상 1개(~1GB) + 추출된 작은 jpg들"뿐.

프레임 선택:
  각 QA의 Start~End(프레임 번호) 구간을 N등분한 위치에서 프레임을 뽑는다
  (논문과 동일한 "균등 샘플링"). fps는 필요 없다 — 절대 프레임 번호를 그대로 쓴다.

출력:
  data/frames/{sample_id}/frame_00.jpg ... frame_{N-1}.jpg
  sample_id = "{Clip}_{start}_{end}"  (convert_to_vqa.py와 동일 규칙)
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import requests
from remotezip import RemoteZip

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _auth_session():
    """HF 토큰(hf auth login으로 저장된 값)을 실은 requests 세션.
    remotezip은 날것의 HTTP 요청이라, 이 세션으로 Authorization 헤더를 붙여야
    로그인된 요청이 되어 다운로드 속도 제한이 풀린다."""
    s = requests.Session()
    try:
        from huggingface_hub import get_token
        tok = get_token()
        if tok:
            s.headers.update({"Authorization": f"Bearer {tok}"})
    except Exception:
        pass
    return s
FRAMES_ROOT = os.path.join(HERE, "data", "frames")
VIDEO_TMP = os.path.join(HERE, "data", "videos", "_tmp")
INDEX_PATH = os.path.join(HERE, "data", "videos", "video_index.json")

BATCH_URLS = [
    f"https://huggingface.co/datasets/amathislab/LEMONADE/resolve/main/videos_batch_{i}.zip"
    for i in range(5)
]


def build_index() -> dict:
    """clip 이름 → {batch_url, member} 색인. 각 zip 목차만 읽으므로 가볍다."""
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, encoding="utf-8") as f:
            return json.load(f)
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    index = {}
    for url in BATCH_URLS:
        print(f"목차 읽는 중: {url.split('/')[-1]}")
        with RemoteZip(url, session=_auth_session()) as z:
            for name in z.namelist():
                if not name.endswith("_hololens.mp4"):
                    continue
                clip = os.path.basename(name)[: -len("_hololens.mp4")]
                index[clip] = {"batch_url": url, "member": name}
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"색인 저장: {INDEX_PATH} ({len(index)}개 clip)")
    return index


def load_windows(splits) -> dict:
    """변환된 JSON에서 clip별로 뽑아야 할 창(window)을 모은다.
    반환: {clip: {sample_id: {"start","end","n","images"}}}"""
    by_clip = {}
    for split in splits:
        fp = os.path.join(HERE, "data", "converted", f"{split}.json")
        if not os.path.exists(fp):
            continue
        for r in json.load(open(fp, encoding="utf-8")):
            m = r["meta"]
            clip = m["clip"]
            by_clip.setdefault(clip, {})[m["sample_id"]] = {
                "start": m["start"], "end": m["end"],
                "n": len(r["images"]), "images": r["images"],
            }
    return by_clip


def already_done(win) -> bool:
    return all(os.path.exists(os.path.join(HERE, p)) for p in win["images"])


def extract_from_video(video_path: str, windows: dict, max_side: int, quality: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ! 영상 열기 실패: {video_path}")
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10**9
    saved = 0
    # start 기준 정렬해 순차 탐색에 유리하게
    for sid, win in sorted(windows.items(), key=lambda kv: kv[1]["start"]):
        if already_done(win):
            continue
        out_dir = os.path.join(HERE, "data", "frames", sid)
        os.makedirs(out_dir, exist_ok=True)
        idxs = np.linspace(win["start"], win["end"], win["n"])
        idxs = [int(min(max(0, round(x)), total - 1)) for x in idxs]
        for i, fidx in enumerate(idxs):
            out_p = os.path.join(out_dir, f"frame_{i:02d}.jpg")
            if os.path.exists(out_p):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok:
                continue
            if max_side and max(frame.shape[:2]) > max_side:
                h, w = frame.shape[:2]
                s = max_side / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)),
                                   interpolation=cv2.INTER_AREA)
            cv2.imwrite(out_p, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved += 1
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--clips", nargs="+", default=None,
                    help="특정 clip만 처리(디버그용)")
    ap.add_argument("--limit-clips", type=int, default=None,
                    help="앞에서 N개 clip만 처리(파이프라인 검증용)")
    ap.add_argument("--max-side", type=int, default=512,
                    help="프레임 긴 변 최대 픽셀(0이면 원본 유지). VRAM·디스크 절약")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--keep-videos", action="store_true",
                    help="추출 후 임시 mp4를 지우지 않음")
    args = ap.parse_args()

    index = build_index()
    by_clip = load_windows(args.splits)

    clips = list(by_clip.keys())
    if args.clips:
        clips = [c for c in clips if c in args.clips]
    if args.limit_clips:
        clips = clips[: args.limit_clips]

    os.makedirs(VIDEO_TMP, exist_ok=True)
    print(f"\n처리할 clip: {len(clips)}개, 프레임 긴변={args.max_side or '원본'}\n")

    total_saved = 0
    for ci, clip in enumerate(clips, 1):
        windows = by_clip[clip]
        if all(already_done(w) for w in windows.values()):
            print(f"[{ci}/{len(clips)}] {clip}: 이미 완료, 건너뜀")
            continue
        if clip not in index:
            print(f"[{ci}/{len(clips)}] {clip}: 색인에 없음(영상 없음?), 건너뜀")
            continue

        entry = index[clip]
        member = entry["member"]
        local_mp4 = os.path.join(VIDEO_TMP, os.path.basename(member))
        if not os.path.exists(local_mp4):
            print(f"[{ci}/{len(clips)}] {clip}: 영상 range 다운로드 중...")
            with RemoteZip(entry["batch_url"], session=_auth_session()) as z:
                z.extract(member, VIDEO_TMP)
                # zip 내부 경로가 하위폴더면 평탄화
                extracted = os.path.join(VIDEO_TMP, member)
                if extracted != local_mp4 and os.path.exists(extracted):
                    os.replace(extracted, local_mp4)

        n = extract_from_video(local_mp4, windows, args.max_side, args.jpeg_quality)
        total_saved += n
        print(f"[{ci}/{len(clips)}] {clip}: 프레임 {n}장 저장 (창 {len(windows)}개)")

        if not args.keep_videos and os.path.exists(local_mp4):
            os.remove(local_mp4)

    print(f"\n완료. 총 {total_saved:,}장 저장 → data/frames/")


if __name__ == "__main__":
    main()
