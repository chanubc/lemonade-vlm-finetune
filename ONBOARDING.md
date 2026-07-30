# 온보딩 — LEMONADE VLM 파인튜닝 프로젝트

처음 오신 팀원이 **처음부터 끝까지 따라 하며** 프로젝트를 재현할 수 있도록 정리한 문서.

## 이 프로젝트가 뭐예요?
요리하는 사람을 1인칭으로 찍은 데이터(**EPFL-Smart-Kitchen-30**, arXiv 2506.01608)의
객관식 영상질의응답 벤치마크 **LEMONADE** 로, 영상+언어 모델 **Qwen2.5-VL-3B** 를
파인튜닝(추가 학습)하는 실험입니다.

**핵심 결과:** 학습 전 41.0% → 학습 후 **71.5%** (test 정확도, +30%p). 자세한 건
[results/after_comparison.md](results/after_comparison.md).

## 용어 30초 정리
- **VLM(영상·이미지+언어 모델)**: 이미지를 보고 글로 답하는 AI. 여기선 Qwen2.5-VL.
- **VQA(영상질의응답)**: 이미지+질문 → 정답. LEMONADE는 4지선다(A/B/C/D).
- **LoRA / QLoRA**: 모델 전체가 아니라 작은 추가 층만 학습하는 기법(QLoRA는 4비트로 압축해 12GB GPU에서도 가능).
- **어댑터**: 학습으로 나온 그 작은 층. 원본 모델에 얹어서 쓴다(119MB).

---

## 0. 준비물
- NVIDIA GPU. RTX 5070(12GB)에서 검증됨. **RTX 50xx(Blackwell)은 CUDA 12.8+/torch 2.7+ 필수.**
- [uv](https://docs.astral.sh/uv/) (파이썬 패키지 관리 — 이 프로젝트는 pip 대신 uv로 전부 관리)
- (데모용) Docker
- HuggingFace 계정 + **read 토큰** (데이터 다운로드 속도 때문에 필요)

## 1. 환경 세팅
```bash
git clone <이 저장소>
cd <프로젝트 폴더>
uv sync                     # 데이터 준비용 패키지
uv sync --group train       # 학습/추론용(torch cu128, transformers, llamafactory 등)
uv run hf auth login        # HuggingFace read 토큰 붙여넣기 (다운로드 속도 위해)
```
GPU 확인:
```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
# True (12, 0) 이 나와야 함 (5070 기준)
```

## 2. 데이터 준비
```bash
uv run python scripts/download_qa.py          # ① QA 표만 다운로드(비디오 61GB 제외, 1.4MB)
uv run python scripts/make_splits.py          # ② 참가자 단위 train/val/test 분할
uv run python scripts/convert_to_vqa.py       # ③ Perception/Reasoning를 학습 포맷(JSON)으로
uv run python scripts/extract_frames.py --splits train val test
                                              # ④ 필요한 영상만 스트리밍해 프레임 추출
uv run python scripts/convert_to_vqa.py --require-frames   # ⑤ 프레임 없는 항목 제외해 재생성
```
**왜 참가자 단위로 나누나?** 같은 사람 영상이 학습·평가에 겹치면 "그 사람 외우기"로
점수가 부풀려진다. 참가자를 통째로 한쪽에만 넣어야 "처음 보는 사람"에 대한 진짜 실력이 측정된다.

## 3. 학습 전 기준선(before) 측정
```bash
uv run python scripts/evaluate.py --split test --max-pixels 200704 --out out/eval_before.json
# (선택) 비전이 실제로 쓰이는지 대조군:
uv run python scripts/evaluate.py --split test --text-only --out out/eval_textonly.json
```

## 4. 학습 (QLoRA)
```bash
uv run llamafactory-cli train configs/qwen2_5vl_lemonade_qlora.yaml
# 결과 어댑터: out/qwen2p5vl-3b-lemonade-qlora/  (RTX 5070에서 1 epoch ~9시간)
```
OOM(메모리 부족)이 나면 config 하단 "VRAM 줄이는 순서" 참고. 상세는
[configs/TRAINING.md](configs/TRAINING.md).

## 5. 학습 후(after) 평가 & 비교
```bash
uv run python scripts/evaluate.py --split test --max-pixels 200704 \
    --adapter out/qwen2p5vl-3b-lemonade-qlora --out out/eval_after.json
uv run python scripts/compare_results.py      # before vs after 표
```

## 6. 데모 (open-webui에서 직접 써보기)
[docs/DEMO.md](docs/DEMO.md) 참고. 요약:
```bash
uv run python scripts/serve_openai.py         # 모델 서버(:8000)
docker run -d -p 3000:8080 -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e WEBUI_AUTH=False --name open-webui ghcr.io/open-webui/open-webui:main
# http://localhost:3000 → 모델 lemonade-qwen2.5-vl-3b 선택 → 이미지+질문
```

---

## 폴더 지도
```
scripts/   재현 스크립트 (download_qa / make_splits / convert_to_vqa / extract_frames
           / evaluate / compare_results / serve_openai)
configs/   학습 설정(qwen2_5vl_lemonade_qlora.yaml), 병합, 스모크, TRAINING.md
data/      raw(QA표) / splits(분할) / converted(학습JSON) / frames(추출 이미지) — 대용량은 .gitignore
results/   before_baseline.md, after_comparison.md
docs/      DEMO.md
papers/    논문 PDF (.gitignore)
out/       학습 산출물·평가 결과 (.gitignore)
```

## 알아두면 좋은 점 / 함정
- **패키지는 무조건 uv로** (`uv add`, `uv run`, `uv sync`). pip 직접 사용 금지.
- 데이터 `Answers`는 리스트처럼 생긴 **문자열** → `ast.literal_eval`로 파싱. `Start/End`는 **프레임 번호**(초 아님).
- 영상은 61GB zip 5개 → 스크립트가 **필요한 것만 스트리밍**해 프레임만 남기고 삭제(디스크 절약).
- 손상된 영상 클립이 1개 있어 train의 ~372개는 프레임이 없다(`--require-frames`로 자동 제외됨). 학습엔 영향 미미.
- 공개 어댑터: https://huggingface.co/chanubc/Qwen2.5-VL-3B-LEMONADE-LoRA

## 더 해볼 것
- 2 epoch / 더 큰 데이터·해상도(H200) / 나머지 유형(Kinematics 등) 포함
- vLLM 서빙(docs/DEMO.md B절)
