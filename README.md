# LEMONADE VLM Finetune

EPFL-Smart-Kitchen-30 논문([arXiv 2506.01608](https://arxiv.org/abs/2506.01608))의
**LEMONADE** VQA 벤치마크로 **Qwen2.5-VL-3B** 를 파인튜닝한 프로젝트.
1인칭 요리 영상에 대한 4지선다 질의응답(Perception/Reasoning)을 학습한다.

> 처음 오셨다면 → **[ONBOARDING.md](ONBOARDING.md)** (처음부터 끝까지 따라 하기)

## 결과 (한눈에)
test 2,852문제, 학습에 없던 참가자 기준(누수 없음). 찍기 = 25%.

| | before(기본) | **after(파인튜닝)** | Δ |
|---|---|---|---|
| **전체** | 41.0% | **71.5%** | **+30.4** |
| Perception | 40.7% | 67.2% | +26.5 |
| Reasoning | 41.4% | 76.0% | +34.6 |
| easy/medium/hard | 46/41/34% | 76/73/64% | +30/+32/+30 |

QLoRA(4비트), 1 epoch, RTX 5070(12GB)에서 ~9시간. 상세: [results/after_comparison.md](results/after_comparison.md)
공개 어댑터: https://huggingface.co/chanubc/Qwen2.5-VL-3B-LEMONADE-LoRA

## 진행 상황
- [x] 논문 저장 / QA 표 다운로드 / 참가자 단위 분할 / 학습 포맷 변환
- [x] 영상 프레임 추출 (필요한 것만 스트리밍)
- [x] before 기준선 측정 (41.0%, 비전 기여 +12.9%p)
- [x] QLoRA 파인튜닝 (LLaMA-Factory)
- [x] after 평가 및 비교 (71.5%)
- [x] 어댑터 HuggingFace 공개 + open-webui 데모

## 빠른 재현
```bash
uv sync && uv sync --group train
uv run hf auth login
uv run python scripts/download_qa.py
uv run python scripts/make_splits.py
uv run python scripts/convert_to_vqa.py
uv run python scripts/extract_frames.py --splits train val test
uv run python scripts/convert_to_vqa.py --require-frames
uv run llamafactory-cli train configs/qwen2_5vl_lemonade_qlora.yaml
uv run python scripts/evaluate.py --split test --adapter out/qwen2p5vl-3b-lemonade-qlora --out out/eval_after.json
uv run python scripts/compare_results.py
```
전체 설명은 [ONBOARDING.md](ONBOARDING.md), 학습 상세는 [configs/TRAINING.md](configs/TRAINING.md),
데모는 [docs/DEMO.md](docs/DEMO.md).

## 바로 테스트 (clone 후)
전체 데이터(수 GB)는 .gitignore이지만, **데모용 샘플은 git에 포함**돼 있다.
clone 직후 `samples/` 의 이미지 8장 + 질문으로 open-webui에서 바로 테스트 가능.
→ [samples/README.md](samples/README.md) (정답 포함 예제 6개), 실행법은 [docs/DEMO.md](docs/DEMO.md)

## 폴더 구조
```
scripts/   재현 스크립트 (다운로드·분할·변환·프레임추출·평가·비교·서빙·샘플생성)
configs/   학습 config(QLoRA) + 병합 + TRAINING.md
samples/   데모용 소량 샘플 (git 포함): frames + 질문·정답
data/      raw / splits / converted / frames  (대용량 .gitignore)
results/   before/after 결과 문서
docs/      DEMO.md (open-webui 데모)
papers/, out/   논문 PDF·학습 산출물 (.gitignore)
```

## 데이터 규모 (Perception + Reasoning)
| split | QA(프레임 확보 후) | 참가자 |
|---|---|---|
| train | 12,858 | 10명 |
| val | 2,775 | 3명 |
| test | 2,852 | 3명 |

## 원칙
- 파이썬 패키지는 **uv로 전부 관리** (`uv add`/`uv run`/`uv sync`).
- 데이터 분할은 **참가자 단위**(데이터 누수 방지).
