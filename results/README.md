# 조리 위험 인식 VLM — PoC 결과 정리

방향 전환: LEMONADE(조리 행동 VQA) → **"정상 조리 vs 위험(발연/화재) 인식"**.
전체 계획·근거는 [../docs/HANDOFF.md](../docs/HANDOFF.md). 아래는 실행한 실험과 수치.

## 실험 요약

| STEP | 내용 | 핵심 결과 | 파일 |
|---|---|---|---|
| 0 | base VLM 화재 F1 (야외 D-Fire) | F1 **0.819**, 재현율 0.97, 오탐 40% → 단순 화재감지 약함 | [step0_baseline.md](step0_baseline.md) |
| 0-C | 조리 도메인 재측정 | 재현율 100%, **정상 조리 오탐 15%(랜덤)/44%(hard)**, 발연 100% fire 오답 | [step0c_cooking.md](step0c_cooking.md) |
| 1 | 위험 홀드아웃 실물화 | D-Fire는 야외라 부적합 확인 → **NIST FCD 조리유 팬 화재**로 v0 구축(크롭·라벨) | (holdout_v0) |
| 2 | 합성 파이프라인 | 실 조리 프레임 위 실불꽃 컷아웃 합성, normal/smoke/fire | [../scripts/step2_composite.py](../scripts/step2_composite.py) |
| 4 | QLoRA 파인튜닝 3판 | 아래 표 | [step4_comparison.md](step4_comparison.md) |

## STEP 4 — base vs 파인튜닝

| 지표 | base(프롬프트만) | v1 합성360 | v2 개선+하이브리드460 |
|---|---|---|---|
| fire 재현율 | 10/10 | 10/10 | 10/10 |
| 발연 3분류 | 50% | 50% | 50% |
| 오탐 랜덤 | 2.5% | 7.5% | **0%** |
| 오탐 hard | 0% | 8% | **0%** |

## 지금까지의 결론
- **오탐은 프롬프트 엔지니어링으로 대부분 해결됨**(15/44% → 2.5/0%). 파인튜닝은 아직 잘 만든 프롬프트를 **뚜렷이 못 이김**.
- v2 개선안이 v1의 오탐 악화는 되돌렸으나(동급), 진짜 어려운 **발연↔화재 단계(50%)는 미개선**.
- **홀드아웃이 작다(fire 10·smoke 4)** → 결론 신뢰구간이 넓음. **평가 데이터 확보가 1순위.**

## 재현 방법
```bash
uv run --group train python scripts/step0_baseline.py         # STEP0 (D-Fire 준비 후)
uv run --group train python scripts/step0c_cooking_baseline.py # STEP0-C
uv run --group train python scripts/collect_incident_frames.py # NIST 영상→프레임
uv run python scripts/step1_crop_label_holdout.py             # 크롭+라벨
uv run python scripts/step2_composite.py                      # 합성
uv run python scripts/step4_make_vqa.py                       # VQA 변환
uv run llamafactory-cli train configs/qwen2_5vl_cooking_qlora.yaml  # 학습
uv run --group train python scripts/step4_eval.py [--adapter ...]   # 평가
```
대용량 데이터(영상·프레임·합성·어댑터)는 git 제외, 위 스크립트로 재생성.
