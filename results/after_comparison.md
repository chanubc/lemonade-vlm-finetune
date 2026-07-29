# Before / After 비교 — QLoRA 파인튜닝 결과

- 모델: Qwen2.5-VL-3B-Instruct
- 학습: QLoRA(4비트), 1 epoch, train 12,858개(Perception/Reasoning), 9시간(RTX 5070)
- 평가: LEMONADE test 2,852문제, **before/after 동일 조건**(8프레임, 200704픽셀, greedy)
- test 참가자(YH2003·YH2029·YH2030)는 학습에서 한 번도 안 본 사람 → 진짜 일반화 측정

## 결과
| 구분 | before | after | 변화 |
|---|---|---|---|
| **전체** | 41.0% | **71.5%** | **+30.4** |
| Perception | 40.7% | 67.2% | +26.5 |
| Reasoning | 41.4% | 76.0% | +34.6 |
| easy | 46.0% | 76.1% | +30.1 |
| medium | 41.3% | 72.9% | +31.6 |
| hard | 34.0% | 63.5% | +29.5 |

(참고: 텍스트만 28.2%, 찍기 25%)

## 해석
- **전체 정확도가 41% → 71.5%로 +30%p, 거의 2배** 향상. 파인튜닝이 크게 효과적.
- **모든 유형·난이도에서 고르게 상승.** 특히 Reasoning +34.6, 가장 어려운 hard도 +29.5로 대폭 개선.
- 상승이 "외우기"가 아닌 이유: test는 **학습에 없던 참가자**들이다. 모델이 이 도메인의 행동 어휘(동사 33·명사 79)와 질문 형식, 1인칭 시점 판단을 학습한 결과.
- loss도 0.53 → 0.19로 수렴, 과적합 징후 없이 1 epoch만에 큰 이득.

## 재현
- before: `out/eval_before.json`, 텍스트대조군: `out/eval_textonly.json`
- after: `out/eval_after.json`, 어댑터: `out/qwen2p5vl-3b-lemonade-qlora/`
- 비교: `uv run python scripts/compare_results.py`

## 다음에 해볼 만한 것
- 2 epoch 또는 더 큰 데이터/해상도(H200에서) → 추가 향상 여지
- Kinematics/Physical 등 나머지 유형 포함 실험
- 어댑터를 HuggingFace(private)에 백업, vLLM+open-webui로 데모
