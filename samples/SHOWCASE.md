# ⭐ Hard 쇼케이스 — 기본 모델은 틀리고, 파인튜닝 모델은 맞히는 문제

`samples/README.md`의 일반 예제보다 어려운, **Reasoning · hard** 문항.
둘 다 **기본 Qwen2.5-VL은 오답**, **파인튜닝 모델(우리)은 정답**이다.
(테스트법: 해당 폴더의 8장을 모두 첨부 + 아래 프롬프트. 실행은 `docs/DEMO.md`)

---

## 1. 직전 4개 행동 맞히기 (정답: **D**)
- 폴더: `samples/frames/YH2003_2023_05_17_09_08_58_47023_47073/` (8장)
- 기본 모델: **B (오답)** → 파인튜닝: **D (정답)**
- 보기 A/B/C가 거의 같은 "칼+아보카도" 순열이라 사람도 헷갈리는 난도.

**English (학습에 쓰인 형식)**
```text
Answer the following multiple-choice question using the given images.
Question: What were my previous 4 actions ?
Choices:
A. carrying the knife + grabbing the knife + cutting the avocado + carrying the avocado
B. grabbing the avocado + carrying the avocado + carrying the avocado + grabbing the knife
C. carrying the knife + grabbing the knife + carrying the avocado + carrying the avocado
D. carrying the avocado + carrying the spoon + adding the avocado + putting the spoon
Respond only with the letter of the correct answer.
```
**한국어 (번역, OOD 테스트용)**
```text
주어진 이미지들을 사용하여 다음 객관식 질문에 답하세요.
질문: 내가 바로 직전에 한 4개의 행동은 무엇인가요?
보기:
A. 칼 나르기 + 칼 집기 + 아보카도 자르기 + 아보카도 나르기
B. 아보카도 집기 + 아보카도 나르기 + 아보카도 나르기 + 칼 집기
C. 칼 나르기 + 칼 집기 + 아보카도 나르기 + 아보카도 나르기
D. 아보카도 나르기 + 숟가락 나르기 + 아보카도 넣기 + 숟가락 내려놓기
정답에 해당하는 글자 하나만 답하세요.
```

---

## 2. 현재 활동 단계 추론 (정답: **C**)
- 폴더: `samples/frames/YH2003_2023_05_17_09_08_58_6727_6739/` (8장)
- 기본 모델: **B (오답)** → 파인튜닝: **C (정답)**

**English**
```text
Answer the following multiple-choice question using the given images.
Question: I am currently putting the package, what is my current activity ?
Choices:
A. Preparing ingredients
B. Setting up and not cooking
C. Gathering supplies
D. Cooking at the stoves
Respond only with the letter of the correct answer.
```
**한국어**
```text
주어진 이미지들을 사용하여 다음 객관식 질문에 답하세요.
질문: 나는 지금 포장(패키지)을 내려놓고 있는데, 현재 나의 활동 단계는 무엇인가요?
보기:
A. 재료 준비하기
B. 세팅 중이고 요리는 안 함
C. 물품 가져오기(모으기)
D. 스토브에서 요리하기
정답에 해당하는 글자 하나만 답하세요.
```

---

> 이 두 문항은 실제 test 평가에서 뽑은 것으로, 전체 결과(41.0% → 71.5%) 중
> "기본은 틀리고 파인튜닝은 맞힌" 대표 사례다. 상세: `results/after_comparison.md`.
