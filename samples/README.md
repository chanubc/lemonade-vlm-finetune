# 데모 샘플 (git 포함)

clone 후 바로 open-webui에서 테스트할 수 있는 소량 예제.
각 예제는 `samples/frames/<폴더>/` 의 **8장을 모두 첨부** + 아래 질문을 넣고,
모델이 **정답 글자**를 맞히는지 확인한다. (데모 실행법: `docs/DEMO.md`)

> 어댑터는 git에 없다 → 서버 실행 시 HF에서 받는다:
> `uv run python scripts/serve_openai.py --adapter chanubc/Qwen2.5-VL-3B-LEMONADE-LoRA`

## 예제 1 — Perception (easy) · 정답: **D**
이미지: `samples/frames/YH2003_2023_05_17_09_08_58_51162_51186/` (8장)

```
Answer the following multiple-choice question using the given images.
Question: What action am I doing ?
Choices:
A. grabbing the green salad
B. shaking the carrots
C. holding the radish
D. closing the bottle
Respond only with the letter of the correct answer.
```

## 예제 2 — Perception (easy) · 정답: **D**
이미지: `samples/frames/YH2003_2023_06_02_09_20_42_24473_24526/` (8장)

```
Answer the following multiple-choice question using the given images.
Question: What action am I doing ?
Choices:
A. holding the bowl
B. peeling the zucchini
C. peeling the bell pepper
D. carrying the tissue
Respond only with the letter of the correct answer.
```

## 예제 3 — Perception (easy) · 정답: **A**
이미지: `samples/frames/YH2003_2023_05_17_09_08_58_44015_44819/` (8장)

```
Answer the following multiple-choice question using the given images.
Question: What action am I doing ?
Choices:
A. Carry
B. Hold
C. Touch
D. Press
Respond only with the letter of the correct answer.
```

## 예제 4 — Reasoning (medium) · 정답: **C**
이미지: `samples/frames/YH2003_2023_05_17_09_08_58_27866_27916/` (8장)

```
Answer the following multiple-choice question using the given images.
Question: What were my previous 3 actions ?
Choices:
A. grabbing the shallots + tasting the risotto + shaking the carrots
B. sliding the spatula + cleaning the hand + moving the knife
C. grabbing the tissue + carrying the tissue + drying the hand
D. carrying the cucumber   + grabbing the shallots + carrying the tissue
Respond only with the letter of the correct answer.
```

## 예제 5 — Reasoning (medium) · 정답: **D**
이미지: `samples/frames/YH2003_2023_05_17_09_08_58_26819_26869/` (8장)

```
Answer the following multiple-choice question using the given images.
Question: What were my previous 3 actions ?
Choices:
A. washing the spoon + holding the radish + moving the package
B. moving the tomatoes + holding the zucchini + carrying the radish
C. stirring the spatula + touching the package + carrying the trivet
D. touching the recipe + reading the recipe + grabbing the radish
Respond only with the letter of the correct answer.
```

## 예제 6 — Reasoning (hard) · 정답: **A**
이미지: `samples/frames/YH2003_2023_05_17_09_08_58_47023_47073/` (8장)

```
Answer the following multiple-choice question using the given images.
Question: What were my previous 4 actions ?
Choices:
A. carrying the avocado + carrying the avocado + grabbing the knife + carrying the avocado
B. cleaning the avocado + carrying the knife + cutting the avocado + putting the cucumber  
C. carrying the avocado + carrying the glove + cutting the avocado + carrying the knife
D. cutting the avocado + carrying the avocado + carrying the knife + carrying the knife
Respond only with the letter of the correct answer.
```
