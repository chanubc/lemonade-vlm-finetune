# 데모: 파인튜닝 모델을 open-webui에서 직접 써보기

파인튜닝한 Qwen2.5-VL-3B(+LoRA)를 **채팅 UI(open-webui)** 에 붙여, 부엌 이미지를
올리고 질문해보는 데모. 두 갈래로 설명한다.

- **A. 지금 Windows(RTX 5070)에서 바로 되는 길** — transformers 기반 OpenAI 호환 서버
- **B. 나중에 H200/WSL에서 빠르게 돌리는 길** — vLLM

---

## A. Windows에서 바로 (검증됨)

구조: `open-webui(Docker) → http://host.docker.internal:8000/v1 → 모델 서버(로컬)`

### 1) 모델 서버 실행
```bash
uv run python scripts/serve_openai.py
# 로컬 어댑터(out/qwen2p5vl-3b-lemonade-qlora) 사용, bf16, 포트 8000
# HF의 공개 어댑터를 쓰려면:
#   uv run python scripts/serve_openai.py --adapter chanubc/Qwen2.5-VL-3B-LEMONADE-LoRA
```
모델 로드에 ~1분(VRAM ~7-9GB 점유). `http://127.0.0.1:8000/v1` 에 OpenAI 호환 API가 뜬다.

### 2) open-webui 실행 (Docker)
```bash
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=sk-local \
  -e WEBUI_AUTH=False \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

### 3) 사용
1. 브라우저에서 **http://localhost:3000**
2. 모델 목록에서 **`lemonade-qwen2.5-vl-3b`** 선택
3. 부엌 이미지(예: `data/frames/.../frame_00.jpg`)를 첨부하고 질문
   - 예: "What action is the person doing? A. Hold B. Read C. Tap D. Move"

### 3-1) 채팅창에서 어떻게 테스트하나

이 모델은 **이미지를 봐야** 답할 수 있다(비전+언어). 그래서 순서가 중요하다:

1. 상단에서 모델 **`lemonade-qwen2.5-vl-3b`** 선택
2. 채팅 입력창 왼쪽 **`+`(또는 클립) 아이콘**으로 **이미지를 첨부**
   - 한 문항은 원래 프레임 8장을 쓴다. 정확히 보려면 폴더의 8장을 모두 드래그해 첨부.
   - 빠르게 감만 볼 거면 1~2장만 첨부해도 응답은 나온다(정확도는 8장이 최선).
3. **질문 텍스트**를 입력하고 전송

이미지는 로컬 파일에서 첨부한다. 예시 프레임 위치:
`C:\Users\chanwoo\workspace\kaist\data\frames\<폴더>\frame_00.jpg ... frame_07.jpg`

#### 정답을 아는 테스트 예제 (모델이 맞히는지 직접 확인)

**예제 1 — Perception** (정답: **D**)
프레임 폴더: `data/frames/YH2003_2023_05_17_09_08_58_51162_51186/`
채팅창에 붙여넣기:
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

**예제 2 — Reasoning** (정답: **C**)
프레임 폴더: `data/frames/YH2003_2023_05_17_09_08_58_27866_27916/`
채팅창에 붙여넣기:
```
Answer the following multiple-choice question using the given images.
Question: What were my previous 3 actions ?
Choices:
A. grabbing the shallots + tasting the risotto + shaking the carrots
B. sliding the spatula + cleaning the hand + moving the knife
C. grabbing the tissue + carrying the tissue + drying the hand
D. carrying the cucumber + grabbing the shallots + carrying the tissue
Respond only with the letter of the correct answer.
```

**자유 질문도 가능** (형식 없이):
```
이 이미지에서 사람이 뭘 하고 있어? 한 문장으로 설명해줘.
```

> 팁: 각 프레임 폴더 이름의 숫자(`_51162_51186`)가 그 문항의 영상 구간이다.
> 다른 문항을 시험하려면 `data/frames/`의 다른 폴더를 골라 8장을 첨부하면 된다.

### 4) 종료 (GPU 자원 회수)
```bash
docker stop open-webui        # UI 정지 (GPU 안 씀)
# 모델 서버는 실행한 터미널에서 Ctrl+C, 또는 python 프로세스 종료 → VRAM 반환
```

> 참고: `WEBUI_AUTH=False`라 로그인 없이 바로 쓴다. 서버는 bf16이라 vLLM보다 느리지만
> 데모엔 충분하다.

---

## B. H200 / WSL2에서 vLLM (더 빠름)

vLLM은 **Linux 전용**(네이티브 Windows 미지원). H200 서버나 WSL2에서 사용.

```bash
uv pip install vllm
# base 모델 + LoRA 어댑터를 함께 서빙 (OpenAI 호환, 포트 8000)
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
  --enable-lora \
  --lora-modules lemonade=chanubc/Qwen2.5-VL-3B-LEMONADE-LoRA \
  --max-model-len 8192 --port 8000
```
그 다음 open-webui는 위 A-2와 동일하게 띄우고, 모델 이름만 `lemonade`로 선택.

장점: 훨씬 빠른 추론·동시 요청 처리. H200이면 여유 VRAM으로 배치도 크게.

---

## 문제 해결
- open-webui에 모델이 안 보이면: 모델 서버가 떠 있는지(`curl http://127.0.0.1:8000/v1/models`),
  컨테이너에서 호스트가 보이는지(`docker exec open-webui curl http://host.docker.internal:8000/v1/models`) 확인.
- VRAM 부족: `serve_openai.py --max-pixels 100352` 로 이미지 토큰을 줄인다.
