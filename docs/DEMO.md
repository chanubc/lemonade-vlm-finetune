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
