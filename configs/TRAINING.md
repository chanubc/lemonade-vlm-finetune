# 학습(파인튜닝) 가이드 — Qwen2.5-VL-3B + QLoRA on LEMONADE

로컬 RTX 5070(12GB) 기준. Perception/Reasoning만 학습.

## 0. 사전 준비 (한 번)
학습에 쓸 프레임을 먼저 뽑아야 한다(train/val 참가자).
```bash
uv run python scripts/extract_frames.py --splits train val
```
> 12GB에서 빡빡하면 프레임 4장으로 줄여서 재생성하는 편이 안전:
> ```bash
> uv run python scripts/convert_to_vqa.py --frames 4
> uv run python scripts/extract_frames.py --splits train val --max-side 448
> ```

## 1. LLaMA-Factory 설치
```bash
uv add "llamafactory[metrics]"
```
⚠️ **버전 충돌 주의:** 이 프로젝트는 지금 transformers 5.x / torch 2.11(cu128)을 쓴다.
LLaMA-Factory가 더 낮은 transformers를 요구하면 uv가 충돌을 알린다. 그럴 때 선택지:
- (a) LLaMA-Factory를 **별도 uv 환경**에 설치해 학습만 거기서 돌린다(평가 env는 그대로 둠).
- (b) 충돌 시 아래 "대안"의 순수 transformers 학습 스크립트를 쓴다.

## 2. 학습 실행
```bash
uv run llamafactory-cli train configs/qwen2_5vl_lemonade_qlora.yaml
```
- 결과(LoRA 어댑터): `out/qwen2p5vl-3b-lemonade-qlora/`
- 손실 곡선: `out/.../training_loss.png`
- OOM이 나면 config 하단 "VRAM 줄이는 순서"를 따른다.

## 3. before/after 평가
```bash
# before (학습 전 기본 모델) — 이미 out/eval_before.json 로 뽑아둠
# after (학습한 어댑터를 붙여서)
uv run python scripts/evaluate.py --split test --max-pixels 200704 \
    --adapter out/qwen2p5vl-3b-lemonade-qlora --out out/eval_after.json
```
`eval_before.json` vs `eval_after.json`의 전체/카테고리/난이도별 정확도를 비교한다.

## 4. (선택) 서빙용으로 합치기 + open-webui 데모
```bash
uv run llamafactory-cli export configs/merge_lora.yaml       # 어댑터 병합
# 합친 모델(out/qwen2p5vl-3b-lemonade-merged)을 vLLM로 띄우고 open-webui 연결
```

## 대안: LLaMA-Factory 없이 순수 transformers로
버전 충돌이 성가시면, 이미 설치된 transformers+peft+trl 스택으로
`scripts/train_sft.py`(직접 작성) 방식도 가능하다. 필요하면 요청할 것.
