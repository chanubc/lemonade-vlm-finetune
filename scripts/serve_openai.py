"""
파인튜닝한 Qwen2.5-VL-3B(+LoRA)를 OpenAI 호환 API로 띄우는 가벼운 서버.
open-webui 등 OpenAI 호환 클라이언트가 여기에 붙어 이미지+질문을 주고받는다.

Windows에서 그대로 동작(transformers 기반, vLLM 불필요).
실행:
  uv run python scripts/serve_openai.py            # 로컬 어댑터 사용
  uv run python scripts/serve_openai.py --adapter chanubc/Qwen2.5-VL-3B-LEMONADE-LoRA
엔드포인트:
  GET  /v1/models
  POST /v1/chat/completions   (OpenAI 형식, 이미지 image_url 지원: data URI/http)
"""
import argparse
import base64
import io
import time
import urllib.request

import torch
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_NAME = "lemonade-qwen2.5-vl-3b"  # 클라이언트에 보일 모델 이름


def load_image(url: str) -> Image.Image:
    if url.startswith("data:"):                     # data:image/...;base64,XXXX
        b64 = url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if url.startswith("http"):
        with urllib.request.urlopen(url, timeout=30) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    return Image.open(url).convert("RGB")           # 로컬 경로


def to_qwen(messages):
    """OpenAI 메시지 → Qwen 메시지(이미지 자리표시자) + PIL 이미지 목록."""
    qmsgs, images = [], []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            qmsgs.append({"role": m["role"], "content": [{"type": "text", "text": content}]})
            continue
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append({"type": "text", "text": item["text"]})
            elif item.get("type") == "image_url":
                images.append(load_image(item["image_url"]["url"]))
                parts.append({"type": "image"})
        qmsgs.append({"role": m["role"], "content": parts})
    return qmsgs, images


class ChatReq(BaseModel):
    messages: list
    max_tokens: int = 128
    temperature: float = 0.0
    model: str | None = None


def build_app(model, processor):
    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatReq):
        qmsgs, images = to_qwen(req.messages)
        text = processor.apply_chat_template(qmsgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images or None, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=req.max_tokens,
                do_sample=req.temperature > 0, temperature=max(req.temperature, 1e-5),
            )
        out = processor.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or MODEL_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": out},
                         "finish_reason": "stop"}],
        }

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default="out/qwen2p5vl-3b-lemonade-qlora",
                    help="LoRA 어댑터 경로(로컬) 또는 HF repo id. 빈 값이면 기본 모델만.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-pixels", type=int, default=200704)
    args = ap.parse_args()

    print(f"모델 로드: {args.base}" + (f" + LoRA({args.adapter})" if args.adapter else ""))
    processor = AutoProcessor.from_pretrained(args.base, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    import uvicorn
    print(f"서버 시작: http://{args.host}:{args.port}/v1  (모델명: {MODEL_NAME})")
    uvicorn.run(build_app(model, processor), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
