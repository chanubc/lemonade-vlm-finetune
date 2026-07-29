# LEMONADE VLM Finetune

EPFL-Smart-Kitchen-30 논문(arXiv 2506.01608)의 **LEMONADE** VQA 벤치마크로
**Qwen2.5-VL-3B**를 파인튜닝하기 위한 작업 저장소.

> 폴더명은 현재 `kaist`이지만 프로젝트 성격상 `lemonade-vlm-finetune`이 적합.
> (세션 종료 후 이름 변경 방법은 아래 참고)

## 진행 상황
- [x] 논문 저장 (`papers/`)
- [x] QA 표 다운로드 (비디오 제외, `scripts/download_qa.py`)
- [x] 참가자 단위 train/val/test 분할 (`scripts/make_splits.py`)
- [x] Perception/Reasoning 학습 포맷 변환 (`scripts/convert_to_vqa.py`)
- [ ] 비디오 다운로드 + 프레임 추출
- [ ] Qwen2.5-VL-3B 파인튜닝 (LLaMA-Factory)
- [ ] lmms-eval 평가

## 폴더 구조
```
scripts/          재현용 스크립트 (git에 포함)
  download_qa.py     QA 표(parquet)만 다운로드
  make_splits.py     참가자 단위 분할
  convert_to_vqa.py  Qwen2.5-VL 학습 JSON 변환
data/             데이터 (대용량은 .gitignore)
  raw/               lemonade_qa.parquet (36,521 QA)
  splits/            train/val/test.parquet + split_manifest.json
  converted/         train/val/test.json (학습 포맷) + dataset_info.json
  videos/  frames/   (아직 없음) 비디오·추출 프레임
papers/           논문 PDF (.gitignore)
```

## 재현 방법
```bash
python scripts/download_qa.py                 # 1. QA 표 다운로드
python scripts/make_splits.py                 # 2. 참가자 단위 분할
python scripts/convert_to_vqa.py --frames 8   # 3. Perception/Reasoning 변환
# (12GB VRAM이면 --frames 4 권장)
```

## 데이터 규모 (Perception + Reasoning, 변환 결과)
| split | QA 수 |
|---|---|
| train | 13,230 |
| val | 2,775 |
| test | 2,852 |
| 합계 | 18,857 |

## 폴더명 변경 (세션 종료 후)
현재 세션이 이 폴더를 사용 중이라 실행 중에는 변경 불가.
Claude Code를 종료한 뒤 PowerShell에서:
```powershell
Rename-Item C:\Users\chanwoo\workspace\kaist C:\Users\chanwoo\workspace\lemonade-vlm-finetune
```
