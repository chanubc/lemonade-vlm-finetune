# LEMONADE VQA 데이터 (표만, 비디오 제외)

출처: HuggingFace `amathislab/LEMONADE` (논문: EPFL-Smart-Kitchen-30, arXiv 2506.01608)

## 폴더 구조
- `raw/lemonade_qa.parquet` — 전체 36,521개 QA (원본 표, 1.4MB). `scripts/download_qa.py`로 생성.
- `splits/{train,val,test}.parquet` — 참가자 단위로 나눈 split. `scripts/make_splits.py`로 생성.
- `splits/split_manifest.json` — 어느 참가자가 어느 split인지 기록.

## 분할 결과 (참가자 단위, 누수 없음)
| split | QA 개수 | 비율 | 참가자 수 |
|---|---|---|---|
| train | 25,763 | 70.5% | 10명 |
| val | 5,400 | 14.8% | 3명 |
| test | 5,358 | 14.7% | 3명 |

같은 참가자는 한 split에만 존재 → "처음 보는 사람"에 대한 일반화 성능을 측정.

## 컬럼 설명
| 컬럼 | 의미 |
|---|---|
| `Question` | 질문 텍스트 |
| `Answers` | 보기 4개. **리스트처럼 생긴 문자열** → `ast.literal_eval()`로 파싱 필요 |
| `Correct Answer` | 정답을 글자 A/B/C/D로. `Answers` 리스트 순서에 대응 |
| `Clip` | `참가자ID_날짜_시각`. 대응 비디오: `videos/{Clip}_hololens.mp4` |
| `Start`, `End` | 문맥 구간. **단위는 프레임 번호** (초 아님 → `초 = 프레임 / fps`) |
| `Category` / `Subcategory` / `Difficulty` | 분류 정보 |
| `QID` | 질문 템플릿 번호 (0~30) |
| `participant` | `Clip`에서 추출한 참가자 ID (스크립트가 추가한 컬럼) |

## 아직 안 받은 것
- 비디오(약 61GB): `videos/*.mp4`. 프레임 추출(학습 입력 이미지) 단계에서 필요.
