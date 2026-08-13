# FDShield ML

FDShield의 사기 거래 XGBoost 학습·추론 서비스입니다. 모델 계산은 ML 담당자가
전달한 `train1.csv`, 80개 모델 Feature, XGBoost 설정을 기준으로 하며 기존의
GCS, Cloud Run Job, MLflow Registry, Backend Callback, 관리자 수동 승인 인프라는
그대로 사용합니다.

## 고정 계약

| 구분 | 계약 |
|---|---|
| 추론 API 입력 | flat raw60: `transaction_id` 1개 + 실제 전처리 입력 59개 |
| 학습 CSV 입력 | raw64: raw60 + 학습 메타데이터·라벨 4개 |
| 전처리 출력 | 이름과 순서가 고정된 숫자 model80 |
| 모델 | XGBoost binary logistic |
| 기본 판정 임계값 | `0.5` |
| Registry 모델명 | `fdshield-fraud-detector-v2` |
| 로컬 모델 번들 | `models/fdshield-fraud-detector-v2` |

학습과 추론은 모두 `fdshield_ml/common/preprocessor.py`의 같은 raw59 전처리를
사용합니다. `transaction_id`는 응답 연결용이며 model80에 들어가지 않습니다.
학습 raw64는 raw60의 flag 컬럼명만 CSV 별칭인
`flag_deposit_more_than_tenmillion`으로 교체하고, `customer_id`,
`customer_identification_number`, `balance_drain_ratio`, `is_fraud` 4개를 더한
구조입니다. 별칭은 추가 열이 아니며 로딩 시 canonical 이름인
`flag_deposit_more_than_ten_million`으로 되돌립니다.

전달본 DTO·CSV의 기존 불일치는 실제 데이터와 model80 계약을 기준으로
보정합니다. `account_remaining_amount_daily_limit_exceeded`는 bool이 아닌
숫자형 금액으로 받고, Channel·Operating System은 공백 제거 후 소문자로
정규화하며, 없을 수 있는 Operating System은 nullable로 유지합니다. 원본
CSV는 수정하지 않습니다.

## 코드 구조

```text
fdshield_ml/
├── common/                         # 계약·전처리·임계값·XGBoost 공용 로직
│   ├── preprocess_config.py          # raw60/raw64/model80 컬럼 계약
│   └── preprocessor.py               # 학습·추론 공용 raw59 -> model80
├── serving/                        # FastAPI 실시간 추론 서버
│   ├── api/ml_input.py               # /ml/predict 라우트
│   ├── dto/                          # PredictInputDTO, PredictResultDTO
│   ├── service/predict/              # 전처리·예측·SHAP 흐름
│   └── integrations/                 # 로컬·MLflow 모델 로딩
└── training/                       # train1 학습 Cloud Run Job
    ├── dataset.py, data_loader.py      # raw64 검증·로컬/GCS 로딩
    ├── service/train/                  # XGBoost 학습·MLflow 등록 흐름
    ├── integrations/                   # MLflow, Backend Callback
    └── job.py                          # 학습 Job 실행 진입점

models/fdshield-fraud-detector-v2/  # 전달 모델 기반 로컬 candidate
data/open/train1.csv                # 로컬 학습 파일, Git 제외
```

### 처음 보는 순서

1. `common/preprocess_config.py`에서 raw60·raw64·model80 컬럼 계약을 확인합니다.
2. `common/preprocessor.py`에서 학습과 추론이 공유하는 변환을 확인합니다.
3. 실시간 추론은 `serving/app.py` → `serving/api/ml_input.py` →
   `serving/service/predict/predict_service.py` 순으로 읽습니다. 모델 로딩은
   `serving/integrations/`에 있습니다.
4. 학습은 `training/dataset.py` → `training/service/train/model_training.py` →
   `training/service/train/train_service.py` → `training/job.py` 순으로 읽습니다.
5. GCS 로딩은 `training/data_loader.py`, MLflow·Backend Callback은
   `training/integrations/`에서 확인합니다.

`common` 모듈은 학습과 추론에서 같은 계약을 사용하도록 유지합니다.
`serving`과 `training`의 외부 연동은 각 패키지의 `integrations`에 분리합니다.

## 개발 환경

Python 3.11 이상 3.14 미만과 [uv](https://docs.astral.sh/uv/)를 사용합니다.

```bash
uv sync --dev
uv run pytest
```

## 학습 데이터 준비

ML 담당자가 전달한 파일을 `data/open/train1.csv`로 복사합니다.

PowerShell:

```powershell
Copy-Item -LiteralPath '<ML 담당자에게 받은 폴더>\app\datas\train1.csv' -Destination '.\data\open\train1.csv'
```

macOS/Linux:

```bash
cp '<전달 폴더>/app/datas/train1.csv' ./data/open/train1.csv
```

기준 데이터:

- 200,000행, 64열
- 정상 197,000건, 사기 3,000건
- SHA-256: `D025873C5E807976657B30721080D00BF6B6544B887FF339E768E8C13F54E446`

CSV는 Git과 Docker 이미지에 포함하지 않습니다. 운영에서는 같은 파일을 비공개
GCS의 버전 고정 경로에 업로드하고 SHA-256을 확인한 뒤 `TRAINING_DATA_URI`에
`gs://` URI를 넣습니다.

## Training Job

`.env.training.example`을 `.env.training`으로 복사하고 MLflow 계정과 Callback
Token을 설정합니다. 학습 입력은 `TRAINING_DATA_URI` 하나뿐입니다.

```env
TRAINING_DATA_URI=data/open/train1.csv
MLFLOW_EXPERIMENT_NAME=fdshield-binary-training
MLFLOW_REGISTERED_MODEL_NAME=fdshield-fraud-detector-v2
MLFLOW_MODEL_ALIAS=champion
```

로컬 Docker 실행:

```powershell
docker build -f Dockerfile.training -t fdshield/ml-training:local .
docker run --rm --env-file .env.training -v "${PWD}/data/open:/app/data/open:ro" fdshield/ml-training:local
```

Training Job은 raw64 스키마, 라벨, 거래 ID와 model80 전처리를 검증한 뒤
stratified 80/20 분할로 후보 모델을 학습해 Registry에 등록합니다. 별도의 Stub이나
검증 전용 실행 모드는 지원하지 않습니다.

후보 모델은 PR-AUC, ROC-AUC, Recall, Precision, F1, FPR과 판정 임계값을
MLflow에 저장합니다. 같은 model80 계약의 현재 champion이 있으면 같은 검증
데이터에서 비교하고 `metadata/model-comparison.json`을 기록합니다. 결과의
`RECOMMENDED`, `REVIEW_REQUIRED`, `NOT_RECOMMENDED`는 관리자 참고 정보이며 모델을
자동 승격하거나 Serving 트래픽을 변경하지 않습니다.

### Cloud Run Job 환경변수

| 변수 | 용도 |
|---|---|
| `TRAINING_DATA_URI` | 비공개 GCS의 `train1.csv` URI |
| `MLFLOW_EXPERIMENT_NAME` | MLflow 실험 이름 |
| `MLFLOW_REGISTERED_MODEL_NAME` | 기본 `fdshield-fraud-detector-v2` |
| `MLFLOW_MODEL_ALIAS` | 비교할 운영 alias, 기본 `champion` |
| `MODEL_MIN_PR_AUC` | 후보 최소 PR-AUC |
| `MODEL_MIN_RECALL` | 후보 최소 Recall |
| `BACKEND_TRAINING_RUN_ID` | Backend가 만든 학습 실행 ID |
| `TRAINING_RESULT_CALLBACK_URL` | 학습 결과 Callback 주소 |
| `TRAINING_RESULT_CALLBACK_TOKEN` | Secret Manager로 주입할 Callback Token |
| `MLFLOW_TRACKING_URI` | MLflow Tracking Server |
| `MLFLOW_TRACKING_USERNAME` | Secret Manager로 주입할 계정 |
| `MLFLOW_TRACKING_PASSWORD` | Secret Manager로 주입할 비밀번호 |

성공 Callback은 `status`와 `mlflow_run_id`, 실패 Callback은 `status`와 오류 요약을
보냅니다. 모델 버전, 성능 지표와 비교 결과는 MLflow에서 조회하고 Backend
관리자가 승인합니다.

Training 이미지는 `Dockerfile.training`과 `cloudbuild.training.yaml`로 만들며,
배포 Workflow는 Cloud Run Job의 이미지 digest, `TRAINING_DATA_URI`, 등록 모델명을
갱신하고 구형 실행 환경변수를 제거합니다. 이미지 배포 자체가 학습 실행이나 모델
승인을 수행하지는 않습니다.

## Serving

### 로컬 v2 모델

```powershell
Copy-Item .env.example .env
docker compose --env-file .env -f compose.serving.yml up -d --build --wait
```

기본 로컬 번들은 `models/fdshield-fraud-detector-v2`입니다. 이 번들은 전달받은
`xgb_model_weights_12_24_24.json`을 서비스에 이식한 초기 candidate이며, 정규화
수정 이후에는 `train1.csv`로 재학습한 MLflow 모델로 교체해야 합니다.

### MLflow Registry 모델

`.env.serving.example`을 참고해 모델 이름과 정확한 버전을 지정합니다.

```env
ML_PREDICTOR_MODE=mlflow
ML_MODEL_NAME=fdshield-fraud-detector-v2
ML_MODEL_VERSION=1
```

Serving은 alias가 아닌 정확한 Registry 버전을 로드합니다. GitHub의
`ML Serving Cloud Run CD`는 자동 실행되지 않으며, `model_name`과
`model_version`을 입력해 수동 실행합니다. 이 Workflow는 새 이미지를 빌드하고
Cloud Build에서 실제 추론을 검증한 뒤 Cloud Run revision을 **트래픽 0%**로만
준비합니다. 운영 트래픽 전환은 하지 않습니다.

### API

| Method | Path | 설명 |
|---|---|---|
| `GET` | `/health` | 프로세스 상태 |
| `GET` | `/ready` | 모델 로딩 완료 상태 |
| `POST` | `/ml/predict` | 정식 flat raw60 추론 계약 |

응답에는 판정, 확률, 거래별 SHAP과 운영 추적용 `model_name`, `model_version`이
포함됩니다.

## 운영 전환 순서

1. `train1.csv`를 비공개 GCS의 버전 고정 경로에 업로드하고 해시를 확인합니다.
2. Cloud Run Training Job을 실행해 후보 모델을 MLflow Registry에 등록합니다.
3. MLflow의 성능 지표와 모델 비교 결과를 검토합니다.
4. 수동 Serving Workflow에 후보의 정확한 모델명·버전을 넣어 새 코드와 모델을
   트래픽 0% revision으로 준비합니다.
5. Backend 입력 DTO를 raw60 계약으로 전환한 뒤 관리자가 후보를 승인합니다.
6. Backend 승격 API가 tagged revision에 실제 추론 smoke를 수행하고 100% 트래픽
   전환을 완료합니다.

자동 모델 승격은 사용하지 않습니다.

Backend와 ML Serving은 모두 raw60→model80 계약으로 전환되어야 같은 revision을
사용할 수 있습니다. 입력 계약과 모델 Feature 수가 다른 과거 revision으로 되돌리는
것은 지원하지 않습니다.

## 검증

```bash
uv run pytest
```

핵심 검증 항목은 다음과 같습니다.

- train1 raw64 스키마와 `is_fraud` 라벨
- 학습·추론 전처리의 model80 이름과 순서 일치
- 로컬 모델 manifest·해시·Feature 개수
- `/health`, `/ready`, `/ml/predict`
- MLflow 후보 등록·성능 비교·Backend Callback
- 자동 승격 없이 관리자 수동 승인 유지
