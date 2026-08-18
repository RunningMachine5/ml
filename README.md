# FDShield ML

FDShield의 사기 거래 XGBoost 학습·추론 서비스입니다. 모델 계산은 ML 담당자가
전달한 `train1.csv`, 79개 모델 Feature, XGBoost 설정을 기준으로 하며 기존의
GCS, Cloud Run Job, MLflow Registry, Backend Callback, 관리자 수동 승인 인프라는
그대로 사용합니다.

## 고정 계약

| 구분 | 계약 |
|---|---|
| 추론 API 입력 | Backend가 계산한 flat raw51 |
| 학습 CSV 입력 | 원본·메타데이터·라벨을 포함한 train1 raw64 |
| 전처리 출력 | 이름과 순서가 고정된 숫자 model79 |
| 모델 | XGBoost binary logistic |
| 기본 판정 임계값 | `0.5` |
| Registry 모델명 | `fdshield-fraud-detector-v2` |
| 로컬 모델 번들 | `models/fdshield-fraud-detector-v2` |

학습과 추론은 모두 `fdshield_ml/service/preprocessor.py`의 같은 raw51 전처리를
사용합니다. `transaction_id`는 Backend에서 관리하며 ML 요청과 model79에는
들어가지 않습니다.
학습 raw64에는 raw51 외에 고객·계좌 식별값과 학습 메타데이터·라벨이 포함됩니다.
CSV의 기존 이름인 `account_release_suspention`, `transaction_resumed_date`,
`flag_deposit_more_than_tenmillion`은 로딩할 때 현재 계약 이름으로 바꿉니다.

전달본 DTO·CSV의 기존 불일치는 실제 데이터와 model79 계약을 기준으로
보정합니다. `account_remaining_amount_daily_limit_exceeded`는 bool이 아닌
숫자형 금액으로 받고, Channel·Operating System은 공백 제거 후 소문자로
정규화합니다. Operating System과 Access Medium은 nullable이며 값이 없으면
각 One-hot 그룹을 모두 0으로 둡니다. `account_account_type=e`도 model79의
`a~d` One-hot을 모두 0으로 두는 unseen category로 허용하므로 Feature 추가나
별도 열 추가는 하지 않습니다. 계좌 최초 잔액·현재 잔액·일일 한도 초과 잔액은
Backend 계약대로 정수로 받습니다. 외부 canonical 입금 플래그 이름은
`flag_deposit_more_than_ten_million`을 사용하고 원본 CSV는 수정하지 않습니다.

## 코드 구조

```text
fdshield_ml/
├── config/preprocess_config.py      # raw51/raw64/model79 컬럼 계약
├── dto/                              # PredictInputDTO, PredictResultDTO
├── service/                          # ML 담당자가 주로 수정하는 핵심 코드
│   ├── preprocessor.py               # 학습·추론 공용 raw51 -> model79
│   ├── predict/                      # 확률·SHAP·예측 흐름
│   └── train/                        # raw64 검증·XGBoost 학습 흐름
├── infrastructure/                   # 운영 환경 연동
│   ├── data_source.py                # 로컬·GCS 학습 데이터
│   ├── model_loader.py               # 로컬·MLflow 고정 모델 로딩
│   ├── mlflow.py                     # 성능 비교·후보 Registry 등록
│   ├── training_pipeline.py          # core 학습과 MLflow 운영 조립
│   └── backend_callback.py           # 학습 결과 Callback
├── api/ml_input.py                   # /ml/predict 라우트
├── serving.py                        # FastAPI/Cloud Run Serving 진입점
├── training_job.py                   # Cloud Run Training Job 진입점
└── local_train.py                    # MLflow 없는 로컬 학습 명령

models/fdshield-fraud-detector-v2/  # 전달 모델 기반 로컬 candidate
data/open/train1.csv                # 로컬 학습 파일, Git 제외
```

### 처음 보는 순서

1. `config/preprocess_config.py`에서 raw51·raw64·model79 컬럼 계약을 확인합니다.
2. `service/preprocessor.py`에서 학습과 추론이 공유하는 변환을 확인합니다.
3. 실시간 추론은 `serving.py` → `api/ml_input.py` →
   `service/predict/predict_service.py` 순으로 읽습니다.
4. 학습은 `service/train/dataset.py` → `service/train/model_training.py` →
   `service/train/train_service.py` → `training_job.py` 순으로 읽습니다.
5. GCS·MLflow·Backend Callback·모델 로딩은 `infrastructure/`에서 확인합니다.

ML 계산 코드는 `service/`, 배포 환경과 연결되는 코드는 `infrastructure/`에
분리합니다. 따라서 전처리와 모델 학습을 수정할 때 Cloud Run이나 Callback
구현을 따라갈 필요가 없습니다.

`config`, `dto`, `service/predict`, `service/train`, `api` 순서는 doo 원본의
`app/` 구조와 같습니다. 저장소 패키지 이름만 `fdshield_ml`로 유지해 Docker와
Cloud Run의 기존 실행 경로가 바뀌지 않게 했습니다.

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

### MLflow 없이 로컬에서 학습하기

전처리나 XGBoost 설정을 실험할 때는 GCS·MLflow·Backend Callback 설정 없이
같은 core 학습 함수를 실행할 수 있습니다.

```powershell
uv run python -m fdshield_ml.local_train `
  --data data/open/train1.csv `
  --output-dir models/local-training-output `
  --model-name fdshield-fraud-detector-v2 `
  --model-version 1
```

결과 폴더에 `model.json`, `manifest.json`, `metrics.json`이 생성되고,
저장 직후 실제 Serving loader로 모델 해시·model79 Feature 순서·판정 임계값을
검증합니다. 기존 파일을 덮어쓰지 않도록 출력 폴더가 비어 있지 않으면
실행을 거절합니다.

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

Training Job은 raw64 스키마, 라벨, 거래 ID와 model79 전처리를 검증한 뒤
stratified 80/20 분할로 후보 모델을 학습해 Registry에 등록합니다. 별도의 Stub이나
검증 전용 실행 모드는 지원하지 않습니다.

후보 모델은 PR-AUC, ROC-AUC, Recall, Precision, F1, FPR과 판정 임계값을
MLflow에 저장합니다. 같은 model79 계약의 현재 champion이 있으면 같은 검증
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
| `MLFLOW_TRACKING_URI` | GitHub Repository Variable에서 일반 환경변수로 주입할 MLflow Tracking Server URI |
| `MLFLOW_TRACKING_USERNAME` | Secret Manager로 주입할 계정 |
| `MLFLOW_TRACKING_PASSWORD` | Secret Manager로 주입할 비밀번호 |

성공 Callback은 `status`와 `mlflow_run_id`, 실패 Callback은 `status`와 오류 요약을
보냅니다. 모델 버전, 성능 지표와 비교 결과는 MLflow에서 조회하고 Backend
관리자가 승인합니다.

Training 이미지는 `Dockerfile.training`과 `cloudbuild.training.yaml`로 만들며,
CI와 Cloud Build는 컨테이너를 실제 실행해 `fdshield_ml.training_job` 진입점까지
도달하는지 확인합니다. 필수 설정을 주입하지 않은 smoke에서 종료 코드 `2`와
`training_job_configuration_error`가 확인되어야 이미지 빌드가 통과합니다.

배포 Workflow에는 다음 GitHub Repository Variable이 필요합니다.

| Variable | 용도 |
|---|---|
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | GitHub OIDC Workload Identity Provider |
| `GCP_SERVICE_ACCOUNT` | 배포 Workflow가 impersonate할 서비스 계정 |
| `TRAINING_JOB_SERVICE_ACCOUNT` | GCS·Secret Manager에 최소 권한을 가진 Training 전용 실행 계정 |
| `TRAINING_DATA_URI` | 버전이 고정된 비공개 GCS CSV URI |
| `TRAINING_RESULT_CALLBACK_URL` | `{training_run_id}`를 포함한 HTTPS Callback 주소 |
| `MLFLOW_TRACKING_URI` | 자격 증명을 포함하지 않은 MLflow HTTP(S) 주소 |
| `MLFLOW_TRACKING_USERNAME_SECRET` | MLflow 계정이 저장된 Secret 이름 |
| `MLFLOW_TRACKING_PASSWORD_SECRET` | MLflow 비밀번호가 저장된 Secret 이름 |
| `TRAINING_RESULT_CALLBACK_TOKEN_SECRET` | Backend Callback Token이 저장된 Secret 이름 |

Workflow는 `MLFLOW_TRACKING_URI`를 일반 Cloud Run 환경변수로 설정합니다. URI는
HTTP(S) 주소만 허용하고 사용자명·비밀번호, query, fragment, 공백과 쉼표를
거절합니다. `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`,
`TRAINING_RESULT_CALLBACK_TOKEN`만 Secret Manager의 `latest` 버전으로 연결합니다.
Secret 값은 GitHub에 저장하거나 배포 로그에 출력하지 않으며, 배포 후에는 값이 아닌
Secret 참조 구조만 검사합니다. Secret 생성·값 변경은 Workflow가 수행하지 않습니다.

기존 Job의 `MLFLOW_TRACKING_URI`가 Secret 참조이면 먼저 해당 참조를 별도 update로
제거한 뒤 일반 URI를 설정합니다. 반대로 사용자명·비밀번호·Callback Token이 평문
환경변수이면 해당 binding을 먼저 제거한 뒤 Secret 참조를 주입합니다. 서로 다른
타입의 binding을 한 update에서 교체하지 않으며, 배포 Workflow는 Job을 자동 실행하지
않습니다.

배포 시 이미지 digest, 전용 서비스 계정, 학습 데이터·실험명·등록 모델명·Callback
주소를 갱신합니다. 과거 Job의 command/args override는 빈 값으로 초기화해 이미지의
`CMD ["python", "-m", "fdshield_ml.training_job"]`를 사용하도록 강제합니다. 이후
이미지 digest, command/args, 서비스 계정, 일반 환경변수, Secret 참조, 구형 환경변수
부재를 다시 검사합니다. 이미지 배포 자체가 학습 실행이나 모델 승인을 수행하지는
않습니다.

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
준비합니다. revision tag는 Backend 계약과 동일한 `model-v<model_version>`이며,
운영 트래픽 전환은 하지 않습니다.

동일한 모델 버전으로 Workflow를 재실행했을 때 기존 tag가 최신 Ready revision,
검증된 image digest, 정확한 모델 환경변수, 트래픽 0%를 모두 만족하면 해당 revision을
그대로 재사용합니다. 하나라도 다르면 tag를 다른 revision으로 옮기지 않고 배포를
실패시킵니다. 따라서 다른 코드나 모델 설정을 준비하려면 새 모델 버전을 사용해야
합니다.

기존 active revision에서 이어받는 `MLFLOW_TRACKING_URI`는 HTTP(S) origin 또는
origin 뒤 path만 허용합니다. URL 안의 사용자명·비밀번호, query, fragment는 배포
전에 거절하며 MLflow 자격 증명은 별도의 Secret Manager 참조로만 전달합니다.
candidate는 최신 created 및 최신 Ready revision이어야 하고, Service와 revision의
generation 관찰 및 `Ready=True`까지 끝나야 Backend 승인 대상으로 인정됩니다.

### API

| Method | Path | 설명 |
|---|---|---|
| `GET` | `/health` | 프로세스 상태 |
| `GET` | `/ready` | 모델 로딩 완료 상태 |
| `POST` | `/ml/predict` | 정식 flat raw51 추론 계약 |

응답에는 판정, 확률, 거래별 SHAP과 운영 추적용 `model_name`, `model_version`이
포함됩니다.

## 운영 전환 순서

1. `train1.csv`를 비공개 GCS의 버전 고정 경로에 업로드하고 해시를 확인합니다.
2. Cloud Run Training Job을 실행해 후보 모델을 MLflow Registry에 등록합니다.
3. MLflow의 성능 지표와 모델 비교 결과를 검토합니다.
4. 수동 Serving Workflow에 후보의 정확한 모델명·버전을 넣어 새 코드와 모델을
   트래픽 0% revision으로 준비합니다.
5. Backend 입력 DTO를 raw51 계약으로 전환한 뒤 관리자가 후보를 승인합니다.
6. Backend 승격 API가 tagged revision에 실제 추론 smoke를 수행하고 100% 트래픽
   전환을 완료합니다.

자동 모델 승격은 사용하지 않습니다.

Backend와 ML Serving은 모두 raw51→model79 계약으로 전환되어야 같은 revision을
사용할 수 있습니다. 입력 계약과 모델 Feature 수가 다른 과거 revision으로 되돌리는
것은 지원하지 않습니다.

## 검증

```bash
uv run pytest
```

핵심 검증 항목은 다음과 같습니다.

- train1 raw64 스키마와 `is_fraud` 라벨
- 학습·추론 전처리의 model79 이름과 순서 일치
- 로컬 모델 manifest·해시·Feature 개수
- `/health`, `/ready`, `/ml/predict`
- MLflow 후보 등록·성능 비교·Backend Callback
- 자동 승격 없이 관리자 수동 승인 유지
