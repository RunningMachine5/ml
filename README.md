# FDShield ML

FDShield의 팀 공용 MLflow Tracking Server와 사기 탐지 모델 비교 코드를
관리합니다. 학습은 각 팀원의 PC에서 실행하고, 실행 파라미터·평가지표·학습된
모델을 원격 MLflow에 기록합니다.

머신러닝과 MLflow가 처음이라면
[`docs/MODEL_EXPERIMENT_GUIDE.md`](docs/MODEL_EXPERIMENT_GUIDE.md)를 먼저 읽으세요.

```text
팀원 PC의 같은 train.csv
          |
          +-- Logistic Regression
          +-- Decision Tree
          +-- Random Forest
          `-- XGBoost
                  |
                  `-- HTTPS --> mlflow.fdshield.cloud
                                  |- 모델별 파라미터와 평가지표
                                  |- 학습된 Pipeline 모델
                                  `- 입력 스키마와 평가 결과
```

## 처음 보는 팀원을 위한 핵심 설명

MLflow 서버가 모델을 대신 학습하는 것은 아닙니다. 각 팀원 PC에서 학습하고,
MLflow는 학습 한 번을 Run으로 저장해 여러 결과를 한 화면에서 비교합니다.

기존 네 모델을 실험할 때는 Python 코드를 직접 수정할 필요가 없습니다.
`fdshield-train` 명령에서 `--model-type`, `--run-name`, 해당 모델의
하이퍼파라미터만 바꿉니다. 데이터 분할과 Feature 처리를 각자 수정하면 공정한
비교가 되지 않습니다.

각 파일의 역할은 다음과 같습니다.

| 파일 | 역할 |
|---|---|
| `tracking.py` | MLflow 서버 주소와 로그인 정보를 설정하고 연결을 확인 |
| `features.py` | 라벨 변환, 식별자 제외, 시간 특성 생성 |
| `training.py` | 공통 데이터 분할·전처리·네 모델 생성·성능 평가 |
| `train_xgboost.py` | CLI 입력부터 로컬 학습과 MLflow 기록까지 전체 흐름 |
| `MODEL_EXPERIMENT_GUIDE.md` | 모델별 설명, 실행 예시, MLflow 지표 해석 |

실행하면 MLflow에 다음 정보가 기록됩니다.

- Run 이름, 모델 종류, 해당 모델이 실제 사용한 하이퍼파라미터
- 학습/검증 행 수와 클래스 비율
- PR-AUC, Recall, Precision, F1, False Positive Rate
- Confusion Matrix와 분류 리포트
- 모델 입력 컬럼과 제외한 식별자 목록
- 전처리와 분류기를 합친 학습 모델 artifact

원본 `train.csv` 전체와 직접 식별자는 업로드하지 않습니다. 모델 artifact에는
서빙 입력 형식을 보여주기 위해 직접 식별자를 제외한 입력 예시 5행이 포함됩니다.

### 코드를 바꿔야 하는 경우

| 실험 종류 | 코드 수정 여부 |
|---|---|
| 네 모델 중 하나 선택 | `--model-type` 사용, 코드 수정 불필요 |
| 모델 하이퍼파라미터 변경 | CLI 옵션 사용, 코드 수정 불필요 |
| 새로운 Feature를 추가하거나 제외 | `features.py` 변경 필요 |
| LightGBM처럼 새 모델 종류 추가 | `training.py`와 테스트 변경 필요 |

## 포함된 모델

| `--model-type` | 비교 목적 |
|---|---|
| `logistic-regression` | 단순한 선형 기준 모델 |
| `decision-tree` | 이해하기 쉬운 단일 트리 기준 모델 |
| `random-forest` | 여러 트리를 평균내는 앙상블 기준 모델 |
| `xgboost` | 오차를 순차적으로 보완하는 주력 후보 |

## 학습 예제가 하는 일

- `Fraud_Type=m`은 정상(0), `a~l`은 사기(1)로 변환
- `Account_account_number` 기준으로 학습/검증 데이터를 분리해 계좌 누수 방지
- 이름, 식별번호, 계좌번호, IP, MAC, 위치 등 직접 식별자 학습 제외
- 날짜와 시간 간격을 수치 특성으로 변환
- 범주형 One-Hot Encoding 후 선택한 이진분류 모델 학습
- 불균형을 XGBoost의 `scale_pos_weight` 또는 sklearn의 `class_weight`로 보정
- Accuracy보다 PR-AUC, Recall, Precision, F1, False Positive Rate 중심으로 기록
- 전처리와 분류기가 합쳐진 모델 artifact를 MLflow에 저장

## 팀원 최초 설정

[uv](https://docs.astral.sh/uv/) 설치 후 레포 루트에서 의존성을 설치합니다.

```powershell
uv sync --dev
Copy-Item .env.tracking.example .env.tracking
```

`.env.tracking`에 본인에게 발급된 MLflow 계정을 입력합니다. 관리자 계정을 팀
전체가 공유하지 않는 것을 권장합니다.

```env
MLFLOW_TRACKING_URI=https://mlflow.fdshield.cloud
MLFLOW_TRACKING_USERNAME=<팀원 계정>
MLFLOW_TRACKING_PASSWORD=<팀원 비밀번호>
MLFLOW_EXPERIMENT_NAME=fdshield-model-comparison
```

연결부터 확인합니다.

```powershell
uv run fdshield-mlflow-check --env-file .env.tracking
```

`401`은 계정 정보 오류, `403`은 해당 실험에 대한 권한 부족을 의미합니다.

## 데이터 준비

`open/train.csv`는 Git에 올리지 않습니다. 아래 둘 중 하나를 사용합니다.

1. `data/open/train.csv`로 복사
2. 학습할 때 `--data-path`로 PC의 원본 절대 경로 지정

```powershell
Copy-Item '<공유받은 경로>\train.csv' data\open\train.csv
```

## 먼저 작은 테스트 실행

전체 데이터를 돌리기 전에 2만 건과 작은 XGBoost 모델로 흐름을 확인합니다.

```powershell
uv run fdshield-train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type xgboost `
  --max-rows 20000 `
  --n-estimators 50 `
  --run-name "<본인이름>-xgb-smoke"
```

성공하면 콘솔에 MLflow Run URL과 PR-AUC, Recall, Precision, F1이 출력됩니다.

## 전체 데이터 학습

```powershell
uv run fdshield-train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type xgboost `
  --run-name "<본인이름>-xgb-baseline"
```

모델을 바꾸려면 `--model-type`과 그 모델의 하이퍼파라미터만 변경합니다.

```powershell
uv run fdshield-train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type random-forest `
  --run-name "<본인이름>-rf-baseline" `
  --n-estimators 300 `
  --max-depth 10
```

네 모델의 전체 명령과 파라미터 설명은
[`MODEL_EXPERIMENT_GUIDE.md`](docs/MODEL_EXPERIMENT_GUIDE.md)를 확인하세요.

모델 Registry까지 쓸 때만 이름을 추가합니다. 권한이 없다면 이 옵션은 빼야
합니다.

일반 실험은 모두 MLflow Experiment에 기록하되 Registry에는 등록하지 않습니다.
팀에서 성능을 비교한 뒤 실제 배포 후보로 검증할 모델만 Registry에 등록합니다.

```text
일반 학습 결과 -> MLflow Experiment의 Run
성능이 좋은 후보 -> Registry의 candidate
최종 승인 모델 -> Registry의 champion
```

따라서 대부분의 팀원은 `--registered-model-name` 없이 실행하면 됩니다. 후보
모델을 등록할 때만 아래 옵션을 사용합니다.

```powershell
uv run fdshield-train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type xgboost `
  --run-name "candidate-01" `
  --registered-model-name fdshield-fraud-detector
```

## 코드 위치와 테스트

- `src/fdshield_ml/features.py`: 라벨 변환, 식별자 제외, 시간 특성 생성
- `src/fdshield_ml/training.py`: 그룹 분할, 전처리, 네 모델 생성, 평가
- `src/fdshield_ml/train_xgboost.py`: CLI와 MLflow 기록
- `tests/test_training.py`: 가짜 데이터로 네 모델의 전체 학습 흐름 검증

```powershell
uv run pytest
```

팀원이 기존 네 모델을 실험할 때는 Python 코드를 고치지 않습니다. 새 모델을 코드에
추가할 때도 공통 데이터 분할과 평가 함수를 유지해야 결과 비교가 공정합니다.

---

## MLflow Tracking Server 배포

현재 배포 구성은 다음과 같습니다.

- 기존 `backend/docker-compose.yml`의 ParadeDB 재사용
- ParadeDB 안에 `mlflow` 논리 데이터베이스 자동 생성
- MLflow 3.13.0 Tracking Server와 기본 인증
- Artifact는 `mlflow_artifacts` Docker 볼륨에 저장
- MLflow 5000번 포트는 VM의 `127.0.0.1`에만 공개
- 공용 Nginx가 `mlflow.fdshield.cloud` 요청을 MLflow 컨테이너로 전달
- 현재 Nginx 설정은 인증서 발급 전 HTTP 부트스트랩 단계

아래 명령은 세 레포의 상위 폴더에서 실행합니다.

```powershell
Copy-Item ml/.env.example ml/.env
```

`ml/.env`의 DB 비밀번호, MLflow 관리자 비밀번호, CSRF 비밀키를 변경한 다음
실행합니다.

```powershell
docker compose `
  --project-name fdshield `
  --env-file ml/.env `
  -f backend/docker-compose.yml `
  -f ml/deploy/compose.mlflow.yml `
  up -d --build db mlflow-db-init mlflow nginx
```

상태와 로그를 확인합니다.

```powershell
docker compose `
  --project-name fdshield `
  --env-file ml/.env `
  -f backend/docker-compose.yml `
  -f ml/deploy/compose.mlflow.yml `
  ps

docker compose `
  --project-name fdshield `
  --env-file ml/.env `
  -f backend/docker-compose.yml `
  -f ml/deploy/compose.mlflow.yml `
  logs --tail 100 mlflow
```

VM 안에서는 먼저 MLflow와 Nginx를 각각 확인합니다.

```bash
curl -I http://127.0.0.1:5000
curl -I -H 'Host: mlflow.fdshield.cloud' http://127.0.0.1
```

두 번째 요청이 MLflow의 로그인 응답을 반환하면 Nginx의 HTTP Reverse Proxy가
연결된 것입니다. 현재 `deploy/nginx/mlflow.conf`는 Let's Encrypt 인증서를 발급할
수 있도록 80번 포트를 먼저 여는 부트스트랩 설정입니다. 인증서를 발급하고 HTTPS
설정을 적용한 후에만 팀원들의 `.env.tracking`에서
`https://mlflow.fdshield.cloud`를 사용합니다.
