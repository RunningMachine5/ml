# FDShield ML

이 레포는 팀원들이 같은 데이터와 평가 기준으로 여러 사기 탐지 모델을 학습하고,
그 결과를 공용 MLflow 서버에서 비교하기 위한 학습 레포입니다.

## MLflow가 하는 일

MLflow 서버가 모델을 대신 학습하는 것은 아닙니다. 학습은 각 팀원의 PC에서
실행하고, MLflow는 학습 한 번을 하나의 `Run`으로 기록합니다.

```text
팀원 PC에서 모델 학습
        |
        | HTTPS로 결과 기록
        v
공용 MLflow 서버 (mlflow.fdshield.cloud)
        |- 사용한 모델과 하이퍼파라미터
        |- PR-AUC, Recall, Precision, F1 등의 평가지표
        |- Confusion Matrix와 분류 리포트
        `- 전처리를 포함한 학습 모델 파일
```

따라서 각자 다른 모델과 하이퍼파라미터를 시험하더라도 MLflow 화면에서 결과를
한곳에 모아 비교할 수 있습니다. Git에는 학습 코드를 저장하고, MLflow에는 실행할
때마다 달라지는 실험 결과와 모델 artifact를 저장합니다.

## 팀 공통 실험 규칙

- 같은 원본 데이터와 공통 전처리·분할 로직을 사용합니다.
- 기존 모델을 비교할 때는 Python 코드를 수정하지 않고 CLI 옵션만 변경합니다.
- `--run-name`에는 본인 이름, 모델, 실험 목적이 드러나게 적습니다.
- 전체 학습 전 `--max-rows`를 사용해 작은 데이터로 먼저 실행합니다.
- 일반 실험은 Run으로만 기록하고, 검증된 후보만 Model Registry에 등록합니다.
- 원본 CSV와 이름·계좌번호 같은 직접 식별자는 MLflow에 업로드하지 않습니다.

현재 학습 코드는 다음 기준을 공통으로 적용합니다.

- `Fraud_Type=m`은 정상(0), `a~l`은 사기(1)로 변환
- `Account_account_number` 기준으로 학습/검증 데이터를 분리해 계좌 누수 방지
- 이름, 식별번호, 계좌번호, IP, MAC, 위치 등 직접 식별자 학습 제외
- 날짜와 시간 간격을 수치 특성으로 변환
- 범주형 Feature를 One-Hot Encoding
- XGBoost의 `scale_pos_weight` 또는 sklearn의 `class_weight`로 클래스 불균형 보정
- Accuracy보다 PR-AUC, Recall, Precision, F1, False Positive Rate 중심으로 비교

## 사용할 수 있는 모델

| `--model-type` | 용도 |
|---|---|
| `logistic-regression` | 단순한 선형 기준 모델 |
| `decision-tree` | 결과를 이해하기 쉬운 단일 트리 모델 |
| `random-forest` | 여러 트리를 사용하는 앙상블 모델 |
| `xgboost` | 성능 비교의 주력 후보 모델 |

기존 네 모델은 `--model-type`과 하이퍼파라미터만 바꿔 실행할 수 있습니다.
새 Feature를 추가하려면 `features.py`, 새로운 모델 종류를 추가하려면
`training.py`와 관련 테스트를 수정해야 합니다.

## 최초 설정

[uv](https://docs.astral.sh/uv/) 설치 후 레포 루트에서 의존성을 설치합니다.

```powershell
uv sync --dev
Copy-Item .env.tracking.example .env.tracking
```

`.env.tracking`에 팀에서 전달받은 MLflow 계정을 입력합니다. 이 파일에는 비밀번호가
들어가므로 Git에 올리면 안 됩니다.

```env
MLFLOW_TRACKING_URI=https://mlflow.fdshield.cloud
MLFLOW_TRACKING_USERNAME=<팀원 계정>
MLFLOW_TRACKING_PASSWORD=<팀원 비밀번호>
MLFLOW_EXPERIMENT_NAME=fdshield-model-comparison
```

연결을 확인합니다.

```powershell
uv run fdshield-mlflow-check --env-file .env.tracking
```

- `401 Unauthorized`: 아이디 또는 비밀번호 확인
- `403 Forbidden`: 해당 실험에 대한 계정 권한 확인
- 연결 시간 초과: 서버 주소, 인터넷 연결 또는 서버 상태 확인

## 데이터 준비

원본 `train.csv`는 Git에 올리지 않습니다. 공유받은 파일을 다음 위치에 복사하거나
학습 명령에서 절대 경로를 지정합니다.

```powershell
Copy-Item '<공유받은 경로>\train.csv' data\open\train.csv
```

## 작은 데이터로 먼저 실행

전체 데이터를 돌리기 전에 2만 건과 작은 XGBoost 모델로 학습부터 MLflow 기록까지
정상 작동하는지 확인합니다.

```powershell
uv run fdshield-train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type xgboost `
  --max-rows 20000 `
  --n-estimators 50 `
  --run-name "<본인이름>-xgb-smoke"
```

성공하면 콘솔에 MLflow Run URL과 주요 평가지표가 출력되고, 공용 MLflow 웹에서
해당 Run을 확인할 수 있습니다.

## 모델별 실험

XGBoost 전체 데이터 기준 실험 예시입니다.

```powershell
uv run fdshield-train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type xgboost `
  --run-name "<본인이름>-xgb-baseline"
```

Random Forest의 파라미터를 바꾸는 예시입니다.

```powershell
uv run fdshield-train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type random-forest `
  --run-name "<본인이름>-rf-depth10" `
  --n-estimators 300 `
  --max-depth 10
```

모델 비교가 목적이라면 한 번에 여러 값을 크게 바꾸기보다 기준 Run에서 파라미터를
하나씩 변경하는 것이 결과를 해석하기 쉽습니다.

## MLflow에서 결과 확인하기

1. `https://mlflow.fdshield.cloud`에 접속합니다.
2. `fdshield-model-comparison` Experiment를 선택합니다.
3. 비교할 Run을 체크하고 Compare를 선택합니다.
4. 같은 검증 데이터 기준으로 PR-AUC, Recall, Precision, F1을 비교합니다.
5. 성능뿐 아니라 False Positive Rate와 Confusion Matrix도 함께 확인합니다.

사기 탐지에서는 지표 하나만으로 최종 모델을 고르면 안 됩니다.

- `Recall`: 실제 사기를 얼마나 놓치지 않았는지
- `Precision`: 사기라고 판단한 거래 중 실제 사기의 비율
- `F1`: Recall과 Precision의 균형
- `PR-AUC`: 클래스 불균형 데이터에서 모델의 전반적인 분류 성능
- `False Positive Rate`: 정상 거래를 사기로 잘못 판단한 비율

## Model Registry 사용 기준

대부분의 팀원은 `--registered-model-name` 없이 실험합니다. 팀에서 결과를 비교한 뒤
실제 배포 후보로 검증할 모델만 Registry에 등록합니다.

```text
일반 학습 결과 -> MLflow Experiment의 Run
성능이 좋은 후보 -> Registry의 candidate
최종 승인 모델 -> Registry의 champion
```

후보 모델을 등록할 때만 다음 옵션을 추가합니다.

```powershell
uv run fdshield-train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type xgboost `
  --run-name "candidate-01" `
  --registered-model-name fdshield-fraud-detector
```

## 코드 위치와 테스트

| 경로 | 역할 |
|---|---|
| `src/fdshield_ml/tracking.py` | MLflow 주소와 인증 설정 및 연결 확인 |
| `src/fdshield_ml/features.py` | 라벨 변환, 식별자 제외, 시간 Feature 생성 |
| `src/fdshield_ml/training.py` | 공통 데이터 분할, 전처리, 모델 생성 및 평가 |
| `src/fdshield_ml/train_xgboost.py` | CLI 입력, 로컬 학습, MLflow 기록 실행 |
| `tests/test_training.py` | 가짜 데이터로 전체 학습 흐름 검증 |

코드를 수정했다면 테스트를 실행합니다.

```powershell
uv run pytest
```

공통 데이터 분할과 평가 방식을 임의로 변경하면 다른 팀원의 Run과 공정하게 비교할
수 없으므로, 변경이 필요한 경우 팀에서 먼저 기준을 합의합니다.
