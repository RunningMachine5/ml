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
새 Feature를 추가하려면 `fdshield_ml/common/features.py`, 새로운 모델 종류를
추가하려면 `fdshield_ml/training/pipeline.py`와 관련 테스트를 수정해야 합니다.

## 최초 설정

[uv](https://docs.astral.sh/uv/) 설치 후 레포 루트에서 의존성을 설치합니다.

PowerShell:

```powershell
uv sync
Copy-Item .env.tracking.example .env.tracking
```

Linux/macOS/WSL (Bash):

```bash
uv sync
cp .env.tracking.example .env.tracking
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

Windows/Linux 공통:

```console
uv run python -m fdshield_ml.training.tracking --env-file .env.tracking
```

- `401 Unauthorized`: 아이디 또는 비밀번호 확인
- `403 Forbidden`: 해당 실험에 대한 계정 권한 확인
- 연결 시간 초과: 서버 주소, 인터넷 연결 또는 서버 상태 확인

## 데이터 준비

원본 `train.csv`는 Git에 올리지 않습니다. 공유받은 파일을 다음 위치에 복사하거나
학습 명령에서 절대 경로를 지정합니다.

PowerShell:

```powershell
Copy-Item '<공유받은 경로>\train.csv' data\open\train.csv
```

Linux/macOS/WSL (Bash):

```bash
cp '<공유받은 경로>/train.csv' data/open/train.csv
```

## 작은 데이터로 먼저 실행

전체 데이터를 돌리기 전에 2만 건과 작은 XGBoost 모델로 학습부터 MLflow 기록까지
정상 작동하는지 확인합니다.

PowerShell:

```powershell
uv run python -m fdshield_ml.training.train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type xgboost `
  --max-rows 20000 `
  --n-estimators 50 `
  --run-name "<테스트>-xgb-smoke"
```

명령 프롬프트(CMD):

```bat
uv run python -m fdshield_ml.training.train ^
  --env-file .env.tracking ^
  --data-path data/open/train.csv ^
  --model-type xgboost ^
  --max-rows 20000 ^
  --n-estimators 50 ^
  --run-name "<테스트>-xgb-smoke"
```

Linux/macOS/WSL (Bash):

```bash
uv run python -m fdshield_ml.training.train \
  --env-file .env.tracking \
  --data-path data/open/train.csv \
  --model-type xgboost \
  --max-rows 20000 \
  --n-estimators 50 \
  --run-name "<테스트>-xgb-smoke"
```

PowerShell의 줄 연결 문자는 백틱(`` ` ``), CMD는 캐럿(`^`), Bash는 역슬래시(`\`)입니다.
줄 연결 문자 뒤에 공백을 넣으면 다음 줄과 이어지지 않으므로 주의합니다. 한 줄로
전부 입력해도 결과는 같습니다. `--run-name`은 MLflow 화면에서 보이는 제목이므로
한글과 영문을 모두 사용할 수 있고, 모델 성능에는 영향을 주지 않습니다.

성공하면 콘솔에 MLflow Run URL과 주요 평가지표가 출력되고, 공용 MLflow 웹에서
해당 Run을 확인할 수 있습니다.

## 수동 하이퍼파라미터 실험

수동 실험은 팀원이 직접 파라미터 값을 정해 한 번씩 실행하는 방식입니다. 처음에는
기본값으로 Baseline Run을 만든 다음, 한두 개의 값만 변경하면 성능 변화의 원인을
파악하기 쉽습니다. 선택한 모델에서 사용하지 않는 옵션은 모델 생성에 반영되지 않고
MLflow에도 모델 파라미터로 기록되지 않습니다.

### 모든 모델의 공통 옵션

| 옵션 | 기본값 | 의미 및 사용 기준 |
|---|---:|---|
| `--model-type` | `xgboost` | `logistic-regression`, `decision-tree`, `random-forest`, `xgboost` 중 선택 |
| `--run-name` | 자동 생성 | MLflow에 표시할 실험 제목. `<이름>-<모델>-<목적>` 형태 권장 |
| `--comparison-group` | 미지정 | 같은 데이터·분할 조건으로 비교할 Run에 공통 이름 지정. 팀 비교 시 같은 값을 사용 |
| `--max-rows` | 전체 | 빠른 확인에 사용할 최대 행 수. 최종 비교에서는 생략 |
| `--test-size` | `0.2` | 전체 중 검증 데이터 비율. 팀 비교 중에는 변경하지 않음 |
| `--random-state` | `42` | 표본 추출, 데이터 분할, 모델 난수 시드. 팀 비교 중에는 고정 |
| `--n-jobs` | 최대 4 | 모델 학습에 사용할 CPU 작업 수. 개인 PC 상황에 맞게 조정 |
| `--registered-model-name` | 미지정 | Registry에 등록할 때만 사용. 일반 실험에서는 생략 |

현재 `--test-size`로 분리한 데이터는 이름과 달리 **검증(validation) 데이터**로
사용합니다. 여러 파라미터를 이 데이터에서 반복 비교하므로 최종 성능을 보증하는
독립 Test Set은 아닙니다. 팀이 최종 모델을 정할 때는 시연용 입력과 별도로 보관한
라벨 포함 Test Set에서 한 번 더 평가하는 것이 올바른 구조입니다.

### Logistic Regression

가장 단순한 기준 모델입니다. 복잡한 트리 모델이 정말 개선됐는지 확인하는 Baseline으로
먼저 실행하는 것을 권장합니다.

| 옵션 | 기본값 | 의미 | 처음 비교할 값 |
|---|---:|---|---|
| `--logistic-c` | `1.0` | 규제 강도의 역수. 작을수록 규제가 강해져 모델이 단순해짐 | `0.01`, `0.1`, `1`, `10` |
| `--logistic-max-iter` | `1000` | 학습 최대 반복 횟수. 수렴 경고가 날 때 늘림 | `1000`, `2000` |

PowerShell:

```powershell
uv run python -m fdshield_ml.training.train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type logistic-regression `
  --logistic-c 0.1 `
  --run-name "테스트-logistic-c0.1"
```

Linux/macOS/WSL (Bash):

```bash
uv run python -m fdshield_ml.training.train \
  --env-file .env.tracking \
  --data-path data/open/train.csv \
  --model-type logistic-regression \
  --logistic-c 0.1 \
  --run-name "테스트-logistic-c0.1"
```

### Decision Tree

트리 한 개를 사용해 구조가 단순하고 설명하기 쉽지만, 깊이가 커지면 학습 데이터에
과적합되기 쉽습니다.

| 옵션 | 기본값 | 의미 | 처음 비교할 값 |
|---|---:|---|---|
| `--max-depth` | `5` | 트리의 최대 깊이. 클수록 복잡한 규칙을 학습 | `3`, `5`, `8`, `12` |
| `--min-samples-leaf` | `5` | 하나의 최종 리프에 필요한 최소 샘플 수 | `1`, `5`, `10`, `20` |

PowerShell:

```powershell
uv run python -m fdshield_ml.training.train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type decision-tree `
  --max-depth 8 `
  --min-samples-leaf 10 `
  --run-name "테스트-tree-depth8-leaf10"
```

Linux/macOS/WSL (Bash):

```bash
uv run python -m fdshield_ml.training.train \
  --env-file .env.tracking \
  --data-path data/open/train.csv \
  --model-type decision-tree \
  --max-depth 8 \
  --min-samples-leaf 10 \
  --run-name "테스트-tree-depth8-leaf10"
```

### Random Forest

서로 다른 여러 트리의 결과를 합쳐 단일 Decision Tree보다 안정적인 성능을 기대하는
모델입니다. 트리 수와 깊이를 늘릴수록 대체로 학습 시간과 메모리 사용량도 증가합니다.

| 옵션 | 기본값 | 의미 | 처음 비교할 값 |
|---|---:|---|---|
| `--n-estimators` | `300` | 생성할 트리 개수 | `100`, `300`, `500` |
| `--max-depth` | `5` | 각 트리의 최대 깊이 | `5`, `10`, `15` |
| `--min-samples-leaf` | `5` | 리프의 최소 샘플 수 | `1`, `5`, `10` |
| `--max-features` | `sqrt` | 각 분기에서 후보로 볼 Feature 수 방식 | `sqrt`, `log2` |

PowerShell:

```powershell
uv run python -m fdshield_ml.training.train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type random-forest `
  --n-estimators 300 `
  --max-depth 10 `
  --min-samples-leaf 5 `
  --max-features sqrt `
  --run-name "테스트-rf-depth10"
```

Linux/macOS/WSL (Bash):

```bash
uv run python -m fdshield_ml.training.train \
  --env-file .env.tracking \
  --data-path data/open/train.csv \
  --model-type random-forest \
  --n-estimators 300 \
  --max-depth 10 \
  --min-samples-leaf 5 \
  --max-features sqrt \
  --run-name "테스트-rf-depth10"
```

### XGBoost

이번 프로젝트의 주력 성능 후보입니다. `learning_rate`를 낮추면 보통 더 많은 트리가
필요하므로 `n_estimators`와 함께 조정합니다. 너무 깊은 트리나 너무 많은 트리는
학습 시간을 늘리고 과적합을 만들 수 있습니다.

| 옵션 | 기본값 | 의미 | 처음 비교할 값 |
|---|---:|---|---|
| `--n-estimators` | `300` | 순차적으로 학습할 트리 개수 | `100`, `300`, `500` |
| `--max-depth` | `5` | 각 트리의 최대 깊이 | `3`, `5`, `8` |
| `--learning-rate` | `0.05` | 트리 하나의 보정 반영 크기 | `0.01`, `0.05`, `0.1` |
| `--subsample` | `0.8` | 트리마다 사용할 학습 행 비율 | `0.6`, `0.8`, `1.0` |
| `--colsample-bytree` | `0.8` | 트리마다 사용할 Feature 비율 | `0.6`, `0.8`, `1.0` |
| `--min-child-weight` | `1.0` | 자식 노드 분할에 필요한 최소 가중치 | `1`, `5`, `10` |

PowerShell:

```powershell
uv run python -m fdshield_ml.training.train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type xgboost `
  --n-estimators 300 `
  --max-depth 5 `
  --learning-rate 0.05 `
  --subsample 0.8 `
  --colsample-bytree 0.8 `
  --min-child-weight 1 `
  --run-name "테스트-xgb-baseline"
```

CMD:

```bat
uv run python -m fdshield_ml.training.train ^
  --env-file .env.tracking ^
  --data-path data/open/train.csv ^
  --model-type xgboost ^
  --n-estimators 300 ^
  --max-depth 5 ^
  --learning-rate 0.05 ^
  --subsample 0.8 ^
  --colsample-bytree 0.8 ^
  --min-child-weight 1 ^
  --run-name "테스트-xgb-baseline"
```

Linux/macOS/WSL (Bash):

```bash
uv run python -m fdshield_ml.training.train \
  --env-file .env.tracking \
  --data-path data/open/train.csv \
  --model-type xgboost \
  --n-estimators 300 \
  --max-depth 5 \
  --learning-rate 0.05 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --min-child-weight 1 \
  --run-name "테스트-xgb-baseline"
```

모델 비교가 목적이라면 한 번에 여러 값을 크게 바꾸기보다 기준 Run에서 파라미터를
하나씩 변경하는 것이 결과를 해석하기 쉽습니다.

## Optuna 자동 하이퍼파라미터 탐색

Optuna는 사람이 값을 하나씩 입력하는 대신 여러 파라미터 조합을 제안하고, 검증
`PR-AUC`가 가장 높은 조합을 찾는 라이브러리입니다. 이 레포에서는 다음처럼 동작합니다.

```text
Optuna Study 1개 = 한 모델 종류의 전체 탐색 작업
  ├─ Trial 0 = 첫 번째 파라미터 조합 학습 + 검증 PR-AUC
  ├─ Trial 1 = 두 번째 파라미터 조합 학습 + 검증 PR-AUC
  ├─ ...
  └─ Best Trial 파라미터로 모델을 다시 학습하고 모델 artifact 저장
```

- 학습과 Optuna 실행은 **팀원 PC**에서 수행합니다.
- 모든 Trial은 같은 학습/검증 분할을 사용하므로 파라미터만 비교합니다.
- 공용 MLflow에는 부모 Study Run과 그 아래의 Trial Run들이 기록됩니다.
- 각 Trial은 파라미터·평가지표·학습 시간만 기록해 서버 저장 공간을 아낍니다.
- 탐색 종료 후 Best Trial의 모델만 부모 Run의 `model` artifact로 저장합니다.
- `--registered-model-name`을 지정한 경우에도 Best 모델 하나만 Registry에 등록합니다.

### 자동 탐색 범위

| 모델 | Optuna가 자동으로 바꾸는 값 |
|---|---|
| Logistic Regression | `logistic_c`: `0.001`~`100` 로그 스케일 |
| Decision Tree | `max_depth`: `2`~`16`, `min_samples_leaf`: `1`~`50` |
| Random Forest | `n_estimators`: `100`~`500`, `max_depth`: `3`~`20`, `min_samples_leaf`: `1`~`30`, `max_features`: `sqrt`/`log2` |
| XGBoost | `n_estimators`: `100`~`600`, `max_depth`: `3`~`10`, `learning_rate`: `0.01`~`0.3`, `subsample`: `0.6`~`1.0`, `colsample_bytree`: `0.6`~`1.0`, `min_child_weight`: `1`~`20` |

탐색 범위는 `fdshield_ml/training/tuning.py` 한 곳에서 관리합니다. 초보 팀원은 이 파일을
수정하지 말고 모델 종류, Trial 수, 표본 크기만 CLI에서 선택하면 됩니다.

### 1단계: Optuna Smoke Test

처음에는 2만 행과 5회 Trial로 접속, 학습, MLflow 기록이 모두 되는지만 확인합니다.

PowerShell:

```powershell
uv run python -m fdshield_ml.training.tune `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type xgboost `
  --max-rows 20000 `
  --n-trials 5 `
  --study-name "테스트-xgb-optuna-smoke"
```

CMD:

```bat
uv run python -m fdshield_ml.training.tune ^
  --env-file .env.tracking ^
  --data-path data/open/train.csv ^
  --model-type xgboost ^
  --max-rows 20000 ^
  --n-trials 5 ^
  --study-name "테스트-xgb-optuna-smoke"
```

Linux/macOS/WSL (Bash):

```bash
uv run python -m fdshield_ml.training.tune \
  --env-file .env.tracking \
  --data-path data/open/train.csv \
  --model-type xgboost \
  --max-rows 20000 \
  --n-trials 5 \
  --study-name "테스트-xgb-optuna-smoke"
```

### 2단계: 전체 데이터 자동 탐색

Smoke Test가 성공하면 `--max-rows`를 빼고 20회 정도부터 시작합니다. XGBoost는 각
Trial마다 모델 하나를 새로 학습하므로 Trial 수를 무작정 크게 잡지 않습니다.

PowerShell:

```powershell
uv run python -m fdshield_ml.training.tune `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type xgboost `
  --n-trials 20 `
  --study-name "테스트-xgb-optuna-20"
```

Linux/macOS/WSL (Bash):

```bash
uv run python -m fdshield_ml.training.tune \
  --env-file .env.tracking \
  --data-path data/open/train.csv \
  --model-type xgboost \
  --n-trials 20 \
  --study-name "테스트-xgb-optuna-20"
```

최대 실행 시간을 제한하려면 초 단위 `--timeout`을 함께 지정합니다. 아래는 최대
1시간 또는 30회 Trial 중 먼저 도달한 조건에서 탐색을 끝냅니다.

PowerShell:

```powershell
uv run python -m fdshield_ml.training.tune `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type random-forest `
  --n-trials 30 `
  --timeout 3600 `
  --study-name "테스트-rf-optuna-1h"
```

Linux/macOS/WSL (Bash):

```bash
uv run python -m fdshield_ml.training.tune \
  --env-file .env.tracking \
  --data-path data/open/train.csv \
  --model-type random-forest \
  --n-trials 30 \
  --timeout 3600 \
  --study-name "테스트-rf-optuna-1h"
```

### Optuna 실행 옵션

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `--model-type` | `xgboost` | 자동 탐색할 모델 하나 선택 |
| `--n-trials` | `20` | 시도할 최대 파라미터 조합 수 |
| `--timeout` | 제한 없음 | 전체 탐색 최대 시간(초) |
| `--study-name` | `optuna-<모델>` | MLflow 부모 Run과 Optuna Study 제목 |
| `--comparison-group` | 미지정 | 같은 조건의 수동 학습 Run과 Study를 한 화면에서 묶을 공통 이름 |
| `--max-rows` | 전체 | Smoke Test용 표본 행 수 |
| `--n-jobs` | 최대 4 | Trial 안에서 모델이 사용할 CPU 작업 수 |
| `--registered-model-name` | 미지정 | Best 모델을 Registry에 등록할 때만 지정 |

일반 자동 탐색에서는 Registry 옵션을 생략합니다. 팀이 결과를 검토한 후 배포 후보를
등록하기로 정했을 때만 아래 옵션을 기존 명령 끝에 추가합니다.

```text
--registered-model-name fdshield-fraud-detector
```

### MLflow에서 Optuna 결과 읽기

1. `fdshield-model-comparison` Experiment에서 `run_kind=optuna_study`인 부모 Run을 찾습니다.
2. 부모 Run의 `validation_pr_auc`와 `best_*` 파라미터를 확인합니다.
3. 자식 Trial Run에서는 각 조합의 지표와 `training_seconds`를 비교합니다.
4. `tuning/study_summary.json`에는 전체 Trial, Best Trial, 파라미터가 저장됩니다.
5. 부모 Run의 `model`에는 Best 파라미터로 다시 학습한 모델만 저장됩니다.

Optuna는 가장 높은 검증 PR-AUC 조합을 자동으로 선택하지만, 그 모델이 무조건 운영에
가장 적합하다는 뜻은 아닙니다. Recall, Precision, False Positive Rate, 학습 시간도
확인하고, 최종 후보는 별도의 Test Set과 시연 시나리오로 검증합니다.

## MLflow에서 결과 확인하기

1. `https://mlflow.fdshield.cloud`에 접속합니다.
2. `fdshield-model-comparison` Experiment를 선택합니다.
3. 비교할 Run을 체크하고 Compare를 선택합니다.
4. 같은 검증 데이터 기준으로 PR-AUC, Recall, Precision, F1을 비교합니다.
5. 성능뿐 아니라 False Positive Rate와 Confusion Matrix도 함께 확인합니다.

### 팀 모델 비교 대시보드 만들기

팀원이 같은 데이터, `--test-size`, `--random-state`로 실험할 때는 동일한
`--comparison-group`을 지정합니다. 예시는 다음과 같습니다.

```text
--comparison-group fdshield-open-v1-full-model-comparison
```

MLflow의 Training runs 검색창에는 다음 필터를 입력합니다.

```text
tags.comparison_group = "fdshield-open-v1-full-model-comparison"
```

그다음 `validation_pr_auc`를 내림차순으로 정렬하고 Chart 화면의 검색창에
`validation_pr_auc`를 입력하면 동일 조건의 모델 성능을 막대 차트로 비교할 수 있습니다.
`owner` 태그에는 `.env.tracking`의 사용자 이름이, `run_kind`에는 수동 학습인지 Optuna
Study인지가 자동 기록되므로 누가 어떤 방식으로 실행했는지도 함께 확인할 수 있습니다.

### 검증 성능 지표 읽는 법

지표 이름의 `validation_` 접두사는 학습에 사용하지 않은 검증 분할에서 계산했다는
뜻입니다. 현재 검증 데이터는 여러 모델과 파라미터를 비교하는 데 반복 사용하므로,
이 결과를 최종 Test Set 성능으로 해석하면 안 됩니다.

먼저 분류 결과의 네 가지 경우를 알아두면 각 지표를 이해하기 쉽습니다.

| 구분 | 의미 |
|---|---|
| TP (True Positive) | 실제 사기를 사기로 올바르게 탐지 |
| FN (False Negative) | 실제 사기를 정상으로 잘못 판단하여 놓침 |
| FP (False Positive) | 정상 거래를 사기로 잘못 경고 |
| TN (True Negative) | 정상 거래를 정상으로 올바르게 판단 |

| MLflow 지표 | 의미 | 해석 방법 |
|---|---|---|
| `validation_pr_auc` | 모든 임곗값에서 Precision과 Recall의 관계를 종합한 면적 | 불균형 사기 탐지의 대표 비교 지표. 높을수록 좋음 |
| `validation_roc_auc` | 사기 거래가 정상 거래보다 높은 위험 점수를 받을 가능성을 종합 | 높을수록 좋지만 정상 거래가 매우 많으면 과하게 좋아 보일 수 있음 |
| `validation_precision` | `TP / (TP + FP)`, 사기 경고 중 실제 사기의 비율 | 낮으면 담당자가 확인할 허위 경고가 많아짐 |
| `validation_recall` | `TP / (TP + FN)`, 전체 실제 사기 중 탐지한 비율 | 낮으면 실제 사기를 많이 놓침 |
| `validation_f1` | Precision과 Recall의 조화평균 | 두 지표가 모두 높아야 높아지는 균형 점수 |
| `validation_false_positive_rate` | `FP / (FP + TN)`, 정상 거래 중 사기로 잘못 경고한 비율 | 낮을수록 좋으며 실제 FP 건수도 함께 확인 |
| `validation_accuracy` | `(TP + TN) / 전체`, 모든 거래에서 정답을 맞힌 비율 | 정상 거래가 압도적으로 많은 데이터에서는 참고용으로만 사용 |

예를 들어 다음과 같은 결과가 있다고 가정합니다.

```text
PR-AUC   = 0.3028
ROC-AUC  = 0.9525
Precision = 0.3503
Recall    = 0.2743
FPR       = 0.00485
```

- Precision `0.3503`: 사기 경고 100건 중 실제 사기는 약 35건입니다.
- Recall `0.2743`: 실제 사기 100건 중 약 27건만 잡고 약 73건을 놓칩니다.
- FPR `0.00485`: 정상 거래 10,000건 중 약 49건을 사기로 잘못 경고합니다.
- ROC-AUC가 `0.9525`로 높아도 PR-AUC와 Recall이 낮으므로 좋은 사기 탐지 모델이라고
  단정할 수 없습니다.
- PR-AUC `0.3028`은 정확도가 30.28%라는 뜻이 아니라 Precision-Recall 곡선 아래의
  면적입니다.

사기 탐지에서는 지표 하나만으로 최종 모델을 고르면 안 됩니다. 팀 모델 비교 시에는
다음 순서로 확인하는 것을 권장합니다.

1. `PR-AUC`로 불균형 데이터에서의 전반적인 탐지 성능을 비교합니다.
2. `Recall`로 실제 사기를 얼마나 놓치는지 확인합니다.
3. `Precision`으로 경고 중 실제 사기의 비율을 확인합니다.
4. `F1`으로 Precision과 Recall의 균형을 확인합니다.
5. `False Positive Rate`와 FP 건수로 정상 거래 오탐 부담을 확인합니다.
6. `ROC-AUC`와 `Accuracy`는 보조 지표로 확인합니다.

MLflow 화면에 지표 옆으로 표시되는 `model` 링크는 성능 점수가 아닙니다. 해당 Run이
저장한 학습 모델, 입력 스키마, 라이브러리 정보와 모델 파일을 확인하는 링크입니다.

## Model Registry 사용 기준

대부분의 팀원은 `--registered-model-name` 없이 실험합니다. 팀에서 결과를 비교한 뒤
실제 배포 후보로 검증할 모델만 Registry에 등록합니다.

```text
일반 학습 결과 -> MLflow Experiment의 Run
성능이 좋은 후보 -> Registry의 candidate
최종 승인 모델 -> Registry의 champion
```

후보 모델을 등록할 때만 다음 옵션을 추가합니다.

PowerShell:

```powershell
uv run python -m fdshield_ml.training.train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type xgboost `
  --run-name "candidate-01" `
  --registered-model-name fdshield-fraud-detector
```

Linux/macOS/WSL (Bash):

```bash
uv run python -m fdshield_ml.training.train \
  --env-file .env.tracking \
  --data-path data/open/train.csv \
  --model-type xgboost \
  --run-name "candidate-01" \
  --registered-model-name fdshield-fraud-detector
```

## 코드 위치와 테스트

| 경로 | 역할 |
|---|---|
| `fdshield_ml/common/feature_contract.py` | 학습·추론이 공유하는 원본 입력 컬럼 계약 |
| `fdshield_ml/common/features.py` | 라벨 변환, 식별자 제외, 시간 Feature 생성 |
| `fdshield_ml/serving/main.py` | 로컬 Docker와 Cloud Run Service 실행 진입점 |
| `fdshield_ml/serving/app.py` | Backend가 호출하는 FastAPI 추론 API |
| `fdshield_ml/serving/predictor.py` | 실제 모델 교체 전 규칙 기반 Stub 예측기 |
| `fdshield_ml/training/job.py` | Cloud Run Training Job 실행 진입점 |
| `fdshield_ml/training/tracking.py` | MLflow 주소와 인증 설정 및 연결 확인 |
| `fdshield_ml/training/pipeline.py` | 공통 데이터 분할, 전처리, 모델 생성 및 평가 |
| `fdshield_ml/training/train.py` | CLI 입력, 로컬 학습, MLflow 기록 실행 |
| `fdshield_ml/training/tuning.py` | 모델별 Optuna 탐색 범위와 Best 설정 복원 |
| `fdshield_ml/training/tune.py` | Optuna Study 실행, Trial 및 Best 모델 MLflow 기록 |
| `tests/test_training.py` | 가짜 데이터로 전체 학습 흐름 검증 |
| `tests/test_tuning.py` | 모델별 탐색 범위와 Best 설정 복원 검증 |
| `tests/test_serving.py` | 상태 확인, 요청 검증, Stub 추론 계약 검증 |

코드를 수정했다면 테스트를 실행합니다.

Windows/Linux 공통:

```console
uv run python -m pytest
```

이 레포는 백엔드처럼 프로젝트 자체를 별도 패키지로 빌드하지 않고, 레포 루트의
`fdshield_ml` 모듈을 직접 실행합니다. 따라서 `uv sync` 중 프로젝트 빌드용 임시
Python을 생성하지 않습니다.

공통 데이터 분할과 평가 방식을 임의로 변경하면 다른 팀원의 Run과 공정하게 비교할
수 없으므로, 변경이 필요한 경우 팀에서 먼저 기준을 합의합니다.

## ML 서빙 스켈레톤 실행

실제 모델 파일을 연결하기 전에는 현재 공개 데이터에서 모델이 사용하는 원본 55개
컬럼을 받는 결정적 규칙 기반 Stub이 실행됩니다. 요청은 `transaction_id`와
`features`로 구성하며, `features`의 이름과 값은 학습 Pipeline에 들어가는 원본 형태를
유지합니다. 정답 라벨 `Fraud_Type`과 현재 모델에서 제외한 식별정보는 받지 않습니다.
같은 요청은 항상 같은 결과를 반환하므로 Backend 연동과 시연 흐름을 먼저 검증할 수
있습니다.

```json
{
  "transaction_id": "TEST_000001",
  "features": {
    "Customer_Birthyear": 1960,
    "Customer_Gender": "female",
    "Transaction_Datetime": "2003-01-11 20:29:36",
    "Transaction_Amount": -9450000,
    "Channel": "mobile"
  }
}
```

위 예시는 구조를 줄여 쓴 것이며 실제 요청의 `features`에는
`fdshield_ml/common/feature_contract.py`에 정의한 55개 컬럼이 모두 필요합니다.
현재 Stub 응답의 `shap`은 실제 SHAP가 아니라 원본 컬럼 이름을 사용한 임시
기여도입니다.

```console
uv run uvicorn fdshield_ml.serving.app:app --host 0.0.0.0 --port 8001
```

실행 후 API 문서는 `http://localhost:8001/docs`에서 확인합니다. 상태 확인은
`GET /health`, 준비 상태는 `GET /ready`, 추론은 `POST /predict`를 사용합니다.

서빙 설정은 다음 환경변수로 변경할 수 있습니다.

| 환경변수 | 기본값 | 역할 |
|---|---|---|
| `ML_FRAUD_THRESHOLD` | `0.55` | 사기로 판정할 확률 임계값 |
| `ML_MODEL_NAME` | `fdshield-rule-based-stub` | 응답에 표시할 모델 이름 |
| `ML_MODEL_VERSION` | `0` | 응답에 표시할 Stub 버전 |

실제 모델을 연결할 때는 MLflow에 저장된 전처리 포함 Pipeline을 로드하고,
`StubPredictor`만 실제 예측기 구현으로 교체합니다.

Docker로 Stub 서버를 실행할 때는 다음 명령을 사용합니다.

```console
docker compose -f compose.serving.yml up --build
```

컨테이너는 `http://localhost:8001`에서 요청을 받고, Cloud Run에서도 같은 이미지를
사용할 수 있도록 내부 포트는 `PORT=8080`으로 실행됩니다. 현재 Docker 실행 경로는
MLflow와 연결하지 않으며 항상 `StubPredictor`를 사용합니다.

## Cloud Run Training Job 스켈레톤

실제 데이터 다운로드와 모델 학습을 연결하기 전에는 Training Job 컨테이너의 실행,
환경변수 전달, 로그 수집, 종료 코드만 확인하는 Stub을 사용합니다. 로컬 설정 파일은
`.env.training.example`을 복사해 준비합니다.

```console
Copy-Item .env.training.example .env.training
```

학습용 이미지를 빌드하고 실행합니다.

```console
docker build -f Dockerfile.training -t fdshield/ml-training:local .
docker run --rm --env-file .env.training fdshield/ml-training:local
```

정상 실행되면 `training_job_started`, `training_job_completed` JSON 로그를 출력하고
종료 코드 `0`을 반환합니다. 필수 설정이 없거나 지원하지 않는 값이면
`training_job_configuration_error` 로그와 종료 코드 `2`를 반환합니다.

현재 지원하는 설정은 다음과 같습니다.

| 환경변수 | 현재 값 | 역할 |
|---|---|---|
| `TRAINING_JOB_TYPE` | `binary` | 실행할 학습 작업 종류 |
| `TRAINING_DATA_URI` | `stub://local-data` | Stub 데이터 위치, 추후 `gs://` 사용 |
| `MLFLOW_EXPERIMENT_NAME` | `fdshield-binary-training` | 추후 학습 결과를 기록할 MLflow Experiment |

현재 Stub은 MLflow에 접속하거나 실제 모델을 학습하지 않습니다. Cloud Run Job 실행
구조를 검증한 뒤 GCS 데이터 다운로드와 기존 `training.train` 학습 흐름을 연결합니다.

### Training 이미지 Cloud Build

`cloudbuild.training.yaml`은 Serving용 `Dockerfile.serving`이 아니라
`Dockerfile.training`을 사용해 Training 이미지만 빌드하고 Artifact Registry에
업로드합니다. 이미지 태그는 실행 시 `_DEPLOY_IMAGE` substitution으로 명시합니다.

```console
DEPLOY_IMAGE="asia-northeast3-docker.pkg.dev/project-4cc3406c-72d8-4907-a5d/fdshield/ml-training:manual"

gcloud builds submit \
  --project="project-4cc3406c-72d8-4907-a5d" \
  --config="cloudbuild.training.yaml" \
  --substitutions="_DEPLOY_IMAGE=${DEPLOY_IMAGE}" \
  .
```

이 설정은 이미지를 빌드하고 업로드하는 역할만 합니다. Cloud Run Job 생성과 학습
실행은 별도 단계이며, 이미지 업로드만으로 학습이 자동 시작되지는 않습니다.

### Serving·Training 이미지 변경 범위

Python 테스트는 모든 Pull Request에서 실행합니다. Docker 이미지 빌드는 변경된
파일이 실제 이미지에 포함되는 경우에만 실행합니다.

Serving 배포에서는 `cloudbuild.serving.yaml`이 `Dockerfile.serving`을 지정하고,
Training 이미지 빌드에서는 `cloudbuild.training.yaml`이
`Dockerfile.training`을 지정합니다.

| 변경 영역 | Serving 이미지 CI | Training 이미지 CI |
|---|---:|---:|
| `fdshield_ml/serving/**`, Serving Dockerfile·의존성 | 실행 | 생략 |
| `fdshield_ml/training/**`, Training Dockerfile·의존성 | 생략 | 실행 |
| `fdshield_ml/common/**`, 호환 모듈, `.dockerignore` | 실행 | 실행 |
| 테스트·문서만 변경 | 생략 | 생략 |

`Dockerfile.serving`은 `common/`과 `serving/`을 복사하고,
`Dockerfile.training`은
`common/`과 `training/`을 복사합니다. 따라서 한 영역의 코드 변경이 다른 이미지의
불필요한 재빌드로 이어지지 않습니다. Serving Cloud Run CD도 같은 경로 조건을
사용합니다.
