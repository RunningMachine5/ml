# FDShield 모델 실험 입문 가이드

이 문서는 머신러닝과 MLflow를 처음 사용하는 팀원이 FDShield 모델을 같은 조건으로
학습하고 결과를 비교할 수 있도록 만든 안내서입니다. 코드를 먼저 수정하기보다 이
문서의 명령을 그대로 실행해 전체 흐름을 확인하세요.

## 1. 우리가 하는 일

FDShield의 학습 데이터에는 정상 거래와 여러 종류의 사기 거래가 들어 있습니다.
학습 코드가 `Fraud_Type=m`을 정상 `0`, `a~l`을 사기 `1`로 합쳐서 **이진분류**
문제로 바꿉니다.

```text
같은 train.csv
      |
      +-- Logistic Regression
      +-- Decision Tree
      +-- Random Forest
      `-- XGBoost
              |
              v
     같은 분할·Feature·평가지표
              |
              v
        공용 MLflow에서 비교
```

모델 학습은 팀원 PC에서 실행됩니다. MLflow 서버는 모델을 대신 학습하지 않고,
각 실행의 파라미터·평가지표·모델 파일을 한곳에 모아 비교할 수 있게 해줍니다.
MLflow에서 학습 한 번의 기록을 **Run**, 여러 Run을 모은 공간을
**Experiment**라고 부릅니다.

## 2. 네 모델을 넣은 이유

| 모델 | 역할 | 처음 볼 때 기억할 점 |
|---|---|---|
| Logistic Regression | 단순한 선형 기준 모델 | 복잡한 모델이 정말 필요한지 확인하는 출발점 |
| Decision Tree | 한 개의 의사결정나무 | 이해하기 쉽지만 깊어지면 과적합하기 쉬움 |
| Random Forest | 여러 트리의 결과를 평균 | 단일 트리보다 안정적이지만 모델이 커질 수 있음 |
| XGBoost | 이전 트리의 오차를 다음 트리가 보완 | 성능이 좋은 경우가 많지만 조정할 값도 많음 |

모델이 복잡하다고 항상 더 좋은 것은 아닙니다. 같은 데이터와 같은 평가 방법으로
비교했을 때 복잡한 모델의 장점이 실제 지표로 확인되어야 합니다.

## 3. 처음 한 번만 준비하기

레포 루트에서 다음 명령을 실행합니다.

```powershell
uv sync --dev
Copy-Item .env.tracking.example .env.tracking
```

`.env.tracking`에 본인에게 발급된 MLflow 계정을 입력합니다.

```env
MLFLOW_TRACKING_URI=https://mlflow.fdshield.cloud
MLFLOW_TRACKING_USERNAME=<본인 계정>
MLFLOW_TRACKING_PASSWORD=<본인 비밀번호>
MLFLOW_EXPERIMENT_NAME=fdshield-model-comparison
```

공용 서버 연결과 권한을 먼저 확인합니다.

```powershell
uv run fdshield-mlflow-check --env-file .env.tracking
```

`data/open/train.csv`가 없다면 공유받은 동일한 파일을 복사합니다. CSV는 Git에
올리지 않습니다.

```powershell
Copy-Item '<공유받은 경로>\train.csv' data\open\train.csv
```

## 4. 팀 전체가 반드시 고정할 조건

모델만 공정하게 비교하려면 아래 조건을 팀원마다 바꾸면 안 됩니다.

- 같은 `train.csv`
- 같은 Git 커밋의 학습 코드
- `--test-size 0.2`
- `--random-state 42`
- 같은 `MLFLOW_EXPERIMENT_NAME`
- 같은 Feature 처리와 평가 함수
- 최종 비교에서는 `--max-rows`를 사용하지 않음

`--max-rows`는 전체 학습 전에 코드와 MLflow 연결을 빠르게 확인하는 Smoke
Test용입니다. 일부 행으로 돌린 결과와 전체 데이터 결과를 서로 비교하면 안 됩니다.

## 5. 가장 먼저 Smoke Test 실행하기

본인이 맡은 모델 하나를 골라 2만 행으로 전체 흐름을 확인합니다.

```powershell
uv run fdshield-train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type xgboost `
  --max-rows 20000 `
  --n-estimators 50 `
  --run-name "<이름>-xgb-smoke"
```

성공하면 콘솔에 MLflow Run 주소와 PR-AUC, Recall, Precision, F1이 출력됩니다.
MLflow 웹에서도 같은 Run이 보이는지 확인합니다.

## 6. 모델별 전체 실행 예시

### Logistic Regression

```powershell
uv run fdshield-train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type logistic-regression `
  --logistic-c 1.0 `
  --run-name "<이름>-logistic-001"
```

`logistic-c`가 작을수록 모델의 계수를 더 강하게 제한합니다. 처음에는
`0.1`, `1.0`, `10.0` 정도만 비교하면 충분합니다.

### Decision Tree

```powershell
uv run fdshield-train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type decision-tree `
  --max-depth 5 `
  --min-samples-leaf 5 `
  --run-name "<이름>-tree-001"
```

`max-depth`가 커질수록 더 복잡한 규칙을 만들고 과적합하기 쉽습니다.
`min-samples-leaf`를 키우면 너무 적은 거래만 설명하는 잎을 줄일 수 있습니다.

### Random Forest

```powershell
uv run fdshield-train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type random-forest `
  --n-estimators 300 `
  --max-depth 10 `
  --min-samples-leaf 5 `
  --max-features sqrt `
  --run-name "<이름>-rf-001"
```

`n-estimators`는 트리 개수입니다. 늘리면 대체로 결과가 안정적이지만 학습 시간과
모델 크기도 증가합니다.

### XGBoost

```powershell
uv run fdshield-train `
  --env-file .env.tracking `
  --data-path data/open/train.csv `
  --model-type xgboost `
  --n-estimators 300 `
  --max-depth 5 `
  --learning-rate 0.05 `
  --subsample 0.8 `
  --colsample-bytree 0.8 `
  --min-child-weight 1 `
  --run-name "<이름>-xgb-001"
```

`learning-rate`를 낮추면 보통 더 많은 트리가 필요합니다. 한 번에 모든 값을 크게
바꾸기보다 기준 Run에서 한두 값씩 변경하면 결과를 설명하기 쉽습니다.

## 7. 파라미터와 코드 중 무엇을 바꾸나

| 하고 싶은 일 | 방법 |
|---|---|
| 네 모델 중 하나 선택 | `--model-type` 변경 |
| 모델 내부 설정 비교 | 해당 CLI 파라미터 변경 |
| Run을 본인 결과로 구분 | `--run-name` 변경 |
| Feature 추가·삭제 | 팀 회의 후 `features.py` 변경 |
| 데이터 분할 방식 변경 | 개인 실험에서 변경하지 말고 팀 합의 필요 |
| 새로운 모델 종류 추가 | `training.py`의 모델 생성 코드와 테스트 변경 |

팀원이 기존 네 모델을 실험할 때는 Python 코드를 직접 수정할 필요가 없습니다.
코드를 각자 수정하면 전처리나 분할 조건까지 달라져 MLflow 숫자를 공정하게 비교할
수 없게 됩니다.

## 8. MLflow에서 볼 항목

한 Run에는 다음 내용이 기록됩니다.

- 모델 종류와 그 모델이 실제로 사용한 하이퍼파라미터
- 학습/검증 행 수와 사기 거래 수
- 학습 시간
- PR-AUC, ROC-AUC, Precision, Recall, F1, False Positive Rate
- Confusion Matrix와 분류 리포트
- 전처리와 분류기를 함께 저장한 모델 artifact
- 직접 식별자가 제거된 모델 입력 예시 5행과 입력 스키마

원본 CSV 전체, 계좌번호, 개인식별번호, IP, MAC 등 직접 식별자는 MLflow에
업로드하지 않습니다. 모델 artifact에는 실제 서빙 입력 형식을 보여주기 위해 직접
식별자를 제외한 입력 예시 5행이 포함됩니다.

## 9. 어떤 지표로 고를까

정상 거래가 훨씬 많은 데이터에서는 모든 거래를 정상이라고 예측해도 Accuracy가
높게 나올 수 있습니다. 따라서 Accuracy만 보고 모델을 고르면 안 됩니다.

1. `validation_pr_auc`: 불균형 이진분류의 전체적인 순위 성능을 보는 1차 기준
2. `validation_recall`: 실제 사기 중 잡아낸 비율
3. `validation_false_positive_rate`: 정상 거래를 사기로 잘못 막은 비율
4. `validation_precision`: 사기라고 알린 거래 중 실제 사기의 비율

현재 예제의 F1, Precision, Recall은 검증 데이터에서 F1이 가장 높은 판정 임계값을
찾은 뒤 같은 검증 데이터로 계산한 값이므로 실험 중 참고용입니다. 모델 후보를
고를 때는 우선 PR-AUC를 기준으로 비교하고, 최종 성능을 발표하기 전에는 별도로
보관한 Test 데이터에서 한 번 더 평가하는 것이 안전합니다.

## 10. 팀 실험 규칙 추천

- Run 이름: `<담당자>-<모델>-<번호>` 형식 사용
- Smoke Test Run에는 반드시 `smoke` 표시
- 한 번 실행한 Run을 지우기보다 실패 이유를 메모로 남김
- 일반 Run은 Registry에 등록하지 않음
- 성능이 좋은 2~3개 후보만 `candidate`로 등록
- 최종 검증과 팀 승인을 거친 모델 하나를 `champion`으로 지정

모델 선택 이유는 “점수가 가장 높아서”로 끝내지 말고 PR-AUC, Recall,
False Positive Rate, 학습 시간과 모델 크기를 함께 설명합니다.
