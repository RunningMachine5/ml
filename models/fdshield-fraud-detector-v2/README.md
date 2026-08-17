# fdshield-fraud-detector v2 local candidate

ML 담당자가 전달한 `train1.csv`를 현재 공용 전처리로 학습한
79-Feature XGBoost JSON 로컬 모델이다. Backend raw51 연동과 로컬 Serving을
확인하는 candidate로 사용한다.

- 원본 입력 계약: snake_case raw51 (`raw51-model79-v1`)
- 모델 입력 계약: 순서가 고정된 숫자 Feature 79개
- 판정 임계값: `0.5`
- 학습 데이터: `train1-v1` 200,000건
- 모델 형식: XGBoost JSON

운영 반영 시에는 같은 계약으로 Cloud Run Training Job을 실행해 MLflow에
후보를 등록하고 관리자 수동 승인 절차를 거쳐야 한다.
