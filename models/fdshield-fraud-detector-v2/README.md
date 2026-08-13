# fdshield-fraud-detector v2 local candidate

ML 담당자가 전달한 `xgb_model_weights_12_24_24.json`을 원본 그대로 복사한
80-Feature XGBoost JSON 모델이다. `train1.csv` 기반 새 계약의 로컬 이식과
Backend 연동을 확인하는 초기 candidate로만 사용한다.

- 원본 입력 계약: snake_case 60개 (`raw60-model80-v1`)
- 모델 입력 계약: 순서가 고정된 숫자 Feature 80개
- 판정 임계값: `0.5`
- 학습 데이터: `train1-v1` 200,000건
- 모델 형식: XGBoost JSON

전달 학습 코드에는 입금 플래그 컬럼 오타와 Channel/OS 대소문자 불일치가
있었다. 이 모델은 그 상태에서 생성된 이식 기준 모델이므로, 공용 전처리를
바로잡은 뒤에는 `train1-v1`으로 재학습하고 MLflow 수동 승인 절차를 거쳐
교체해야 한다.
