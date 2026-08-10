# fdshield-fraud-detector v5

팀원이 MLflow 계정이나 원격 VM 없이 실제 모델로 로컬 연동을 확인할 수 있도록 Git에
고정한 XGBoost native 모델이다.

- 등록 모델: `fdshield-fraud-detector`
- Registry 버전: `5`
- 입력 계약: 원본 54개 → 공통 전처리 → 모델 Feature 91개
- 판정 임계값: `0.55`
- 모델 형식: XGBoost UBJ

원본 MLflow pickle은 Linux Serving 이미지 안에서 로드한 뒤 XGBoost native UBJ로
내보냈다. 원본과 native 모델의 MLflow input example 5행 예측확률 최대 절대 오차는
`0.0`이다. 시작할 때 `manifest.json`의 SHA-256과 91개 Feature 이름·순서를 검증하므로
파일이 바뀌거나 입력 계약이 어긋나면 Serving이 준비 상태가 되지 않는다.

이 파일은 신뢰된 프로젝트 저장소의 모델이다. 출처를 알 수 없는 외부 모델 파일로
교체하지 않는다.
