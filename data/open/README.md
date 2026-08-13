# 로컬 학습 데이터 위치

ML 담당자가 전달한 `train1.csv`를 이 디렉터리로 복사해 로컬 학습에
사용합니다. 저장 위치와 학습 Job 설정은 다음과 같습니다.

```env
TRAINING_DATA_URI=data/open/train1.csv
```

PowerShell:

```powershell
Copy-Item -LiteralPath '<전달 폴더>\app\datas\train1.csv' -Destination '.\data\open\train1.csv'
```

macOS/Linux:

```bash
cp '<전달 폴더>/app/datas/train1.csv' ./data/open/train1.csv
```

기준 파일은 200,000행, 64열이며 SHA-256은 다음과 같습니다.

```text
D025873C5E807976657B30721080D00BF6B6544B887FF339E768E8C13F54E446
```

이 파일의 raw64는 추론 raw60의 flag 이름을 CSV 별칭으로 교체하고
`customer_id`, `customer_identification_number`, `balance_drain_ratio`, `is_fraud`
4개를 더한 구조입니다. `transaction_id`와 학습 메타데이터·라벨을 제외한
실제 전처리 입력은 59개이고, 출력은 순서가 고정된 model80입니다.

raw64 계약은 `fdshield_ml/common/preprocess_config.py`, model80 변환은
`fdshield_ml/common/preprocessor.py`에서 추론과 공유합니다. 로컬 파일과 GCS
객체를 가져오는 방식은 `fdshield_ml/training/data_loader.py`에서 처리합니다.

`data/open/*`는 `.gitignore`로 제외됩니다. CSV와 식별성 데이터는 Git이나
Docker 이미지에 포함하지 않습니다. Cloud Run Training Job에서는 같은 파일을
비공개 GCS의 버전 고정 경로에 업로드하고 SHA-256을 확인한 뒤 그 `gs://` URI를
사용합니다.
