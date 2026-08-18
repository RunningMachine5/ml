"""
[데이터 조립 코드]
추론 결과의 model79 SHAP 값을 API에서 보기 쉬운 원본 피처 단위로 묶는다.
"""

from fdshield_ml.config.preprocess_config import CATEGORICAL_LEVELS

OUTPUT_NAMES = {
    "seconds_since_last_transaction": "time_difference",
    "distance_since_last_transaction": "distance",
}


def shap_decode(shap: dict[str, float]) -> dict[str, float]:
    """원-핫 인코딩된 피처를 원래 범주 이름으로 합친다."""

    decoded: dict[str, float] = {}
    for feature, value in shap.items():
        # customer_gender_male/female처럼 나뉜 값을 customer_gender로 합친다.
        category = next(
            (
                name
                for name in CATEGORICAL_LEVELS
                if feature.startswith(f"{name}_")
            ),
            None,
        )
        output_name = category or OUTPUT_NAMES.get(feature, feature)
        decoded[output_name] = round(decoded.get(output_name, 0.0) + value, 3)

    return decoded
