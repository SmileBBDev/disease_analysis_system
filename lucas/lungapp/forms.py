# lungapp/forms.py
from django import forms

# 컬럼 이름 → 한글 라벨 매핑
LABEL_MAP = {
    "age": "나이",
    "gender": "성별",
    "pack_years": "흡연량 (Pack-years)",
    "radon_exposure": "라돈 노출 정도",
    "asbestos_exposure": "석면 노출 여부",
    "secondhand_smoke_exposure": "간접 흡연 노출 여부",
    "copd_diagnosis": "COPD 진단 여부",
    "alcohol_consumption": "음주량",
    "family_history": "가족력 여부",
}

def make_predict_form(schema: dict):
    """
    schema = {
      'feature_cols': [...],
      'num_cols': [...],
      'cat_cols': [...],
      'cat_choices': {col: [v1, v2, ...]},
      'num_stats': {col: {'min': ..., 'max': ...}}
    }
    """
    feature_cols = schema.get("feature_cols", [])
    num_cols     = set(schema.get("num_cols", []))
    cat_cols     = set(schema.get("cat_cols", []))
    cat_choices  = schema.get("cat_choices", {})
    num_stats    = schema.get("num_stats", {})

    class _PredictForm(forms.Form):
        pass

    for col in feature_cols:
        key = col.strip().lower()  # 공백 제거, 소문자 처리
        label = LABEL_MAP.get(key, col)  # ← 한글 라벨 적용
        print("labellable : ", label)
        if col in cat_cols:
            choices = [(v, v) for v in cat_choices.get(col, [])]
            if choices:
                field = forms.ChoiceField(
                    label=label, choices=choices, required=True,
                    widget=forms.Select(attrs={"class": "form-select"})
                )
            else:
                field = forms.CharField(
                    label=label, required=True,
                    widget=forms.TextInput(attrs={"class": "form-control"})
                )
        else:
            stats = num_stats.get(col, {})
            attrs = {"class": "form-control", "step": "any"}
            if stats.get("min") is not None:
                attrs["min"] = stats["min"]
            if stats.get("max") is not None:
                attrs["max"] = stats["max"]
            field = forms.FloatField(
                label=label, required=True,
                widget=forms.NumberInput(attrs=attrs)
            )

        setattr(_PredictForm, col, field)

    return _PredictForm