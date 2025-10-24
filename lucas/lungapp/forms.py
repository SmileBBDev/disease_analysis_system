# lungapp/forms.py
from django import forms

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
        if col in cat_cols:
            choices = [(v, v) for v in cat_choices.get(col, [])]
            if choices:
                field = forms.ChoiceField(
                    label=col, choices=choices, required=True,
                    widget=forms.Select(attrs={"class": "form-select"})
                )
            else:
                field = forms.CharField(
                    label=col, required=True,
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
                label=col, required=True,
                widget=forms.NumberInput(attrs=attrs)
            )

        setattr(_PredictForm, col, field)

    return _PredictForm
