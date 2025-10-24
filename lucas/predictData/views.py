# train (데이터 훈련 views.py)
from django.shortcuts import render
from django.contrib import messages
import pandas as pd
import numpy as np
import logging
from lungapp.utils import load_model, load_feature_schema, _build_manual_fields_from_schema
from lungapp.forms import make_predict_form

log = logging.getLogger(__name__)

def data_predict(request):
    context = {}
    log.debug("▶ predictData index 진입 | method=%s action=%s", request.method, request.POST.get("action"))


    # 모델 로드 (GET이든 POST든 form을 보여주려면 필요)
    store = load_model()
    if not store:
        messages.warning(request, "먼저 '학습'을 수행해 모델을 저장하세요.")
        return render(request, "predictData/predict.html", context)

    # 모델 스키마 로드
    model_schema = load_feature_schema()
    model_feature_cols = model_schema.get("feature_cols", [])
    model_num_cols = set(model_schema.get("num_cols", []))

    # form과 수동필드 준비 (항상 보여지게)
    PredictForm = make_predict_form(model_schema)
    context["manual_fields"] = _build_manual_fields_from_schema(model_schema)



    if request.method == "POST" and request.POST.get("action") == "predict":
        store = load_model()
        if not store:
            messages.error(request, "먼저 '학습'을 수행해 모델을 저장하세요.")
        else:
            model_schema = load_feature_schema()
            model_feature_cols = model_schema.get("feature_cols", [])
            model_num_cols = set(model_schema.get("num_cols", []))

            PredictForm = make_predict_form(model_schema)
            form = PredictForm(request.POST)
            context["form"] = form
            context["manual_fields"] = _build_manual_fields_from_schema(model_schema)

            try:
                input_data = {}
                missing = []
                if form.is_valid() and form.fields:
                    for col in model_feature_cols:
                        if col in form.cleaned_data:
                            v = form.cleaned_data[col]
                            input_data[col] = [float(v)] if col in model_num_cols else [str(v)]
                        else:
                            missing.append(col)
                    x_df = pd.DataFrame(input_data, columns=model_feature_cols)
                else:
                    for col in model_feature_cols:
                        v = request.POST.get(col, None)
                        if v is None or v == "":
                            missing.append(col)
                            input_data[col] = [np.nan]
                        else:
                            input_data[col] = [float(v)] if col in model_num_cols else [str(v)]
                    if missing:
                        messages.error(request, f"다음 입력이 비어 있습니다: {', '.join(missing)}")
                        return render(request, "predictData/index.html", context)
                    x_df = pd.DataFrame(input_data, columns=model_feature_cols)

                pipe = store["pipeline"]
                class_names = store["class_names"]
                pred = pipe.predict(x_df)[0]
                proba = pipe.predict_proba(x_df)[0]
                prob_table = [(class_names[i], round(float(p), 4)) for i, p in enumerate(proba)]

                context["pred_label"] = str(pred)
                context["pred_table"] = prob_table
                messages.success(request, f"예측 결과: {pred}")
            except Exception as e:
                messages.error(request, f"예측 실패: {e}")
                log.exception("predict 실패: %s", e)

    return render(request, "predictData/predict.html", context)