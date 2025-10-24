from django.shortcuts import render
from django.contrib import messages
import pandas as pd
import logging
from lungapp.utils import train_and_save_model, load_feature_schema, _parse_int, _build_manual_fields_from_schema
from lungapp.forms import make_predict_form

log = logging.getLogger(__name__)

def data_study(request):
    context = {}
    log.debug("▶ studyData index 진입 | method=%s action=%s", request.method, request.POST.get("action"))

    if request.method == "POST" and request.POST.get("action") == "train":
        try:
            n_per_class = _parse_int(request.POST.get("n_per_class", ""), 1000)
            result = train_and_save_model(n_per_class=n_per_class)

            metrics_df = pd.DataFrame([result["metrics"]]).T.reset_index()
            metrics_df.columns = ["Metric", "Value"]
            context["metrics_html"] = metrics_df.to_html(classes="table table-bordered table-sm", index=False)
            context["roc_url"] = result["roc_url"]

            model_schema = load_feature_schema()
            context["form"] = make_predict_form(model_schema)()
            context["manual_fields"] = _build_manual_fields_from_schema(model_schema)

            messages.success(request, f"학습 완료 (클래스당 {n_per_class}개). 입력 폼이 모델 스키마로 갱신되었습니다.")
        except Exception as e:
            messages.error(request, f"학습 실패: {e}")
            log.exception("train 실패: %s", e)

    return render(request, "studyData/studyData.html", context)