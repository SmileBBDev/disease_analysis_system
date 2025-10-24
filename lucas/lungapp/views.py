# lungapp/views.py
import os
from django.shortcuts import render
from django.contrib import messages
from django.conf import settings
import numpy as np
import pandas as pd
import logging  # [LOG] 추가

from .utils import (
    get_dataset_df,
    train_and_save_model,
    load_model,
    load_feature_schema,       # 모델/CSV에서 스키마 얻기
    build_schema_from_dataframe,
)
from .forms import make_predict_form

# [LOG] 앱 전용 로거
log = logging.getLogger("app.flow")


# ───────────────────────────────────────────────
# 하드코딩 스키마 (모델이 없을 때만 임시 사용해서 폼을 띄우는 용도)
# ───────────────────────────────────────────────
HARDCODED_SCHEMA = {
    "feature_cols": [
        "Age", "Gender", "Smoking", "Yellow_Fingers", "Anxiety", "Peer_Pressure",
        "Chronic_Disease", "Fatigue", "Allergy", "Wheezing", "Alcohol_Consuming",
        "Coughing", "Shortness_of_Breath", "Swallowing_Difficulty", "Chest_Pain"
    ],
    "num_cols": ["Age"],
    "cat_cols": [
        "Gender", "Smoking", "Yellow_Fingers", "Anxiety", "Peer_Pressure",
        "Chronic_Disease", "Fatigue", "Allergy", "Wheezing", "Alcohol_Consuming",
        "Coughing", "Shortness_of_Breath", "Swallowing_Difficulty", "Chest_Pain"
    ],
    "cat_choices": {
        "Gender": ["M", "F"],
        "Smoking": ["YES", "NO"],
        "Yellow_Fingers": ["YES", "NO"],
        "Anxiety": ["YES", "NO"],
        "Peer_Pressure": ["YES", "NO"],
        "Chronic_Disease": ["YES", "NO"],
        "Fatigue": ["YES", "NO"],
        "Allergy": ["YES", "NO"],
        "Wheezing": ["YES", "NO"],
        "Alcohol_Consuming": ["YES", "NO"],
        "Coughing": ["YES", "NO"],
        "Shortness_of_Breath": ["YES", "NO"],
        "Swallowing_Difficulty": ["YES", "NO"],
        "Chest_Pain": ["YES", "NO"],
    },
    "num_stats": {"Age": {"min": 1, "max": 120}},
}


def _parse_int(value, default):
    try:
        v = int(value)
        return v if v > 0 else default
    except Exception:
        return default


def _build_manual_fields_from_schema(schema: dict):
    """
    템플릿에서 수동 렌더링(fallback)할 수 있도록 필드 메타 생성.
    (활성 스키마 기준)
    """
    fields = []
    feature_cols = schema.get("feature_cols", [])
    num_cols = set(schema.get("num_cols", []))
    cat_cols = set(schema.get("cat_cols", []))
    cat_choices = schema.get("cat_choices", {})
    num_stats = schema.get("num_stats", {})

    for col in feature_cols:
        if col in cat_cols:
            fields.append({
                "name": col, "label": col, "type": "select",
                "choices": cat_choices.get(col, []),
            })
        elif col in num_cols:
            stat = num_stats.get(col, {})
            fields.append({
                "name": col, "label": col, "type": "number",
                "min": stat.get("min"), "max": stat.get("max"),
            })
        else:
            fields.append({"name": col, "label": col, "type": "text"})
    return fields


def _pick_active_schema():
    """
    활성 스키마 결정 규칙:
      1) 모델이 존재하면 => 모델 스키마(= load_feature_schema) [예측과 100% 일치]
      2) 모델이 없으면   => CSV에서 생성된 스키마가 있으면 그걸 사용
      3) 둘 다 없으면   => 하드코딩 스키마
    """
    store = load_model()
    if store:
        return load_feature_schema(), True  # (schema, from_model=True)

    try:
        df = get_dataset_df()
        sc = build_schema_from_dataframe(df)
        if sc and sc.get("feature_cols"):
            return sc, False
    except Exception:
        pass

    return HARDCODED_SCHEMA, False


def index(request):
    context = {}
    log.debug("▶ index 진입 | method=%s action=%s", request.method, request.POST.get("action"))  # [LOG]

    # ── 활성 스키마 선택 (모델 스키마가 최우선) ──
    active_schema, from_model = _pick_active_schema()
    log.debug("스키마 선택 완료 | from_model=%s", from_model)  # [LOG]

    # 폼 / 수동필드 모두 활성 스키마로 생성 (항상 렌더되도록)
    PredictForm = make_predict_form(active_schema)
    form = PredictForm()
    manual_fields = _build_manual_fields_from_schema(active_schema)

    context["form"] = form
    context["manual_fields"] = manual_fields

    # 디버그 메시지: 지금 어떤 스키마인지/필드 몇 개인지
    try:
        field_names = list(form.fields.keys())
        src = "MODEL" if from_model else "CSV/HARDCODED"
        messages.info(request, f"[DEBUG] active schema: {src} | fields({len(field_names)}): {', '.join(field_names) if field_names else 'NONE'}")
        log.debug("폼 준비 | schema=%s | field_count=%d", src, len(field_names))  # [LOG]
    except Exception as e:
        messages.info(request, "[DEBUG] form has no .fields")
        log.warning("폼 필드 접근 실패: %s", e)  # [LOG]

    # ── 액션 처리 ──
    if request.method == "POST":
        action = request.POST.get("action")
        log.debug("POST 처리 시작 | action=%s", action)  # [LOG]

        if action == "load_data":
            try:
                log.debug("load_data 시작")  # [LOG]
                df = get_dataset_df()
                show = df.head(500).copy() if len(df) > 500 else df
                note = f"총 {len(df)}행 중 500행만 표시" if len(df) > 500 else f"총 {len(df)}행 표시"
                context["data_html"] = show.to_html(classes="table table-striped table-sm", index=False)
                context["data_note"] = note
                messages.success(request, f"데이터 로드 완료: {os.path.basename(settings.LUNG_CANCER_CSV)}")
                log.debug("load_data 완료 | rows=%d", len(df))  # [LOG]
            except Exception as e:
                messages.error(request, f"데이터 로드 실패: {e}")
                log.exception("load_data 실패: %s", e)  # [LOG]

        elif action == "train":
            try:
                log.debug("train 시작 | n_per_class(raw)=%s", request.POST.get("n_per_class", ""))  # [LOG]
                n_per_class = _parse_int(request.POST.get("n_per_class", ""), 1000)
                result = train_and_save_model(n_per_class=n_per_class)

                # 성능/ROC 표시
                metrics_df = pd.DataFrame([result["metrics"]]).T.reset_index()
                metrics_df.columns = ["Metric", "Value"]
                context["metrics_html"] = metrics_df.to_html(classes="table table-bordered table-sm", index=False)
                context["roc_url"] = result["roc_url"]

                # 학습 후에는 반드시 모델 스키마로 폼/수동필드 갱신
                model_schema = load_feature_schema()
                context["form"] = make_predict_form(model_schema)()
                context["manual_fields"] = _build_manual_fields_from_schema(model_schema)

                messages.success(request, f"학습 완료 (클래스당 {n_per_class}개). 입력 폼이 모델 스키마로 갱신되었습니다.")
                log.debug("train 완료 | n_per_class=%d | metrics=%s | roc_url=%s",
                          n_per_class, result.get("metrics"), result.get("roc_url"))  # [LOG]
            except Exception as e:
                messages.error(request, f"학습 실패: {e}")
                log.exception("train 실패: %s", e)  # [LOG]

        elif action == "predict":
            store = load_model()
            if not store:
                messages.error(request, "먼저 '학습'을 수행해 모델을 저장하세요.")
                log.warning("predict 요청 거부: 모델 없음")  # [LOG]
            else:
                # 예측은 무조건 "모델 스키마" 기준으로!
                model_schema = load_feature_schema()
                model_feature_cols = model_schema.get("feature_cols", [])
                model_num_cols = set(model_schema.get("num_cols", []))

                # 폼도 모델 스키마로 다시 만들기(안전)
                PredictForm = make_predict_form(model_schema)
                form = PredictForm(request.POST)
                context["form"] = form
                context["manual_fields"] = _build_manual_fields_from_schema(model_schema)

                try:
                    log.debug("predict 시작 | features=%d", len(model_feature_cols))  # [LOG]
                    # row는 model_feature_cols "순서"대로 채워야 함
                    if form.is_valid() and form.fields:
                        # 🔴 핵심: DataFrame으로 바로 구성 (컬럼명까지 포함)
                        input_data = {}
                        missing = []
                        for col in model_feature_cols:
                            if col in form.cleaned_data:
                                v = form.cleaned_data[col]
                                # 수치형/범주형 형 변환
                                if col in model_num_cols:
                                    input_data[col] = [float(v)]
                                else:
                                    input_data[col] = [str(v)]
                            else:
                                missing.append(col)
                        if missing:
                            messages.warning(request, f"[WARN] 모델이 기대하는 피처가 폼에 없음: {', '.join(missing)}")
                            log.warning("predict: cleaned_data 누락 | %s", missing)  # [LOG]
                        x_df = pd.DataFrame(input_data, columns=model_feature_cols)
                    else:
                        # 폼이 유효하지 않거나 비어 있으면 manual_fields에서 추출
                        input_data = {}
                        missing = []
                        for col in model_feature_cols:
                            v = request.POST.get(col, None)
                            if v is None or v == "":
                                missing.append(col)
                                input_data[col] = [np.nan]  # 비어있으면 NaN -> imputer로 처리 시도
                            else:
                                if col in model_num_cols:
                                    try:
                                        input_data[col] = [float(v)]
                                    except Exception:
                                        input_data[col] = [np.nan]
                                else:
                                    input_data[col] = [str(v)]
                        if missing:
                            messages.error(request, f"다음 입력이 비어 있습니다: {', '.join(missing)}")
                            log.warning("predict: manual 입력 누락 | %s", missing)  # [LOG]
                            return render(request, "lungapp/index.html", context)
                        x_df = pd.DataFrame(input_data, columns=model_feature_cols)

                    pipe = store["pipeline"]
                    class_names = store["class_names"]

                    # 🔴 넘파이 배열이 아니라 DataFrame을 그대로 전달
                    pred = pipe.predict(x_df)[0]
                    proba = pipe.predict_proba(x_df)[0]
                    prob_table = [(class_names[i], round(float(p), 4)) for i, p in enumerate(proba)]

                    context["pred_label"] = str(pred)
                    context["pred_table"] = prob_table
                    messages.success(request, f"예측 결과: {pred}")
                    log.debug("predict 완료 | label=%s | proba=%s", pred, prob_table)  # [LOG]
                except Exception as e:
                    messages.error(request, f"예측 실패: {e}")
                    log.exception("predict 실패: %s", e)  # [LOG]

    log.debug("✓ index 종료 | context_keys=%s", list(context.keys()))  # [LOG]
    return render(request, "lungapp/model.html", context)
