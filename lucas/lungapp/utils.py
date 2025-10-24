import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from django.conf import settings
from django.db import transaction                # DB 트랜잭션용 추가
from lungapp.models import LungSample            # DB에서 데이터 읽기용 추가

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, RocCurveDisplay
)

# 저장 경로
MODEL_PATH = settings.MEDIA_ROOT / "lung_svm_model.joblib"
META_PATH  = settings.MEDIA_ROOT / "lung_model_meta.json"
ROC_PATH   = settings.MEDIA_ROOT / "lung_roc.png"

_DEFAULT_EXCLUDES = {
    "pation_id", "patient_id", "id", "ID",
    "target", "label", "class", "y", "diagnosis", "outcome",
    "created_at",
}
EXCLUDE_FEATURES = set(getattr(settings, "LUNG_EXCLUDE_FEATURES", [])) | _DEFAULT_EXCLUDES


# DB 데이터를 DataFrame으로 변환하는 보조함수
def _qs_to_df(qs) -> pd.DataFrame:
    rows = list(qs.values())
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    # 🔸 변경됨: DB에서 읽은 직후 불필요 컬럼(id, created_at) 제거
    return df.drop(columns=[c for c in ["id", "created_at"] if c in df.columns], errors="ignore")


# 균형 샘플 DataFrame을 lung_samples 테이블에 bulk insert
def _bulk_insert_df(df: pd.DataFrame):
    objs = []
    for _, r in df.iterrows():
        objs.append(LungSample(
            patient_id=int(r.get("patient_id", 0)),
            age=int(r.get("age", 0)),
            gender=str(r.get("gender", "")),
            pack_years=(float(r["pack_years"]) if pd.notna(r.get("pack_years")) else None),
            radon_exposure=str(r.get("radon_exposure", "")),
            asbestos_exposure=str(r.get("asbestos_exposure", "")),
            secondhand_smoke_exposure=str(r.get("secondhand_smoke_exposure", "")),
            copd_diagnosis=str(r.get("copd_diagnosis", "")),
            alcohol_consumption=str(r.get("alcohol_consumption", "")),
            family_history=str(r.get("family_history", "")),
            lung_cancer=str(r.get("lung_cancer", r.get("target", ""))),
        ))
    with transaction.atomic():
        LungSample.objects.bulk_create(objs, batch_size=1000)


# 🔸 get_dataset_df() — DB가 우선, 없으면 CSV→정제→균형→DB 저장
def get_dataset_df() -> pd.DataFrame:
    """
    DB에 데이터가 있으면 DB→DataFrame 반환.
    없으면 CSV를 읽고 (NA 없는 행만) → 양/불 500개씩 균형샘플을 만들어
    lung_samples 테이블에 저장한 뒤 그 데이터를 반환.
    """
    qs = LungSample.objects.all()
    df_db = _qs_to_df(qs)
    if not df_db.empty:
        preferred = [
            "patient_id","age","gender","pack_years",
            "radon_exposure","asbestos_exposure","secondhand_smoke_exposure",
            "copd_diagnosis","alcohol_consumption","family_history",
            "lung_cancer"
        ]
        others = [c for c in df_db.columns if c not in preferred]
        return df_db[[c for c in preferred if c in df_db.columns] + others]

    # DB가 비었으면 CSV에서 생성
    csv_path = settings.LUNG_CANCER_CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    # 타깃 추정
    target_col = getattr(settings, "LUNG_TARGET_NAME", None)
    if not target_col or target_col not in df.columns:
        for c in ["target","label","class","y","diagnosis","outcome","LUNG_CANCER","lung_cancer"]:
            if c in df.columns:
                target_col = c
                break
        else:
            target_col = df.columns[-1]

    # NA 없는 행만 사용
    excluded = set(EXCLUDE_FEATURES) | {target_col}
    usable_cols = [c for c in df.columns if c not in excluded] + [target_col]
    df_clean = df[usable_cols].dropna(axis=0, how="any").reset_index(drop=True)
    if df_clean.empty:
        raise ValueError("CSV에서 NA 없는 행을 찾지 못했습니다.")

    # 균형 샘플 (각 500)
    df_bal, top2 = _balanced_binary_sample(df_clean, target_col, n_per_class=500, random_state=42)

    # target명을 모델 스키마에 맞게 lung_cancer로 통일
    if target_col != "lung_cancer" and "lung_cancer" not in df_bal.columns:
        df_bal = df_bal.rename(columns={target_col: "lung_cancer"})

    _bulk_insert_df(df_bal)

    # DB 다시 읽어서 반환 (여기서도 _qs_to_df가 id/created_at 제거)
    return _qs_to_df(LungSample.objects.all())


def _infer_target_column(df: pd.DataFrame) -> str:
    target_from_settings = getattr(settings, "LUNG_TARGET_NAME", None)
    if target_from_settings and target_from_settings in df.columns:
        return target_from_settings
    for c in ["target", "label", "class", "y", "diagnosis", "outcome", "LUNG_CANCER", "lung_cancer"]:
        if c in df.columns:
            return c
    return df.columns[-1]


def _split_feature_types_excluding(df: pd.DataFrame, target_col: str):
    excluded = set(EXCLUDE_FEATURES) | {target_col}
    usable_cols = [c for c in df.columns if c not in excluded]
    num_cols = df[usable_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in usable_cols if c not in num_cols]
    feature_cols = usable_cols[:]
    return feature_cols, num_cols, cat_cols


def _build_pipeline(num_cols, cat_cols, random_state=42):
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop"
    )
    pipe = Pipeline([
        ("preprocess", pre),
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state)),
    ])
    return pipe


def _balanced_binary_sample(df: pd.DataFrame, target_col: str, n_per_class: int = 1000, random_state: int = 42):
    vc = df[target_col].value_counts().sort_values(ascending=False)
    top_classes = vc.index.tolist()[:2]
    if len(top_classes) < 2:
        raise ValueError(f"타깃 클래스가 2개 미만입니다. 고유값: {df[target_col].unique()!r}")
    rng = np.random.RandomState(random_state)
    parts = []
    for cls in top_classes:
        sub = df[df[target_col] == cls]
        take = min(len(sub), n_per_class)
        parts.append(sub.sample(n=take, random_state=rng))
    small = pd.concat(parts, axis=0).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    return small, top_classes


def _save_roc(y_true, proba, class_names, out_path):
    plt.figure(figsize=(7, 5))
    classes = np.unique(y_true)
    if len(classes) == 2:
        y_bin = label_binarize(y_true, classes=classes).ravel()
        RocCurveDisplay.from_predictions(y_bin, proba[:, 1], name="ROC — positive")
        plt.plot([0, 1], [0, 1], "--", lw=1)
        plt.title("ROC (Binary)")
    else:
        y_bin = label_binarize(y_true, classes=classes)
        for i in range(len(classes)):
            name = class_names[i] if i < len(class_names) else f"Class {i}"
            RocCurveDisplay.from_predictions(y_bin[:, i], proba[:, i], name=f"ROC — {name}")
        plt.plot([0, 1], [0, 1], "--", lw=1)
        plt.title("ROC (Multiclass OvR)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def build_schema_from_dataframe(df: pd.DataFrame):
    target_col = _infer_target_column(df)
    feature_cols, num_cols, cat_cols = _split_feature_types_excluding(df, target_col)
    cat_choices = {}
    for col in cat_cols:
        vals = df[col].astype(str).dropna().unique().tolist()
        cat_choices[col] = sorted(vals)[:20]
    num_stats = {}
    for col in num_cols:
        s = df[col]
        try:
            num_stats[col] = {"min": float(np.nanmin(s)), "max": float(np.nanmax(s))}
        except Exception:
            num_stats[col] = {"min": None, "max": None}
    return {
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_choices": cat_choices,
        "num_stats": num_stats,
    }


def train_and_save_model(test_size=0.2, random_state=42, n_per_class: int = 1000):
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    df_full = get_dataset_df()   # DB 버전으로 자동 대체됨 (이미 id/created_at 제거된 DF)
    target_col = _infer_target_column(df_full)
    df_small, used_classes = _balanced_binary_sample(
        df_full, target_col, n_per_class=n_per_class, random_state=random_state
    )
    y_raw = df_small[target_col]
    drop_cols = {target_col} | (set(df_small.columns) & EXCLUDE_FEATURES)
    X_df = df_small.drop(columns=list(drop_cols))
    y = y_raw.astype(str).values
    class_names = pd.Series(y).astype("category").cat.categories.tolist()
    feature_cols, num_cols, cat_cols = _split_feature_types_excluding(df_small, target_col)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pipe = _build_pipeline(num_cols, cat_cols, random_state=random_state)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)
    metrics = {
        "Samples per class": {str(k): int((df_small[target_col] == k).sum()) for k in used_classes},
        "Total samples": int(len(df_small)),
        "Accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
        "Precision (macro)": round(float(precision_score(y_te, y_pred, average="macro", zero_division=0)), 4),
        "Recall (macro)": round(float(recall_score(y_te, y_pred, average="macro", zero_division=0)), 4),
        "F1 (macro)": round(float(f1_score(y_te, y_pred, average="macro", zero_division=0)), 4),
    }
    classes = np.unique(y_te)
    if len(classes) == 2:
        y_bin = label_binarize(y_te, classes=classes).ravel()
        auc = roc_auc_score(y_bin, y_proba[:, 1])
        metrics["AUC"] = round(float(auc), 4)
    else:
        y_bin = label_binarize(y_te, classes=classes)
        auc = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
        metrics["AUC (macro, OvR)"] = round(float(auc), 4)
    _save_roc(y_te, y_proba, class_names, ROC_PATH)
    cat_choices = {}
    if len(cat_cols) > 0:
        onehot = pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
        for col, cats in zip(cat_cols, onehot.categories_):
            cat_choices[col] = [str(c) for c in cats]
    num_stats = {}
    for col in num_cols:
        s = X_df[col] if col in X_df.columns else df_small[col]
        try:
            num_stats[col] = {"min": float(np.nanmin(s)), "max": float(np.nanmax(s))}
        except Exception:
            num_stats[col] = {"min": None, "max": None}
    joblib.dump(
        {
            "pipeline": pipe,
            "class_names": class_names,
            "feature_cols": feature_cols,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "cat_choices": cat_choices,
            "num_stats": num_stats,
        },
        MODEL_PATH,
    )
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_col": target_col,
                "feature_cols": feature_cols,
                "class_names": class_names,
                "used_classes": [str(c) for c in used_classes],
                "n_per_class": int(n_per_class),
                "total_samples_used": int(len(df_small)),
                "num_cols": num_cols,
                "cat_cols": cat_cols,
                "cat_choices": cat_choices,
                "num_stats": num_stats,
                "excluded_features": sorted(list(EXCLUDE_FEATURES)),
            },
            f, ensure_ascii=False, indent=2,
        )
    return {
        "metrics": metrics,
        "roc_url": settings.MEDIA_URL + ROC_PATH.name,
        "model_saved": True,
        "feature_cols": feature_cols,
    }


def load_model():
    if os.path.exists(MODEL_PATH):
        store = joblib.load(MODEL_PATH)
        return {
            "pipeline": store.get("pipeline"),
            "class_names": store.get("class_names", []),
            "feature_cols": store.get("feature_cols", []),
            "num_cols": store.get("num_cols", []),
            "cat_cols": store.get("cat_cols", []),
            "cat_choices": store.get("cat_choices", {}),
            "num_stats": store.get("num_stats", {}),
        }
    return None


def load_feature_schema():
    store = load_model()
    if store:
        return {
            "feature_cols": store.get("feature_cols", []),
            "num_cols": store.get("num_cols", []),
            "cat_cols": store.get("cat_cols", []),
            "cat_choices": store.get("cat_choices", {}),
            "num_stats": store.get("num_stats", {}),
        }
    df = get_dataset_df()
    return build_schema_from_dataframe(df)


def get_feature_cols_from_data_or_model():
    schema = load_feature_schema()
    return schema["feature_cols"]




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