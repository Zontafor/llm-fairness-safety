# import json
# import joblib
# import pandas as pd
# from pathlib import Path
# from sklearn.pipeline import Pipeline
# from typing import Dict, Optional, List
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# TARGET_CANDIDATES = ["charges", "premium", "price", "annual_premium"]
# CAT_HINTS = {"sex","gender","region","smoker","smoking_status","marital_status","occupation"}
# NUM_HINTS = {"age","bmi","children","dependents","credit_score","income","mileage"}

# def _infer_schema(df: pd.DataFrame) -> Dict[str, List[str]]:
#     cols_lower = {c: c.lower() for c in df.columns}
#     target = None
#     for cand in TARGET_CANDIDATES:
#         for c, lc in cols_lower.items():
#             if cand == lc:
#                 target = c; break
#         if target: break
#     if target is None:
#         numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
#         if not numeric_cols:
#             raise ValueError("Could not infer a numeric target. Please provide schema_override.")
#         target = numeric_cols[-1]

#     X = df.drop(columns=[target])
#     num_cols, cat_cols = [], []
#     for c in X.columns:
#         lc = c.lower()
#         if lc in NUM_HINTS: num_cols.append(c)
#         elif lc in CAT_HINTS: cat_cols.append(c)

#     remaining = [c for c in X.columns if c not in num_cols + cat_cols]
#     for c in remaining:
#         (num_cols if pd.api.types.is_numeric_dtype(X[c]) else cat_cols).append(c)

#     return {"target": target, "num": num_cols, "cat": cat_cols}

# def schema_report(csv_path="data/insurance.csv", out_dir="figures"):
#     Path(out_dir).mkdir(parents=True, exist_ok=True)
#     df = pd.read_csv(csv_path)
#     schema = _infer_schema(df)

#     # Missing value summary
#     nulls = df.isnull().sum().rename("missing_count").to_frame()
#     nulls["missing_pct"] = (nulls["missing_count"] / len(df)).round(4)

#     # Save artifacts
#     pd.set_option("display.max_rows", 1000)
#     nulls_path = Path(out_dir) / "csv_missing_values.csv"
#     nulls.to_csv(nulls_path, index=True)

#     schema_json = {
#         "inferred_target": schema["target"],
#         "numeric_features": schema["num"],
#         "categorical_features": schema["cat"],
#         "rows": int(len(df)),
#         "cols": int(df.shape[1]),
#         "missing_summary_csv": str(nulls_path)
#     }
#     (Path(out_dir) / "csv_schema.json").write_text(json.dumps(schema_json, indent=2))
#     print("[schema] Saved:", nulls_path, "and", Path(out_dir) / "csv_schema.json")
#     return schema_json, nulls

# def train_or_load_premium_model(
#     csv_path: str = "data/insurance.csv",
#     save_path: str = "models/premium_model.pkl",
#     schema_override: Optional[Dict[str, List[str]]] = None
# ):
#     sp = Path(save_path)
#     if sp.exists():
#         return joblib.load(sp)

#     df = pd.read_csv(csv_path)
#     schema = schema_override or _infer_schema(df)
#     y = df[schema["target"]]
#     X = df.drop(columns=[schema["target"]])

#     # --- NEW: remember the training feature order for inference ---
#     trained_columns = list(X.columns)  # NEW

#     # --- imputers before scaling/encoding ---
#     num_pipe = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scale", StandardScaler(with_mean=False)),
#     ])
#     cat_pipe = Pipeline([
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("ohe", OneHotEncoder(handle_unknown="ignore")),
#     ])

#     pre = ColumnTransformer([
#         ("num", num_pipe, schema["num"]),
#         ("cat", cat_pipe, schema["cat"]),
#     ])

#     model = Pipeline([
#         ("pre", pre),
#         ("rf", RandomForestRegressor(n_estimators=300, random_state=42))
#     ])

#     Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
#     model.fit(Xtr, ytr)

#     model.trained_schema_ = schema
#     model.trained_columns_ = X.columns.tolist()

#     sp.parent.mkdir(parents=True, exist_ok=True)
#     joblib.dump(model, sp)

#     try:
#         r2 = model.score(Xte, yte)
#         print(f"[premium_model] Trained. R^2 on holdout: {r2:.3f}")
#     except Exception:
#         pass
#     return model

#     try:
#         pre = model.named_steps["pre"]
#         trained_cols = []
#         for _, _, colsel in pre.transformers_:
#             if isinstance(colsel, (list, tuple, pd.Index)):
#                 trained_cols.extend(list(colsel))
#         model.trained_columns_ = trained_cols
#     except Exception:
#         pass

#     # Attach schema and force NumPy output to avoid pandas-name path ---
#     model.trained_columns_ = trained_columns  # NEW
#     try:
#         model.set_output(transform="default")  # NEW (sklearn>=1.2); safe to ignore if not available
#     except Exception:
#         pass

#     sp.parent.mkdir(parents=True, exist_ok=True)
#     joblib.dump(model, sp)

#     try:
#         r2 = model.score(Xte, yte)
#         print(f"[premium_model] Trained. R^2 on holdout: {r2:.3f}")
#     except Exception:
#         pass
#     return model

import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from typing import Dict, Optional, List
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_CANDIDATES = ["charges", "premium", "price", "annual_premium"]
CAT_HINTS = {"sex","gender","region","smoker","smoking_status","marital_status","occupation"}
NUM_HINTS = {"age","bmi","children","dependents","credit_score","income","mileage"}

def _infer_schema(df: pd.DataFrame) -> Dict[str, List[str]]:
    cols_lower = {c: c.lower() for c in df.columns}
    target = None
    for cand in TARGET_CANDIDATES:
        for c, lc in cols_lower.items():
            if cand == lc:
                target = c; break
        if target: break
    if target is None:
        numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
        if not numeric_cols:
            raise ValueError("Could not infer a numeric target. Please provide schema_override.")
        target = numeric_cols[-1]

    X = df.drop(columns=[target])
    num_cols, cat_cols = [], []
    for c in X.columns:
        lc = c.lower()
        if lc in NUM_HINTS: num_cols.append(c)
        elif lc in CAT_HINTS: cat_cols.append(c)

    remaining = [c for c in X.columns if c not in num_cols + cat_cols]
    for c in remaining:
        (num_cols if pd.api.types.is_numeric_dtype(X[c]) else cat_cols).append(c)

    return {"target": target, "num": num_cols, "cat": cat_cols}

def train_or_load_premium_model(
    csv_path: str = "data/insurance.csv",
    save_path: str = "models/premium_model.pkl",
    schema_override: Optional[Dict[str, List[str]]] = None
):
    sp = Path(save_path)
    if sp.exists():
        model = joblib.load(sp)
        # If the cached model is missing the schema attribute, force retrain once.
        if not hasattr(model, "trained_columns_"):
            try:
                sp.unlink()
            except Exception:
                pass
        else:
            return model

    df = pd.read_csv(csv_path)
    schema = schema_override or _infer_schema(df)
    y = df[schema["target"]]
    X = df.drop(columns=[schema["target"]])

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, schema["num"]),
        ("cat", cat_pipe, schema["cat"]),
    ])

    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=42))
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(Xtr, ytr)

    model.trained_columns_ = X.columns.tolist()

    sp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, sp)

    try:
        r2 = model.score(Xte, yte)
        print(f"[premium_model] Trained. R^2 on holdout: {r2:.3f}")
    except Exception:
        pass
    return model