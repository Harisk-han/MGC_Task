"""
Baseline conversion-likelihood model for MGC leads.

Data decisions (also in README):
1. Dedup first. crm_record_hash reveals ~160 leads entered twice by different
   agents (identical hash, different lead_id). Training on both copies would
   let the same lead land in train AND test after a random split — leakage.
   Kept the first occurrence per hash, dropped the rest.
2. Dropped token_amount_received_pkr. This field is 0 for 91% of the leads
   that are 0, and *every single* converted=1 lead has a non-zero token —
   it's set at booking time, i.e. it's a symptom of conversion, not a
   predictor. Including it would make the model look great and be useless
   (the token amount isn't known until after the thing we're predicting has
   already happened).
3. Dropped lead_id and crm_record_hash — identifiers, no predictive signal.
4. Dropped created_at as a raw field but kept a lightweight derived feature
   (day of week) since response-time effects are plausible; didn't engineer
   further to stay within the "no tuning" baseline scope.
5. city has messy casing/abbreviations (ISLAMABAD / Islamabad / ISB / Rwp /
   khi ...) — normalized to a canonical set before one-hot encoding, otherwise
   the model would treat 'Islamabad' and 'ISB' as unrelated categories and
   silently lose signal.
6. bedrooms is missing for ~40% of rows, but it's missing *because*
   property_type is Plot or Commercial Shop (bedrooms don't apply) — not
   random. Filled with 0 rather than a mean, since 0 correctly encodes "not
   applicable" and a mean would invent a fake bedroom count for a shop.
7. budget_pkr_lac and agent_experience_years: missing for a few hundred rows
   with no structural explanation found — imputed with the median.
8. area: 477 missing, filled as 'Unknown' category rather than dropped, since
   dropping ~5% of rows loses signal for no real gain here.

Metric: the target is heavily imbalanced (~93% not converted / ~7% converted).
Accuracy on this data is close to meaningless — a model that always predicts
"not converted" scores ~93% and is worthless to a sales team deciding who to
call first. Reported PR-AUC (average precision) instead, since it's threshold-
free and, unlike ROC-AUC, doesn't get flattered by the huge majority class —
it directly reflects how well the model ranks the rare positive class, which
is exactly the "who do we call first" use case.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
import joblib
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "leads.csv"
MODEL_PATH = Path(__file__).parent / "model.joblib"

CITY_MAP = {
    "ISLAMABAD": "Islamabad", "ISB": "Islamabad",
    "RAWALPINDI": "Rawalpindi", "RWP": "Rawalpindi",
    "LAHORE": "Lahore",
    "KARACHI": "Karachi", "KHI": "Karachi",
    "PESHAWAR": "Peshawar",
    "FAISALABAD": "Faisalabad",
    "MULTAN": "Multan",
    "GUJRANWALA": "Gujranwala",
    "ABBOTTABAD": "Abbottabad",
}


def normalize_city(city: str) -> str:
    key = str(city).strip().upper()
    return CITY_MAP.get(key, str(city).strip().title())


def load_and_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 1. Dedup on crm_record_hash, keep first occurrence
    before = len(df)
    df = df.drop_duplicates(subset="crm_record_hash", keep="first").copy()
    print(f"Deduped: {before} -> {len(df)} rows ({before - len(df)} duplicates removed)")

    # 2 & 3. Drop leaky / identifier columns
    df = df.drop(columns=["token_amount_received_pkr", "lead_id", "crm_record_hash"])

    # 4. Derive day-of-week from created_at, then drop the raw timestamp
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["created_dow"] = df["created_at"].dt.dayofweek
    df = df.drop(columns=["created_at"])

    # 5. Normalize city
    df["city"] = df["city"].apply(normalize_city)

    # 6. bedrooms: 0 = not applicable (Plot / Commercial Shop)
    df["bedrooms"] = df["bedrooms"].fillna(0)

    # 8. area: fill missing as explicit category
    df["area"] = df["area"].fillna("Unknown")

    return df


def build_pipeline(numeric_cols, categorical_cols) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),  # covers #7 (budget, agent_experience)
        ("scale", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        # class_weight='balanced' because of the ~93/7 imbalance (decision #8/metric note)
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def train_and_save_pipeline(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH):
    df = load_and_clean(data_path)

    y = df["converted"]
    X = df.drop(columns=["converted"])

    numeric_cols = [
        "budget_pkr_lac", "bedrooms", "first_response_minutes", "calls_made",
        "total_call_seconds", "whatsapp_replies", "site_visits",
        "agent_experience_years", "is_overseas", "referred_by_existing_client",
        "has_financing_approved", "created_dow",
    ]
    categorical_cols = ["source", "city", "area", "property_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\nTest set: {len(y_test)} leads, {y_test.sum()} converted ({100*y_test.mean():.1f}%)")
    print(f"\nPrimary metric — PR-AUC (average precision): {pr_auc:.4f}")
    print(f"(baseline / random for this class balance: {y_test.mean():.4f})")
    print(f"For reference — ROC-AUC: {roc_auc:.4f}")
    print("\nClassification report (threshold=0.5, for context only — PR-AUC is the metric that matters here):")
    print(classification_report(y_test, y_pred, digits=3))

    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")
    return pipeline


def main():
    train_and_save_pipeline()


if __name__ == "__main__":
    main()
