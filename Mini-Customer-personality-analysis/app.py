import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Marketing Campaign Intelligence",
    page_icon="🎯",
    layout="wide",
)


# ---------------- DATA LOADING ----------------
def load_and_prepare_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    df = df.dropna(subset=["Response"]).copy()

    df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
    df["Income"] = df["Income"].fillna(df["Income"].median())

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y", errors="coerce")
    df["Dt_Customer"] = df["Dt_Customer"].fillna(df["Dt_Customer"].mode()[0])

    current_date = df["Dt_Customer"].max()
    df["Customer_Days"] = (current_date - df["Dt_Customer"]).dt.days

    df["Age"] = 2026 - df["Year_Birth"]
    df["TotalChildren"] = df["Kidhome"] + df["Teenhome"]

    for col in ["Z_CostContact", "Z_Revenue"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    return df


# ---------------- FEATURE ENGINEERING ----------------
def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    y = df["Response"].astype(int).values

    drop_cols = ["Response"]
    if "ID" in df.columns:
        drop_cols.append("ID")

    X = df.drop(columns=drop_cols).copy()

    # ✅ SAFE datetime conversion (FIXED)
    if "Dt_Customer" in X.columns:
        dt = pd.to_datetime(X["Dt_Customer"], errors="coerce")
        dt = dt.fillna(pd.Timestamp("1970-01-01"))
        X["Dt_Customer"] = dt.astype("int64")

    X = pd.get_dummies(X, drop_first=True)

    return X, y


# ---------------- SEGMENTATION ----------------
def make_segments(df: pd.DataFrame) -> pd.DataFrame:
    seg_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
        "Income",
    ]

    seg_data = df[seg_cols].copy()
    seg_scaled = StandardScaler().fit_transform(seg_data)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Segment"] = kmeans.fit_predict(seg_scaled)

    segment_names = {
        0: "Moderate Buyers",
        1: "High Value Customers",
        2: "Low Engagement Customers",
    }

    df["Segment_Name"] = df["Segment"].map(segment_names)

    return df


# ---------------- MODEL TRAINING ----------------
@st.cache_data(show_spinner=False)
def train_pipeline(data_path: str):
    df = load_and_prepare_data(data_path)
    df = make_segments(df)

    X, y = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        random_state=42,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Align full dataset
    X_full, _ = build_feature_matrix(df)
    for col in X_train.columns:
        if col not in X_full.columns:
            X_full[col] = 0
    X_full = X_full[X_train.columns]

    df["Pred_Prob_Response_Yes"] = model.predict_proba(X_full)[:, 1]
    df["Pred_Response"] = np.where(
        df["Pred_Prob_Response_Yes"] >= 0.5, "Yes", "No"
    )

    # SHAP
    if SHAP_AVAILABLE:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_full)
    else:
        shap_values = None

    global_importance = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    return df, model, X_full, shap_values, global_importance, accuracy, auc


# ---------------- UI ----------------
st.markdown(
    """
    <div style="padding:18px;border-radius:14px;background: linear-gradient(90deg,#0ea5e9,#6366f1,#ec4899); color:white;">
        <h1>🎯 Marketing Campaign Intelligence Dashboard</h1>
        <p>Predict response • Segment customers • Explain model decisions</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ✅ FIXED PATH (important for deployment)
data_file = "marketing_campaign.csv"

try:
    df, model, X_full, shap_values, global_importance, accuracy, auc = train_pipeline(data_file)
except FileNotFoundError:
    st.error("❌ Dataset not found. Upload 'marketing_campaign.csv' in same folder.")
    st.stop()


# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("ROC-AUC", f"{auc:.4f}")
col3.metric("Customers", len(df))

st.markdown("---")


# ---------------- FILTERS ----------------
left, right = st.columns([1, 2])

with left:
    segment = st.selectbox("Segment", ["All"] + list(df["Segment_Name"].unique()))

    income = st.slider("Min Income",
                       int(df["Income"].min()),
                       int(df["Income"].max()),
                       int(df["Income"].median()))

    children = st.slider("Max Children",
                         int(df["TotalChildren"].min()),
                         int(df["TotalChildren"].max()),
                         int(df["TotalChildren"].max()))

    recency = st.slider("Max Recency",
                        int(df["Recency"].min()),
                        int(df["Recency"].max()),
                        int(df["Recency"].max()))

    threshold = st.slider("Threshold", 0.1, 0.9, 0.5)

with right:
    st.subheader("Segment Distribution")
    st.bar_chart(df["Segment_Name"].value_counts())


# ---------------- FILTER DATA ----------------
filtered = df.copy()

if segment != "All":
    filtered = filtered[filtered["Segment_Name"] == segment]

filtered = filtered[
    (filtered["Income"] >= income) &
    (filtered["TotalChildren"] <= children) &
    (filtered["Recency"] <= recency)
]

if filtered.empty:
    filtered = df

top = filtered.sort_values("Pred_Prob_Response_Yes", ascending=False).iloc[0]


# ---------------- OUTPUT ----------------
st.success(f"Prediction: {'Yes' if top['Pred_Prob_Response_Yes'] >= threshold else 'No'}")
st.info(f"Probability: {top['Pred_Prob_Response_Yes']:.2%}")
st.warning(f"Segment: {top['Segment_Name']}")


# ---------------- TOP CUSTOMERS ----------------
st.subheader("Top Customers")

st.dataframe(
    filtered.sort_values("Pred_Prob_Response_Yes", ascending=False)
    [["Segment_Name", "Income", "TotalChildren", "Recency", "Pred_Prob_Response_Yes"]]
    .head(10)
)


# ---------------- XAI ----------------
st.subheader("Explanation")

if SHAP_AVAILABLE and shap_values is not None:
    contrib = pd.Series(shap_values[int(top.name)], index=X_full.columns)
    st.dataframe(contrib.sort_values(ascending=False).head(10))
else:
    st.dataframe(global_importance.head(10))


st.success("✅ Project Working Successfully!")
