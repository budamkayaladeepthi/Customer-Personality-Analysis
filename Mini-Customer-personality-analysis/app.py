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
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


st.set_page_config(
    page_title="Marketing Campaign Intelligence",
    page_icon="🎯",
    layout="wide",
)


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


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    y = df["Response"].astype(int).values
    # Exclude unique identifier so the model doesn't learn from arbitrary IDs.
    drop_cols = ["Response"]
    if "ID" in df.columns:
        drop_cols.append("ID")
    X = df.drop(columns=drop_cols).copy()

    if "Dt_Customer" in X.columns:
        X["Dt_Customer"] = pd.to_datetime(X["Dt_Customer"], errors="coerce").view("int64")

    X = pd.get_dummies(X, drop_first=True)
    return X, y


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

    X_full, _ = build_feature_matrix(df)
    for col in X_train.columns:
        if col not in X_full.columns:
            X_full[col] = 0
    X_full = X_full[X_train.columns]

    df["Pred_Prob_Response_Yes"] = model.predict_proba(X_full)[:, 1]
    df["Pred_Response"] = np.where(df["Pred_Prob_Response_Yes"] >= 0.5, "Yes", "No")

    if SHAP_AVAILABLE:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_full)
    else:
        explainer = None
        shap_values = None

    global_importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(
        ascending=False
    )

    return df, model, X_full, shap_values, global_importance, accuracy, auc


st.markdown(
    """
    <div style="padding:18px;border-radius:14px;background: linear-gradient(90deg,#0ea5e9,#6366f1,#ec4899); color:white;">
        <h1 style="margin-bottom:6px;">🎯 Marketing Campaign Intelligence Dashboard</h1>
        <p style="margin:0;">Predict response • Segment customers • Explain model decisions (XAI)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

data_file = "../Data/marketing_campaign.csv"

try:
    df, model, X_full, shap_values, global_importance, accuracy, auc = train_pipeline(data_file)
except FileNotFoundError:
    st.error("File not found: marketing_campaign.csv. Keep app.py in the same folder as the dataset.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", f"{accuracy:.4f}")
col2.metric("ROC-AUC", f"{auc:.4f}")
col3.metric("Customers", f"{len(df)}")

st.markdown("---")

left, right = st.columns([1, 2])
with left:
    st.subheader("Controls")
    segment_options = ["All"] + sorted(df["Segment_Name"].dropna().unique().tolist())
    selected_segment = st.selectbox("Select Segment", segment_options)
    income_threshold = st.slider(
        "Min Income",
        int(df["Income"].min()),
        int(df["Income"].max()),
        int(df["Income"].median()),
        1000,
    )
    max_children = st.slider(
        "Max Total Children",
        int(df["TotalChildren"].min()),
        int(df["TotalChildren"].max()),
        int(df["TotalChildren"].max()),
        1,
    )
    max_recency = st.slider(
        "Max Recency (days since last purchase)",
        int(df["Recency"].min()),
        int(df["Recency"].max()),
        int(df["Recency"].max()),
        1,
    )
    threshold = st.slider("Prediction Threshold", 0.05, 0.95, 0.50, 0.05)

with right:
    st.subheader("Segment Distribution")
    seg_counts = df["Segment_Name"].value_counts()
    st.bar_chart(seg_counts)

filtered_df = df.copy()
if selected_segment != "All":
    filtered_df = filtered_df[filtered_df["Segment_Name"] == selected_segment]

eligible_customers = filtered_df[
    (filtered_df["Income"] >= income_threshold)
    & (filtered_df["TotalChildren"] <= max_children)
    & (filtered_df["Recency"] <= max_recency)
]
if eligible_customers.empty:
    st.info(
        "No customer matches all selected filters. Showing best match from selected segment."
    )
    eligible_customers = filtered_df

customer_row = eligible_customers.sort_values("Pred_Prob_Response_Yes", ascending=False).iloc[0]
customer_idx = int(customer_row.name)
prob = float(customer_row["Pred_Prob_Response_Yes"])
pred = "Yes" if prob >= threshold else "No"

card1, card2, card3 = st.columns(3)
card1.info(f"**Predicted Response:** {pred}")
card2.success(f"**Response Probability (Yes):** {prob:.2%}")
card3.warning(f"**Segment:** {customer_row['Segment_Name']}")

st.markdown("### Top Likely Responders (Filtered)")
top_responders = eligible_customers[
    eligible_customers["Pred_Prob_Response_Yes"] >= threshold
].sort_values("Pred_Prob_Response_Yes", ascending=False)

display_cols = [
    "Segment_Name",
    "Income",
    "TotalChildren",
    "Recency",
    "Pred_Prob_Response_Yes",
]
st.dataframe(top_responders[display_cols].head(10))

st.markdown("### Why this prediction? (XAI)")
if SHAP_AVAILABLE and shap_values is not None:
    contrib = pd.Series(shap_values[customer_idx], index=X_full.columns)
    top_pos = contrib.sort_values(ascending=False).head(5)
    top_neg = contrib.sort_values(ascending=True).head(5)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top Positive Factors**")
        st.dataframe(top_pos.rename("SHAP Value"))
    with c2:
        st.markdown("**Top Negative Factors**")
        st.dataframe(top_neg.rename("SHAP Value"))
else:
    st.info("SHAP is not installed. Showing global feature importance instead.")
    st.dataframe(global_importance.head(10).rename("Importance"))

st.markdown("### Final Output of Project")
st.success(
    "The system predicts campaign response, segments customers by behavior/spending, "
    "and explains likely response drivers for each customer."
)
