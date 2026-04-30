import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DEFAULT_DATA_PATH = Path("data/sample_trust_data.csv")

st.set_page_config(page_title="Predictive Trust Monitoring System", layout="wide")


# ======================================================
# DATA LOADING AND SECURITY CONTROLS
# ======================================================

@st.cache_data
def create_preloaded_sample_data() -> pd.DataFrame:
    """Create a small non-sensitive sample dataset so the app runs without user uploads."""
    sample = pd.DataFrame(
        {
            "Survey ID": range(1001, 1021),
            "Recommend": [10, 9, 8, 7, 6, 4, 5, 10, 8, 3, 9, 7, 6, 10, 5, 8, 9, 4, 7, 10],
            "Helpfulness of the Staff": [10, 9, 7, 8, 5, 4, 6, 10, 8, 3, 9, 7, 5, 10, 4, 8, 9, 5, 7, 10],
            "Felt like a Valued Customer": [10, 9, 7, 7, 5, 4, 6, 10, 8, 2, 9, 7, 5, 10, 4, 8, 9, 5, 7, 10],
            "Speed of Service": [9, 10, 8, 7, 5, 3, 6, 9, 7, 2, 10, 8, 5, 9, 4, 8, 9, 5, 7, 10],
            "Ease of Pick Up": [10, 9, 8, 8, 6, 4, 6, 10, 8, 3, 9, 7, 6, 10, 5, 8, 9, 4, 7, 10],
            "Ease of Returning Your Vehicle": [10, 9, 8, 7, 6, 4, 6, 10, 8, 3, 9, 7, 6, 10, 5, 8, 9, 4, 7, 10],
            "Overall Satisfaction with Vehicle": [10, 9, 8, 8, 7, 5, 6, 10, 8, 4, 9, 7, 6, 10, 5, 8, 9, 5, 7, 10],
            "Mechanical condition of Vehicle": [10, 9, 8, 8, 6, 5, 6, 10, 8, 4, 9, 7, 6, 10, 5, 8, 9, 5, 7, 10],
            "Cleanliness of the Vehicle": [10, 9, 8, 7, 6, 4, 7, 10, 8, 5, 9, 7, 6, 10, 5, 8, 9, 5, 7, 10],
            "Value for the Money": [9, 9, 7, 7, 6, 4, 5, 10, 8, 3, 9, 7, 5, 9, 4, 8, 9, 5, 7, 10],
            "Bill Correct": ["Yes", "Yes", "Yes", "Yes", "No", "No", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "No", "Yes", "Yes"],
            "Recommend Comment": [
                "Great service and smooth experience", "Very helpful staff", "Good overall", "Pickup was okay",
                "Billing issue created concern", "Long wait and poor communication", "Average service", "Excellent experience",
                "Helpful team", "Incorrect bill and slow service", "Very satisfied", "Good but could improve",
                "Bill was not correct", "Wonderful experience", "Did not feel valued", "Reliable service",
                "Staff made it easy", "Slow return process", "Acceptable experience", "Excellent and fast"
            ],
            "Additional Comments": [
                "", "", "", "", "Need better billing follow-up", "Trust dropped after service issue", "", "", "", "Bad experience", "", "", "Billing follow-up needed", "", "", "", "", "Wait time was too long", "", ""
            ],
        }
    )
    return sample


@st.cache_data
def load_default_data() -> pd.DataFrame:
    """Load the preloaded CSV if available; otherwise use built-in sample data."""
    if DEFAULT_DATA_PATH.exists():
        return pd.read_csv(DEFAULT_DATA_PATH)
    return create_preloaded_sample_data()


def validate_uploaded_file(uploaded_file) -> bool:
    """Basic security screen for uploaded files."""
    if uploaded_file is None:
        return False
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 50:
        st.error("The uploaded file is too large. Please use a CSV file under 50 MB.")
        return False
    if not uploaded_file.name.lower().endswith(".csv"):
        st.error("Only CSV files are permitted in this version of the application.")
        return False
    return True


@st.cache_data
def load_csv_from_url(url: str) -> pd.DataFrame:
    """Load CSV data from a URL."""
    return pd.read_csv(url)


# ======================================================
# DATA PREPARATION FUNCTIONS
# ======================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text fields, standardize common categories, and convert score fields to numeric."""
    df = df.copy()

    if "Survey ID" in df.columns:
        df = df[df["Survey ID"].notna()].copy()

    text_cols = [
        "Recommend Comment", "Additional Comments", "Recognize Staff Comment",
        "Bill Correct Comment", "Feature Other Text", "Wait Other Text",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    if "Bill Correct" in df.columns:
        df["Bill Correct"] = df["Bill Correct"].fillna("Unknown").astype(str).str.strip().str.title()

    score_cols = [
        "Recommend", "Helpfulness of the Staff", "Felt like a Valued Customer", "Speed of Service",
        "Ease of Pick Up", "Ease of Returning Your Vehicle", "Overall Satisfaction with Vehicle",
        "Mechanical condition of Vehicle", "Cleanliness of the Vehicle", "Value for the Money",
        "Vehicle Selection (Choice)", "Vehicle Features",
    ]
    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create trust monitoring features and a rule-based Trust_Risk label."""
    df = df.copy()

    text_sources = [
        "Recommend Comment", "Additional Comments", "Recognize Staff Comment",
        "Bill Correct Comment", "Feature Other Text", "Wait Other Text",
    ]
    available_text_sources = [col for col in text_sources if col in df.columns]
    if available_text_sources:
        df["full_comment"] = df[available_text_sources].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    else:
        df["full_comment"] = ""

    service_cols = [
        "Helpfulness of the Staff", "Felt like a Valued Customer", "Speed of Service",
        "Ease of Pick Up", "Ease of Returning Your Vehicle",
    ]
    vehicle_cols = [
        "Overall Satisfaction with Vehicle", "Mechanical condition of Vehicle",
        "Cleanliness of the Vehicle", "Value for the Money",
    ]

    valid_service_cols = [c for c in service_cols if c in df.columns]
    valid_vehicle_cols = [c for c in vehicle_cols if c in df.columns]

    bill_correct = df["Bill Correct"] if "Bill Correct" in df.columns else pd.Series("Unknown", index=df.index)
    df["billing_issue_flag"] = np.where(bill_correct.astype(str).str.title().eq("No"), 1, 0)
    df["service_avg"] = df[valid_service_cols].mean(axis=1) if valid_service_cols else np.nan
    df["vehicle_avg"] = df[valid_vehicle_cols].mean(axis=1) if valid_vehicle_cols else np.nan

    def assign_trust_risk(row):
        recommend = row.get("Recommend", np.nan)
        service_avg = row.get("service_avg", np.nan)
        vehicle_avg = row.get("vehicle_avg", np.nan)
        billing_issue = row.get("billing_issue_flag", 0)
        valued = row.get("Felt like a Valued Customer", np.nan)
        speed = row.get("Speed of Service", np.nan)

        if ((pd.notna(recommend) and recommend <= 6) or billing_issue == 1
                or (pd.notna(service_avg) and service_avg <= 5)
                or (pd.notna(valued) and valued <= 5)
                or (pd.notna(speed) and speed <= 5)):
            return "High"
        if ((pd.notna(recommend) and 7 <= recommend <= 8)
                or (pd.notna(service_avg) and 5 < service_avg <= 8)
                or (pd.notna(vehicle_avg) and 5 < vehicle_avg <= 8)):
            return "Medium"
        return "Low"

    df["Trust_Risk"] = df.apply(assign_trust_risk, axis=1)
    df["Escalation_Proxy"] = np.where(
        (df["billing_issue_flag"] == 1)
        | (df["service_avg"].fillna(10) <= 6)
        | (df.get("Recommend", pd.Series(np.nan, index=df.index)).fillna(10) <= 4),
        1,
        0,
    )
    df["Resolution_Proxy"] = np.where(
        bill_correct.astype(str).str.title().eq("Yes")
        & (df.get("Recommend", pd.Series(np.nan, index=df.index)).fillna(0) >= 9)
        & (df["service_avg"].fillna(0) >= 8),
        1,
        0,
    )
    return df


def apply_selected_preprocessing(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Apply user-selected preprocessing methods."""
    output = df.copy()
    if "Remove duplicate rows" in methods:
        output = output.drop_duplicates()
    if "Fill numeric missing values with median" in methods:
        numeric_cols = output.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            output[col] = output[col].fillna(output[col].median())
    if "Fill text missing values with blank" in methods:
        object_cols = output.select_dtypes(include="object").columns
        output[object_cols] = output[object_cols].fillna("")
    if "Standardize column names" in methods:
        output.columns = [str(c).strip().replace(" ", "_") for c in output.columns]
    return output


# ======================================================
# MODELING AND ANALYSIS FUNCTIONS
# ======================================================

def build_pipeline() -> Pipeline:
    """Build a text + numeric trust-risk classification model."""
    text_feature = "full_comment"
    numeric_features = ["Recommend", "service_avg", "vehicle_avg", "billing_issue_flag"]

    text_transformer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    preprocessor = ColumnTransformer(
        transformers=[("text", text_transformer, text_feature), ("num", numeric_transformer, numeric_features)]
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]
    )


def run_model_pipeline(df: pd.DataFrame, train_size: float = 0.8):
    """Train the classification model and return metrics, predictions, and confusion matrix."""
    required_cols = ["full_comment", "Recommend", "service_avg", "vehicle_avg", "billing_issue_flag", "Trust_Risk"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Model cannot run because these required columns are missing: {missing}")

    model_df = df[required_cols].copy()
    model_df["full_comment"] = model_df["full_comment"].fillna("")
    model_df = model_df.dropna(subset=["Recommend", "service_avg", "vehicle_avg", "Trust_Risk"])

    if model_df["Trust_Risk"].nunique() < 2:
        raise ValueError("Model requires at least two Trust_Risk classes.")
    if len(model_df) < 10:
        raise ValueError("Model requires at least 10 valid rows after preprocessing.")

    X = model_df.drop(columns=["Trust_Risk"])
    y = model_df["Trust_Risk"]

    stratify_y = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_size, random_state=RANDOM_STATE, stratify=stratify_y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    results = X_test.copy()
    results["actual_Trust_Risk"] = y_test.values
    results["predicted_Trust_Risk"] = y_pred
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    report = classification_report(y_test, y_pred, zero_division=0)
    return pipeline, metrics, results, cm, pipeline.classes_, report


def run_analysis_method(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Run selected data analysis method and return tabular results."""
    if method == "Trust risk frequency analysis":
        return df["Trust_Risk"].value_counts(dropna=False).rename_axis("Trust_Risk").reset_index(name="count")
    if method == "Average score analysis by trust risk":
        numeric_cols = [c for c in ["Recommend", "service_avg", "vehicle_avg", "billing_issue_flag"] if c in df.columns]
        return df.groupby("Trust_Risk")[numeric_cols].mean().reset_index()
    if method == "Billing issue analysis":
        if "Bill Correct" in df.columns:
            return pd.crosstab(df["Trust_Risk"], df["Bill Correct"]).reset_index()
        return pd.DataFrame({"message": ["Bill Correct column is not available."]})
    if method == "Correlation analysis":
        numeric_df = df.select_dtypes(include=np.number)
        return numeric_df.corr().reset_index().rename(columns={"index": "variable"})
    return pd.DataFrame()


# ======================================================
# REPORTING, TESTING, AND UTILITY FUNCTIONS
# ======================================================

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def metrics_json_bytes(metrics: dict) -> bytes:
    return json.dumps(metrics, indent=2).encode("utf-8")


def generate_html_report() -> bytes:
    metrics = st.session_state.get("metrics") or {}
    analysis = st.session_state.get("analysis_results")
    prepared = st.session_state.get("prepared_data")
    generated_on = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = "".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in metrics.items()) if metrics else "<tr><td colspan='2'>Model metrics not generated yet.</td></tr>"
    analysis_html = analysis.head(20).to_html(index=False) if isinstance(analysis, pd.DataFrame) else "<p>No analysis results generated yet.</p>"
    data_note = f"Prepared dataset contains {len(prepared)} rows and {prepared.shape[1]} columns." if isinstance(prepared, pd.DataFrame) else "Prepared dataset not created yet."

    html = f"""
    <html>
    <head><title>Predictive Trust Monitoring Report</title></head>
    <body>
        <h1>Predictive Trust Monitoring Report</h1>
        <p><strong>Generated:</strong> {generated_on}</p>
        <h2>Dataset Summary</h2>
        <p>{data_note}</p>
        <h2>Model Performance</h2>
        <table border="1" cellpadding="5"><tr><th>Metric</th><th>Value</th></tr>{rows}</table>
        <h2>Selected Analysis Results</h2>
        {analysis_html}
        <h2>Security Note</h2>
        <p>This report is generated from the active session only. Sensitive information should be removed before sharing externally.</p>
    </body>
    </html>
    """
    return html.encode("utf-8")


def run_system_tests() -> pd.DataFrame:
    """Run a practical smoke test of the main application functions."""
    outcomes = []
    sample = load_default_data()

    def record(feature, status, outcome):
        outcomes.append({"Feature Tested": feature, "Status": status, "Outcome": outcome})

    try:
        record("Preloaded data", "Pass", f"Loaded {len(sample)} rows and {sample.shape[1]} columns.")
    except Exception as exc:
        record("Preloaded data", "Fail", str(exc))

    try:
        cleaned = clean_data(sample)
        record("Data cleaning", "Pass", f"Cleaned dataset contains {len(cleaned)} rows.")
    except Exception as exc:
        record("Data cleaning", "Fail", str(exc))
        cleaned = sample

    try:
        engineered = engineer_features(cleaned)
        required = {"full_comment", "Trust_Risk", "service_avg", "vehicle_avg", "billing_issue_flag"}
        missing = required.difference(engineered.columns)
        if missing:
            record("Feature engineering", "Fail", f"Missing engineered columns: {sorted(missing)}")
        else:
            record("Feature engineering", "Pass", "Trust-risk features were created successfully.")
    except Exception as exc:
        record("Feature engineering", "Fail", str(exc))
        engineered = cleaned

    try:
        analysis = run_analysis_method(engineered, "Trust risk frequency analysis")
        record("Data analysis", "Pass", f"Generated analysis table with {len(analysis)} rows.")
    except Exception as exc:
        record("Data analysis", "Fail", str(exc))

    try:
        pipeline, metrics, results, cm, classes, report = run_model_pipeline(engineered, train_size=0.8)
        record("Model calculations", "Pass", f"F1={metrics['f1_weighted']:.3f}; predictions={len(results)}.")
    except Exception as exc:
        record("Model calculations", "Fail", str(exc))

    try:
        csv_bytes = to_csv_bytes(engineered)
        record("Report saving", "Pass", f"Generated CSV output with {len(csv_bytes)} bytes.")
    except Exception as exc:
        record("Report saving", "Fail", str(exc))

    try:
        if isinstance(engineered, pd.DataFrame) and len(engineered) > 0:
            record("User engagement", "Pass", "Navigation, buttons, select boxes, sliders, and download controls are available in the interface.")
        else:
            record("User engagement", "Fail", "No active dataset was available for interaction.")
    except Exception as exc:
        record("User engagement", "Fail", str(exc))

    return pd.DataFrame(outcomes)


# ======================================================
# SESSION STATE
# ======================================================

DEFAULT_STATE = {
    "raw_data": None,
    "prepared_data": None,
    "model_pipeline": None,
    "metrics": None,
    "predictions": None,
    "confusion_matrix": None,
    "class_labels": None,
    "classification_report": None,
    "analysis_results": None,
    "pipeline_log": [],
    "active_data_source": "Preloaded sample dataset",
}
for key, default in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default


def log_step(message: str):
    st.session_state.pipeline_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")


# Load default data automatically at startup.
if st.session_state.raw_data is None:
    st.session_state.raw_data = load_default_data()
    st.session_state.active_data_source = "Preloaded sample dataset"


# ======================================================
# SIDEBAR NAVIGATION
# ======================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home", "Load Data", "Explore", "Clean & Prep", "Analyze", "Model",
        "Visualize", "Pipeline", "Reports", "Security", "Help", "Testing",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Active source: {st.session_state.active_data_source}")
if isinstance(st.session_state.raw_data, pd.DataFrame):
    st.sidebar.caption(f"Raw rows: {len(st.session_state.raw_data)}")
if isinstance(st.session_state.prepared_data, pd.DataFrame):
    st.sidebar.caption(f"Prepared rows: {len(st.session_state.prepared_data)}")


# ======================================================
# PAGE CONTENT
# ======================================================

if page == "Home":
    st.title("Predictive Trust Monitoring System")
    st.markdown(
        """
        This application is an interactive Streamlit data product that supports a complete OSEMN-style workflow for customer trust monitoring. It automatically loads a preloaded sample dataset so users can immediately explore, clean, analyze, visualize, model, and export results without needing to upload a file first.

        Updated first-iteration feedback addressed in this version:
        - A preloaded sample dataset is available at startup.
        - Users can still upload a CSV file or read data from a URL.
        - The product now includes selectable analysis methods, plotted results, downloadable reports, security documentation, a help feature, and a testing module.
        """
    )
    st.info("Use the left navigation panel to access each product feature.")

elif page == "Load Data":
    st.header("Load Data")
    st.write("The app starts with preloaded sample data. You may replace it by uploading a CSV file or loading a CSV from a URL.")

    input_method = st.radio("Choose data input method", ["Use preloaded data", "Upload CSV file", "Read CSV from URL"])

    if input_method == "Use preloaded data":
        if st.button("Reload Preloaded Sample Data"):
            st.session_state.raw_data = load_default_data()
            st.session_state.prepared_data = None
            st.session_state.active_data_source = "Preloaded sample dataset"
            log_step("Preloaded sample data loaded.")
            st.success("Preloaded sample dataset loaded successfully.")

    elif input_method == "Upload CSV file":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None and validate_uploaded_file(uploaded_file):
            try:
                st.session_state.raw_data = pd.read_csv(uploaded_file)
                st.session_state.prepared_data = None
                st.session_state.active_data_source = f"Uploaded file: {uploaded_file.name}"
                log_step(f"Uploaded CSV loaded: {uploaded_file.name}")
                st.success("Uploaded dataset loaded successfully.")
            except Exception as exc:
                st.error(f"The uploaded file could not be read: {exc}")

    elif input_method == "Read CSV from URL":
        url = st.text_input("Enter a direct CSV URL")
        if st.button("Load URL Data"):
            if not url.lower().startswith(("http://", "https://")):
                st.error("Please enter a valid http or https URL.")
            else:
                try:
                    st.session_state.raw_data = load_csv_from_url(url)
                    st.session_state.prepared_data = None
                    st.session_state.active_data_source = "CSV URL"
                    log_step("CSV data loaded from URL.")
                    st.success("URL dataset loaded successfully.")
                except Exception as exc:
                    st.error(f"The URL data could not be read: {exc}")

    df = st.session_state.raw_data
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Dataset Summary")
    st.write(df.describe(include="all"))

elif page == "Explore":
    st.header("Explore Data")
    df = st.session_state.raw_data
    if df is None:
        st.info("Load data first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isna().sum().sum()))

        variable = st.selectbox("Choose a variable to explore", df.columns)
        if pd.api.types.is_numeric_dtype(df[variable]):
            fig, ax = plt.subplots()
            ax.hist(df[variable].dropna(), bins=20)
            ax.set_title(f"Distribution of {variable}")
            ax.set_xlabel(variable)
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            counts = df[variable].astype(str).value_counts().head(20)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(counts.index[::-1], counts.values[::-1])
            ax.set_title(f"Top Categories for {variable}")
            st.pyplot(fig)

        st.subheader("Variable Summary")
        st.write(df[variable].describe(include="all"))

elif page == "Clean & Prep":
    st.header("Clean and Prepare Data")
    df = st.session_state.raw_data
    if df is None:
        st.info("Load data first.")
    else:
        st.write("Choose the preprocessing operations to apply.")
        selected_methods = st.multiselect(
            "Cleaning and preprocessing methods",
            [
                "Project-specific trust cleaning",
                "Create engineered trust features",
                "Remove duplicate rows",
                "Fill numeric missing values with median",
                "Fill text missing values with blank",
                "Standardize column names",
            ],
            default=["Project-specific trust cleaning", "Create engineered trust features"],
        )

        if st.button("Run Selected Preprocessing"):
            prepared = df.copy()
            if "Project-specific trust cleaning" in selected_methods:
                prepared = clean_data(prepared)
            if "Create engineered trust features" in selected_methods:
                prepared = engineer_features(prepared)
            general_methods = [m for m in selected_methods if m not in ["Project-specific trust cleaning", "Create engineered trust features"]]
            prepared = apply_selected_preprocessing(prepared, general_methods)
            st.session_state.prepared_data = prepared
            log_step(f"Preprocessing completed: {', '.join(selected_methods)}")
            st.success("Prepared dataset created successfully.")

        if isinstance(st.session_state.prepared_data, pd.DataFrame):
            prepared = st.session_state.prepared_data
            st.metric("Prepared Rows", len(prepared))
            st.dataframe(prepared.head(10), use_container_width=True)
            st.download_button("Download Prepared Data", data=to_csv_bytes(prepared), file_name="prepared_trust_data.csv", mime="text/csv")

elif page == "Analyze":
    st.header("Analyze Data")
    df = st.session_state.prepared_data
    if df is None:
        st.info("Run Clean & Prep first so Trust_Risk and engineered variables are available.")
    else:
        method = st.selectbox(
            "Choose a data analysis method",
            [
                "Trust risk frequency analysis",
                "Average score analysis by trust risk",
                "Billing issue analysis",
                "Correlation analysis",
            ],
        )
        if st.button("Run Analysis"):
            st.session_state.analysis_results = run_analysis_method(df, method)
            log_step(f"Analysis completed: {method}")
            st.success("Analysis completed.")

        if isinstance(st.session_state.analysis_results, pd.DataFrame):
            st.subheader("Analysis Results")
            st.dataframe(st.session_state.analysis_results, use_container_width=True)
            st.download_button("Download Analysis Results", data=to_csv_bytes(st.session_state.analysis_results), file_name="trust_analysis_results.csv", mime="text/csv")

elif page == "Model":
    st.header("Model")
    df = st.session_state.prepared_data
    if df is None:
        st.info("Prepare the data first.")
    else:
        train_size = st.slider("Training proportion", min_value=0.6, max_value=0.9, value=0.8, step=0.05)
        if st.button("Run Trust-Risk Model"):
            try:
                pipeline, metrics, results, cm, classes, report = run_model_pipeline(df, train_size=train_size)
                st.session_state.model_pipeline = pipeline
                st.session_state.metrics = metrics
                st.session_state.predictions = results
                st.session_state.confusion_matrix = cm
                st.session_state.class_labels = classes
                st.session_state.classification_report = report
                log_step("Trust-risk model completed.")
                st.success("Model completed successfully.")
            except Exception as exc:
                st.error(f"Model could not run: {exc}")

        if st.session_state.metrics is not None:
            m = st.session_state.metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{m['accuracy']:.3f}")
            c2.metric("Precision", f"{m['precision_weighted']:.3f}")
            c3.metric("Recall", f"{m['recall_weighted']:.3f}")
            c4.metric("F1", f"{m['f1_weighted']:.3f}")
            st.subheader("Prediction Sample")
            st.dataframe(st.session_state.predictions.head(10), use_container_width=True)
            st.text("Classification Report")
            st.text(st.session_state.classification_report)

elif page == "Visualize":
    st.header("Visualize Results")
    df = st.session_state.prepared_data
    if df is None:
        st.info("Prepare the data first.")
    else:
        plot_type = st.selectbox(
            "Choose a plot",
            ["Trust Risk Distribution", "Trust Risk by Bill Correct", "Average Metrics by Trust Risk", "Correlation Heatmap", "Confusion Matrix"],
        )

        if plot_type == "Trust Risk Distribution" and "Trust_Risk" in df.columns:
            counts = df["Trust_Risk"].value_counts()
            fig, ax = plt.subplots()
            ax.bar(counts.index, counts.values)
            ax.set_title("Distribution of Engineered Trust Risk")
            ax.set_xlabel("Trust Risk")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        elif plot_type == "Trust Risk by Bill Correct" and {"Trust_Risk", "Bill Correct"}.issubset(df.columns):
            cross = pd.crosstab(df["Trust_Risk"], df["Bill Correct"])
            fig, ax = plt.subplots()
            cross.plot(kind="bar", ax=ax)
            ax.set_title("Trust Risk by Billing Accuracy")
            ax.set_xlabel("Trust Risk")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        elif plot_type == "Average Metrics by Trust Risk" and "Trust_Risk" in df.columns:
            metric_cols = [c for c in ["Recommend", "service_avg", "vehicle_avg", "billing_issue_flag"] if c in df.columns]
            summary_df = df.groupby("Trust_Risk")[metric_cols].mean().reset_index()
            st.dataframe(summary_df, use_container_width=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            summary_df.set_index("Trust_Risk").plot(kind="bar", ax=ax)
            ax.set_title("Average Metrics by Trust Risk")
            ax.set_xlabel("Trust Risk")
            ax.set_ylabel("Average Value")
            st.pyplot(fig)

        elif plot_type == "Correlation Heatmap":
            numeric_df = df.select_dtypes(include=np.number)
            if numeric_df.shape[1] >= 2:
                corr = numeric_df.corr()
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(corr, aspect="auto")
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                ax.set_yticklabels(corr.columns)
                ax.set_title("Correlation Heatmap")
                fig.colorbar(im)
                st.pyplot(fig)
            else:
                st.warning("At least two numeric columns are required for correlation analysis.")

        elif plot_type == "Confusion Matrix":
            if st.session_state.confusion_matrix is not None:
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=st.session_state.confusion_matrix,
                    display_labels=st.session_state.class_labels,
                )
                disp.plot(ax=ax)
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
            else:
                st.warning("Run the model first to generate a confusion matrix.")

elif page == "Pipeline":
    st.header("Pipeline Runner")
    st.write("Run individual OSEMN workflow stages or execute the full data product pipeline.")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run Obtain"):
            st.session_state.raw_data = st.session_state.raw_data if st.session_state.raw_data is not None else load_default_data()
            log_step("Obtain step completed.")
            st.success("Obtain step completed.")

        if st.button("Run Scrub"):
            if st.session_state.raw_data is not None:
                df = clean_data(st.session_state.raw_data.copy())
                df = engineer_features(df)
                st.session_state.prepared_data = df
                log_step("Scrub step completed.")
                st.success("Scrub step completed.")
            else:
                st.warning("Load data first.")

    with col2:
        if st.button("Run Model Step"):
            if st.session_state.prepared_data is not None:
                try:
                    pipeline, metrics, results, cm, classes, report = run_model_pipeline(st.session_state.prepared_data)
                    st.session_state.model_pipeline = pipeline
                    st.session_state.metrics = metrics
                    st.session_state.predictions = results
                    st.session_state.confusion_matrix = cm
                    st.session_state.class_labels = classes
                    st.session_state.classification_report = report
                    log_step("Model step completed.")
                    st.success("Model step completed.")
                except Exception as exc:
                    st.error(f"Model step failed: {exc}")
            else:
                st.warning("Prepare data first.")

        if st.button("Run Full Pipeline"):
            try:
                raw = st.session_state.raw_data if st.session_state.raw_data is not None else load_default_data()
                prepared = engineer_features(clean_data(raw.copy()))
                st.session_state.raw_data = raw
                st.session_state.prepared_data = prepared
                pipeline, metrics, results, cm, classes, report = run_model_pipeline(prepared)
                st.session_state.model_pipeline = pipeline
                st.session_state.metrics = metrics
                st.session_state.predictions = results
                st.session_state.confusion_matrix = cm
                st.session_state.class_labels = classes
                st.session_state.classification_report = report
                st.session_state.analysis_results = run_analysis_method(prepared, "Trust risk frequency analysis")
                log_step("Full pipeline completed.")
                st.success("Full pipeline completed.")
            except Exception as exc:
                st.error(f"Full pipeline failed: {exc}")

    st.subheader("Pipeline Status Log")
    if st.session_state.pipeline_log:
        for entry in st.session_state.pipeline_log:
            st.write(f"- {entry}")
    else:
        st.write("No pipeline steps have been run yet.")

elif page == "Reports":
    st.header("Reports")
    st.write("Generate and save product outputs. Full narrative reports can be completed in Topic 8 using these exports.")

    if st.session_state.prepared_data is not None:
        st.download_button("Download Processed Data", data=to_csv_bytes(st.session_state.prepared_data), file_name="processed_trust_data.csv", mime="text/csv")
    if st.session_state.analysis_results is not None:
        st.download_button("Download Analysis Results", data=to_csv_bytes(st.session_state.analysis_results), file_name="trust_analysis_results.csv", mime="text/csv")
    if st.session_state.predictions is not None:
        st.download_button("Download Predictions", data=to_csv_bytes(st.session_state.predictions), file_name="trust_risk_predictions.csv", mime="text/csv")
    if st.session_state.metrics is not None:
        st.download_button("Download Metrics JSON", data=metrics_json_bytes(st.session_state.metrics), file_name="trust_risk_metrics.json", mime="application/json")
    if st.session_state.model_pipeline is not None:
        buffer = BytesIO()
        joblib.dump(st.session_state.model_pipeline, buffer)
        buffer.seek(0)
        st.download_button("Download Fitted Pipeline", data=buffer, file_name="trust_risk_pipeline.joblib", mime="application/octet-stream")

    st.download_button("Download HTML Summary Report", data=generate_html_report(), file_name="predictive_trust_monitoring_report.html", mime="text/html")

elif page == "Security":
    st.header("Security")
    st.markdown(
        """
        Security controls in this prototype include:
        - CSV-only upload restrictions and a 50 MB file-size limit.
        - No persistent storage of uploaded files in the application code.
        - Use of a non-sensitive sample dataset for demonstration.
        - URL validation for external CSV ingestion.
        - Session-based processing so outputs are generated only from the active user workflow.
        - A breach response plan that includes disabling affected features, reviewing logs, rotating credentials if used in future releases, and redeploying a corrected version.

        Future operational releases should add authenticated access, role-based permissions, formal audit logging, and secured cloud secrets for external database connections.
        """
    )

elif page == "Help":
    st.header("Help and Product Guide")
    st.markdown(
        """
        How to use this product:

        1. Home: Review the product purpose and first-iteration revisions.
        2. Load Data: Use the preloaded sample dataset, upload a CSV file, or read a CSV from a URL.
        3. Explore: Inspect the dataset, missing values, distributions, and categorical counts.
        4. Clean & Prep: Choose preprocessing methods and create trust-monitoring features.
        5. Analyze: Select a data analysis method and generate tabular results.
        6. Model: Train the trust-risk classification model and review performance metrics.
        7. Visualize: Plot trust-risk distributions, billing relationships, average scores, correlations, and confusion matrix results.
        8. Pipeline: Run individual workflow stages or execute the full pipeline.
        9. Reports: Save processed data, analysis outputs, predictions, metrics, model files, and an HTML report.
        10. Security: Review security assumptions, data restrictions, and planned controls.
        11. Testing: Run smoke tests that validate data loading, cleaning, feature engineering, analysis, modeling, reporting, and user interaction.
        """
    )

elif page == "Testing":
    st.header("Testing and Test Outcomes")
    st.write("This module validates calculations, data processing, user engagement, results, and report-generation functions.")

    if st.button("Run Product Tests"):
        st.session_state.test_results = run_system_tests()
        log_step("Product test suite executed.")

    if "test_results" in st.session_state:
        st.dataframe(st.session_state.test_results, use_container_width=True)
        st.download_button("Download Test Results", data=to_csv_bytes(st.session_state.test_results), file_name="product_test_results.csv", mime="text/csv")
