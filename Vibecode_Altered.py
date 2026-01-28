# ============================================================
# 💎 VIBE-CODE AI STUDIO ULTRA — by Team Innovex Coders
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import json
from datetime import datetime
import os
from openai import OpenAI, RateLimitError

# -----------------------------
# LLM CONFIG
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm_client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# AI MODE CONFIG
# -----------------------------
AI_MODE = "AGENT"  
# Options: "OFFLINE" (rule-based), "AGENT" (Level 3 AI)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Vibe-Code AI Studio Ultra", page_icon="💎", layout="wide")

# -----------------------------
# BUILT-IN DATASETS
# -----------------------------
DATASETS = {
    "Sales Data": pd.DataFrame({
        "Region": ["North", "South", "East", "West", "Central"],
        "Sales": [120, 150, 100, 180, 140],
        "Profit": [30, 40, 25, 60, 35],
        "Year": [2020, 2021, 2022, 2023, 2024]
    }),
    "Student Performance": pd.DataFrame({
        "Student": ["A", "B", "C", "D", "E", "F"],
        "Math": [85, 90, 75, 60, 95, 70],
        "Science": [88, 76, 85, 60, 90, 80],
        "English": [78, 89, 70, 68, 92, 77]
    }),
    "Climate Data": pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "Temperature": [20, 22, 25, 28, 30, 32, 33, 31, 29, 26, 23, 21],
        "Rainfall": [100, 80, 60, 40, 20, 10, 15, 30, 50, 70, 90, 110]
    }),
    "Website Analytics": pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "Visitors": [1200, 1500, 1700, 1600, 2000, 2500],
        "Conversions": [80, 90, 100, 95, 120, 140],
        "Bounce Rate": [60, 55, 52, 50, 48, 45]
    })
}

# -----------------------------
# DATASET REGISTRY (COMPARE ENGINE)
# -----------------------------
def get_dataset_registry():
    registry = {}

    # Built-in datasets
    for name, df in DATASETS.items():
        registry[name] = df.copy()

    return registry

# -----------------------------
# INTENT & THEMES
# -----------------------------
INTENT_KEYWORDS = {
    "Comparison": ["compare", "versus", "across", "rank"],
    "Trend": ["trend", "increase", "decrease", "growth", "year", "time"],
    "Composition": ["share", "portion", "contribution", "part"],
    "Distribution": ["distribution", "spread", "range", "variance"],
    "Relationship": ["relationship", "correlation", "link", "impact"]
}

THEMES = {"Dark": "plotly_dark", "Light": "plotly_white", "Minimal": "simple_white"}

# -----------------------------
# INTELLIGENCE FUNCTIONS
# -----------------------------
def detect_intent(goal: str) -> str:
    goal = goal.lower()
    for intent, keys in INTENT_KEYWORDS.items():
        if any(k in goal for k in keys):
            return intent
    return "Comparison"

def recommend_chart(intent: str):
    charts = {"Comparison": "Bar", "Trend": "Line", "Composition": "Pie",
              "Distribution": "Histogram", "Relationship": "Scatter"}
    return charts.get(intent, "Bar")

def ai_summary(df, x, y, intent):
    """Enhanced AI Summary - multiple reasoning layers"""
    try:
        lines = []
        # General shape
        lines.append(f"Dataset contains **{df.shape[0]} records** and **{df.shape[1]} columns**.")
        # Comparison
        if intent == "Comparison":
            top = df.loc[df[y].idxmax(), x]
            bottom = df.loc[df[y].idxmin(), x]
            gap = round(df[y].max() - df[y].min(), 2)
            lines.append(f"🔹 The highest {y.lower()} is in **{top}**, and the lowest in **{bottom}** (gap = {gap}).")
        # Trend
        if intent == "Trend" and "Year" in df.columns:
            yearly = df.groupby("Year")[y].sum()
            change = round(((yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0]) * 100, 1)
            direction = "increased 📈" if change > 0 else "decreased 📉"
            lines.append(f"Over the years, **{y}** has {direction} by {abs(change)}%.")
        # Correlation check
        if len(df.select_dtypes(include=np.number).columns) > 1:
            corr = df.select_dtypes(include=np.number).corr()
            high_corr = corr.stack().sort_values(ascending=False)
            top_corr = high_corr[high_corr < 1].head(1)
            if not top_corr.empty:
                a, b = top_corr.index[0]
                lines.append(f"📈 The strongest numeric correlation is between **{a}** and **{b}** ({round(top_corr.values[0],2)}).")
        # Outlier hint
        if df[y].std() > df[y].mean() * 0.5:
            lines.append(f"⚠️ Data shows high variability; consider using a box plot to explore outliers.")
        return "\n".join(lines)
    except Exception:
        return "Unable to analyze this dataset deeply."

def create_chart(df, chart_type, x, y, theme):
    title = f"{chart_type} of {y} vs {x}"
    try:
        if chart_type == "Bar":
            fig = px.bar(df, x=x, y=y, color=x, title=title, template=theme)
        elif chart_type == "Line":
            fig = px.line(df, x=x, y=y, markers=True, title=title, template=theme)
        elif chart_type == "Pie":
            fig = px.pie(df, names=x, values=y, hole=0.3, title=f"Composition of {y} by {x}", template=theme)
        elif chart_type == "Area":
            fig = px.area(df, x=x, y=y, title=title, template=theme)
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x, y=y, color=x, size=y, title=title, template=theme)
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=y, nbins=15, title=f"Distribution of {y}", template=theme)
        else:
            fig = px.bar(df, x=x, y=y, title=title, template=theme)
        return fig
    except Exception:
        return px.bar(title="Error rendering chart.")

# -----------------------------
# AI AUTO-QUESTIONS ENGINE
# -----------------------------
def generate_auto_questions(df):
    questions = []

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    if num_cols and cat_cols:
        questions.append(
            f"Which {cat_cols[0]} has the highest {num_cols[0]}?"
        )

    if "year" in [c.lower() for c in df.columns]:
        questions.append(
            f"How has {num_cols[0]} changed over time?"
        )

    if len(num_cols) > 1:
        questions.append(
            f"Is there a relationship between {num_cols[0]} and {num_cols[1]}?"
        )

    if num_cols:
        questions.append(
            f"Which metric shows the most variability?"
        )

    return questions

# -----------------------------
# AI LEARNING MODE EXPLANATION
# -----------------------------
def explain_insight(text, mode):
    if mode == "Beginner":
        return f"Simply put: {text.lower()}."

    if mode == "Intermediate":
        return f"{text} This suggests a meaningful pattern worth observing."

    return f"{text} From an analytical perspective, this insight can be used for strategic decision-making."

# -----------------------------
# AI DATA CONTEXT BUILDER
# -----------------------------
def build_ai_context(df):
    context = {}

    context["shape"] = {
        "rows": df.shape[0],
        "columns": df.shape[1]
    }

    context["columns"] = {
        "numeric": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical": df.select_dtypes(exclude=["number"]).columns.tolist()
    }

    stats = {}
    for col in context["columns"]["numeric"]:
        stats[col] = {
            "mean": float(round(df[col].mean(), 2)),
            "std": float(round(df[col].std(), 2)),
            "min": float(round(df[col].min(), 2)),
            "max": float(round(df[col].max(), 2))
        }

    context["statistics"] = stats

    if len(context["columns"]["numeric"]) > 1:
        corr = df[context["columns"]["numeric"]].corr().round(2)
        context["correlations"] = corr.to_dict()
    else:
        context["correlations"] = {}

    return context



# -----------------------------
# PSEUDO-LLM ANALYST ENGINE (LEVEL 3 OFFLINE)
# -----------------------------
def pseudo_llm_analyst(context):
    insights = []
    questions = []
    warnings = []

    stats = context.get("statistics", {})
    numeric_cols = context.get("columns", {}).get("numeric", [])

    if not numeric_cols:
        return {
            "insights": ["No numeric data available for analysis."],
            "questions": [],
            "warnings": []
        }

    # ---- Importance Scoring ----
    scored_metrics = []
    for col in numeric_cols:
        s = stats[col]
        variability_score = s["std"] / (s["mean"] + 1e-6)
        scored_metrics.append((col, variability_score, s))

    scored_metrics.sort(key=lambda x: x[1], reverse=True)

    # ---- Primary Insight ----
    top_col, top_score, top_stats = scored_metrics[0]

    conf = calculate_confidence(
        top_stats["mean"], top_stats["std"], context["shape"]["rows"]
    )

    insight_text = speak_like_ai(
        f'"{top_col}" stands out as the most influential metric due to its relative variability'
    )

    insights.append(
        f"{insight_text} (Confidence: {conf}%)"
    )
 



    # ---- Risk & Stability Analysis ----
    for col, score, s in scored_metrics:
        if score > 0.6:
            conf = calculate_confidence(
                s["mean"], s["std"], context["shape"]["rows"]
            )

            warning_text = speak_like_ai(
                f'"{col}" demonstrates unstable behavior due to unusually high volatility'
            )

            warnings.append(
                f"{warning_text} (Confidence: {conf}%)"
            )



    # ---- Comparative Reasoning ----
    if len(scored_metrics) > 1:
        a, _, _ = scored_metrics[0]
        b, _, _ = scored_metrics[1]
        questions.append(
            f"How does '{a}' influence or relate to '{b}'?"
        )

    # ---- Fallback Question ----
    if not questions:
        questions.append(
            "What strategic action should be taken based on the strongest observed pattern?"
        )

    return {
        "insights": insights,
        "questions": questions,
        "warnings": warnings
    }


# -----------------------------
# AI REASONING LANGUAGE ENGINE
# -----------------------------
import random

REASONING_STARTERS = [
    "Based on the observed data,",
    "From the dataset patterns,",
    "An initial analysis suggests that",
    "Looking closely at the numbers,"
]

REASONING_CAUSES = [
    "this is likely driven by",
    "this may be explained by",
    "a possible reason is",
    "this seems to occur due to"
]

CONFIDENCE_PHRASES = [
    "with high confidence",
    "with moderate certainty",
    "based on current evidence",
    "though further validation may help"
]

def speak_like_ai(statement):
    return (
        f"{random.choice(REASONING_STARTERS)} "
        f"{statement}, "
        f"{random.choice(CONFIDENCE_PHRASES)}."
    )

# -----------------------------
# CONFIDENCE SCORING ENGINE
# -----------------------------
def calculate_confidence(mean, std, rows):
    """
    Returns a confidence score (0–100)
    Higher rows + lower variability = higher confidence
    """
    if mean == 0:
        return 50

    variability_penalty = min((std / mean) * 40, 40)
    size_bonus = min(np.log1p(rows) * 10, 30)

    confidence = 70 + size_bonus - variability_penalty
    return int(max(30, min(confidence, 95)))

# -----------------------------
# SIMPLE FORECAST ENGINE
# -----------------------------
def forecast_series(values, steps=3):
    """
    Simple trend-based forecasting using linear regression logic
    """
    if len(values) < 3:
        return []

    x = np.arange(len(values))
    y = np.array(values)

    # Fit line y = mx + c
    m, c = np.polyfit(x, y, 1)

    future_x = np.arange(len(values), len(values) + steps)
    forecast = m * future_x + c

    return [round(v, 2) for v in forecast]

# -----------------------------
# AI FORECAST ANALYZER
# -----------------------------
def ai_forecast(df, context, steps=3):
    forecasts = []

    numeric_cols = context["columns"]["numeric"]

    for col in numeric_cols:
        values = df[col].dropna().values

        if len(values) < 5:
            continue

        trend = np.polyfit(range(len(values)), values, 1)[0]
        last = values[-1]

        future = [round(last + trend * (i + 1), 2) for i in range(steps)]

        forecasts.append(
            f"{col} is expected to continue its trend, with next values around {future}"
        )

    return forecasts

# -----------------------------
# AI REPORT DATA AGGREGATOR
# -----------------------------
def build_ai_report_data(
    df,
    dataset_name,
    ai_context,
    agent_output,
    forecast_insights
):
    report = {}

    # --- Meta Info ---
    report["meta"] = {
        "dataset": dataset_name,
        "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rows": ai_context["shape"]["rows"],
        "columns": ai_context["shape"]["columns"]
    }

    # --- KPI Summary ---
    report["kpis"] = []
    for col, stats in ai_context["statistics"].items():
        report["kpis"].append({
            "metric": col,
            "mean": stats["mean"],
            "min": stats["min"],
            "max": stats["max"],
            "std": stats["std"]
        })

    # --- AI Insights ---
    report["insights"] = agent_output.get("insights", [])

    # --- AI Warnings ---
    report["warnings"] = agent_output.get("warnings", [])

    # --- AI Questions ---
    report["questions"] = agent_output.get("questions", [])

    # --- Forecast Section ---
    report["forecast"] = forecast_insights

    return report

# -----------------------------
# AI REPORT WRITER
# -----------------------------
def write_ai_report(report_data):
    """
    Generates a full analytical report as text.
    Uses LLM if available, otherwise smart offline narration.
    """

    # ---------- OFFLINE FALLBACK ----------
    if not OPENAI_API_KEY or AI_MODE != "AGENT":
        lines = []

        lines.append("### 📌 Executive Summary")
        lines.append(
            f"The dataset '{report_data['meta']['dataset']}' contains "
            f"{report_data['meta']['rows']} rows and "
            f"{report_data['meta']['columns']} columns."
        )

        lines.append("\n### 📊 Key Insights")
        for i in report_data["insights"]:
            lines.append(f"- {i}")

        if report_data["warnings"]:
            lines.append("\n### ⚠️ Risks & Anomalies")
            for w in report_data["warnings"]:
                lines.append(f"- {w}")

        if report_data["forecast"]:
            lines.append("\n### 🔮 Forecast Outlook")
            for f in report_data["forecast"]:
                lines.append(f"- {f}")

        lines.append("\n### 🧠 Strategic Recommendations")
        lines.append(
            "Focus on stabilizing high-variability metrics, "
            "monitor trends closely, and validate forecasts with additional data."
        )

        return "\n".join(lines)

    # ---------- LLM MODE ----------
    prompt = f"""
You are a senior business data analyst.

Using the following structured data, write a professional analytical report
with clear sections:

1. Executive Summary
2. Key Insights
3. Risks & Anomalies
4. Forecast Outlook
5. Strategic Recommendations

Write concisely, clearly, and in a confident analytical tone.

DATA:
{json.dumps(report_data, indent=2)}
"""

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You write professional analytical reports."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        return response.choices[0].message.content

    except Exception:
        return "⚠️ AI report generation failed. Please try again."

# -----------------------------
# AI PROMPT BUILDER
# -----------------------------

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def build_agent_prompt(context):
    safe_context = make_json_safe(context)

    return f"""
You are a senior data analyst.

You are given a structured summary of a dataset.
Your task is to:
1. Identify key insights
2. Detect risks or anomalies
3. Suggest meaningful follow-up questions

Respond STRICTLY in valid JSON with this structure:
{{
  "insights": [string],
  "questions": [string],
  "warnings": [string]
}}

DATA SUMMARY:
{json.dumps(safe_context, indent=2)}
"""


# -----------------------------
# AI ANALYST AGENT (LLM-POWERED + SAFE FALLBACK)
# -----------------------------
def ai_analyst_agent(context):
    if not OPENAI_API_KEY:
        return {
            "insights": ["LLM unavailable. Running in offline analysis mode."],
            "questions": [],
            "warnings": []
        }

    prompt = build_agent_prompt(context)

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a careful and analytical AI."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except RateLimitError:
        return pseudo_llm_analyst(context)

    except Exception as e:
        return {
            "insights": ["AI response could not be parsed safely."],
            "questions": [],
            "warnings": [str(e)]
        }


# -----------------------------
# SMART AUTO-CLEAN (ENHANCED)
# -----------------------------
def smart_auto_clean(df):
    report = []
    cleaned = df.copy()

    # Missing values
    nulls = cleaned.isnull().sum()
    for col, count in nulls.items():
        if count > 0:
            if cleaned[col].dtype == "object":
                cleaned[col].fillna("Unknown", inplace=True)
            else:
                cleaned[col].fillna(cleaned[col].median(), inplace=True)
            report.append(f"Filled {count} missing values in '{col}'")

    # Duplicates
    dup_count = cleaned.duplicated().sum()
    if dup_count > 0:
        cleaned.drop_duplicates(inplace=True)
        report.append(f"Removed {dup_count} duplicate rows")

    # Data type correction
    for col in cleaned.columns:
        if cleaned[col].dtype == "object":
            try:
                cleaned[col] = pd.to_numeric(cleaned[col])
                report.append(f"Converted '{col}' to numeric")
            except:
                pass

    if not report:
        report.append("Dataset already clean — no changes required")

    return cleaned, report

# -----------------------------
# ENHANCED KPI DETAILS ENGINE
# -----------------------------
def generate_kpi_details(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns
    kpi_details = []

    for col in numeric_cols:
        kpi_details.append({
            "name": col.replace("_", " ").title(),
            "mean": round(df[col].mean(), 2),
            "max": round(df[col].max(), 2),
            "min": round(df[col].min(), 2),
            "std": round(df[col].std(), 2)
        })

    return kpi_details

# -----------------------------
# AI DATA CLEANING & KPI ENGINE
# -----------------------------
def auto_clean_data(df):
    report = []
    cleaned_df = df.copy()

    # Handle missing values
    for col in cleaned_df.columns:
        if cleaned_df[col].isnull().sum() > 0:
            if cleaned_df[col].dtype == "object":
                cleaned_df[col].fillna("Unknown", inplace=True)
            else:
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            report.append(f"Filled missing values in '{col}'")

    # Remove duplicates
    before = cleaned_df.shape[0]
    cleaned_df.drop_duplicates(inplace=True)
    after = cleaned_df.shape[0]
    if before != after:
        report.append(f"Removed {before - after} duplicate rows")

    # Standardize column names
    cleaned_df.columns = [c.strip().replace(" ", "_").lower() for c in cleaned_df.columns]
    report.append("Standardized column names")

    return cleaned_df, report


def detect_kpis(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    kpis = []

    if numeric_cols:
        kpis.append(("Primary KPI", numeric_cols[0]))
        if len(numeric_cols) > 1:
            kpis.append(("Secondary KPI", numeric_cols[1]))
        if len(numeric_cols) > 2:
            kpis.append(("Efficiency KPI", numeric_cols[-1]))

    return kpis

def generate_compare_report_text(
    dataset_A_name,
    dataset_B_name,
    comparison_payload
):
    lines = []

    lines.append("## 📝 AI Comparison Report\n")
    lines.append(f"**Dataset A:** {dataset_A_name}")
    lines.append(f"**Dataset B:** {dataset_B_name}\n")

    lines.append("### 📊 Metric-by-Metric Comparison")

    score_A = 0
    score_B = 0

    for item in comparison_payload:
        metric = item["metric"]
        A = item["dataset_A"]
        B = item["dataset_B"]

        diff = round(A["mean"] - B["mean"], 2)

        if diff > 0:
            winner = dataset_A_name
            score_A += 1
            direction = "higher"
        elif diff < 0:
            winner = dataset_B_name
            score_B += 1
            direction = "lower"
        else:
            winner = "Both"
            direction = "similar"

        confidence = calculate_confidence(
            mean=max(A["mean"], B["mean"], 1),
            std=abs(diff) + 1e-6,
            rows=10
        )

        lines.append(
            f"- **{metric}**: {winner} shows {direction} average values "
            f"(Δ ≈ {abs(diff)}, confidence ≈ {confidence}%)"
        )

    lines.append("\n### 🏆 Final Verdict")

    if score_A > score_B:
        verdict = dataset_A_name
    elif score_B > score_A:
        verdict = dataset_B_name
    else:
        verdict = "Neither — both datasets perform similarly"

    lines.append(f"**Overall Winner:** {verdict}")

    lines.append(
        "\n### 🧠 Interpretation\n"
        "The comparison highlights consistent performance differences across "
        "shared metrics. Metrics with higher confidence should be prioritized "
        "for strategic decision-making."
    )

    return "\n".join(lines)

# -----------------------------
# AI REPORT VIEWER
# -----------------------------
def display_ai_report(report_text):
    st.markdown("## 📄 AI Analytical Report")

    sections = report_text.split("### ")
    for sec in sections:
        if not sec.strip():
            continue

        title, *body = sec.split("\n")
        content = "\n".join(body)

        with st.expander(title.strip()):
            st.markdown(content)

    # Download option
    st.download_button(
        label="⬇️ Download Report (TXT)",
        data=report_text,
        file_name="ai_analysis_report.txt",
        mime="text/plain"
    )


# -----------------------------
# SIDEBAR
# -----------------------------
tab_main, tab_compare = st.tabs(["📊 Analyze", "🔁 Compare"])

st.sidebar.title("⚙️ Controls")

examples = [
    "Compare regional sales and profits",
    "Show yearly sales growth trend",
    "Analyze composition of revenue by region",
    "Understand distribution of profit margins",
    "Check relationship between expenses and sales"
]

goal = st.sidebar.text_area("🧠 Your Data Goal", placeholder=np.random.choice(examples))
dataset_name = st.sidebar.selectbox("📂 Choose Dataset", list(DATASETS.keys()))
theme_choice = st.sidebar.radio("🎨 Theme", list(THEMES.keys()), horizontal=True)
chart_choice = st.sidebar.selectbox("📈 Chart Type", ["Auto", "Bar", "Line", "Pie", "Area", "Scatter", "Histogram"])
uploaded_file = st.sidebar.file_uploader("⬆️ Upload CSV/Excel", type=["csv", "xlsx"])
st.sidebar.markdown("### 🤖 AI Assistance")

st.sidebar.markdown("### 🎓 AI Learning Mode")
learning_mode = st.sidebar.radio(
    "Explanation level",
    ["Beginner", "Intermediate", "Expert"],
    horizontal=True
)

auto_clean = st.sidebar.checkbox("🧹 Auto-clean my data", value=False)
st.sidebar.caption("💡 Made with ❤️ by Team Innovex Coders")

with tab_main:
    # -----------------------------
    # LOAD DATA
    # -----------------------------
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    else:
        df = DATASETS[dataset_name]


    # -----------------------------
    # SMART AUTO-CLEAN EXECUTION
    # -----------------------------
    if auto_clean:
        df, smart_report = smart_auto_clean(df)
        st.markdown("## 🧠 Smart Auto-Clean Insights")
        for r in smart_report:
            st.write("•", r)

    # -----------------------------
    # AI AGENT EXECUTION (LEVEL 3)
    # -----------------------------
    with st.spinner("🧠 AI is analyzing your data..."):
        ai_context = build_ai_context(df)

        if AI_MODE == "AGENT" and OPENAI_API_KEY:
            agent_output = ai_analyst_agent(ai_context)
        else:
            agent_output = pseudo_llm_analyst(ai_context)


    # -----------------------------
    # BASIC STATS SECTION
    # -----------------------------
    st.markdown("## 📊 Data Overview")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    st.dataframe(df.head())

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    st.markdown("### 📈 Dataset Statistics")
    st.dataframe(df.describe())

    # -----------------------------
    # AUTO KPI DETECTION
    # -----------------------------
    st.markdown("## 🎯 Auto-Detected KPIs")
    kpis = detect_kpis(df)

    if kpis:
        cols = st.columns(len(kpis))
        for col, (label, metric) in zip(cols, kpis):
            col.metric(label, metric)
    else:
        st.info("No numeric KPIs detected.")
    st.caption("💡 KPIs are automatically selected based on numeric importance and data patterns.")	

    # -----------------------------
    # SOFT KPI COLOR PALETTE
    # -----------------------------
    SOFT_COLORS = [
        "#6FCF97",  # soft green
        "#56CCF2",  # soft blue
        "#BB6BD9",  # soft purple
        "#F2C94C",  # soft yellow
        "#2D9CDB",  # calm blue
        "#EB5757",  # soft red
        "#27AE60",  # muted green
        "#9B51E0",  # muted violet
    ]

    # -----------------------------
    # SMART KPI COLOR ENGINE
    # -----------------------------
    def get_kpi_color(kpi_name: str) -> str:
        name = kpi_name.lower()

        if any(x in name for x in ["sales", "profit", "revenue", "income"]):
            return "#4CAF50"   # green → money / growth

        if any(x in name for x in ["visitor", "conversion", "count", "users"]):
            return "#2196F3"   # blue → volume / traffic

        if any(x in name for x in ["rate", "ratio", "percent", "efficiency"]):
            return "#FF9800"   # orange → rates / efficiency

        if any(x in name for x in ["temperature", "temp", "rain", "climate"]):
            return "#00BCD4"   # cyan → environment

        if any(x in name for x in ["year", "month", "time", "date"]):
            return "#9C27B0"   # purple → time

        return "#9E9E9E"       # neutral fallback


    # -----------------------------
    # INTERACTIVE KPI DETAILS
    # -----------------------------
    st.markdown("## 📌 KPI Insights")

    kpi_details = generate_kpi_details(df)
    cols = st.columns(len(kpi_details))
        
        
    for col_ui, kpi in zip(cols, kpi_details):
        with col_ui:
            color = SOFT_COLORS[kpi_details.index(kpi) % len(SOFT_COLORS)]
            confidence = "High" if kpi["std"] < kpi["mean"] * 0.5 else "Medium"
            st.markdown(
                f"""
                <div style="
                    background-color: rgba(255,255,255,0.03);
                    padding: 20px;
                    border-radius: 16px;
                    text-align: center;
                    box-shadow: 0 8px 20px rgba(0,0,0,0.35);
                    border-left: 4px solid {color};	
                ">
                    <h4 style="color:{color}; margin-bottom:12px;">
                        {kpi['name']}
                    </h4>
                    <p><b>Avg</b>: {kpi['mean']}</p>
                    <p><b>Max</b>: {kpi['max']}</p>
                    <p><b>Min</b>: {kpi['min']}</p>
                    <p><b>Volatility</b>: {kpi['std']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.caption(f"Confidence: {confidence}")	

    # -----------------------------
    # KPI AI EXPLANATION (COLLAPSIBLE)
    # -----------------------------
    with st.expander("🧠 How to interpret these KPIs"):
        for kpi in kpi_details:
            st.markdown(
                f"""
                **{kpi['name']}**  
                • Typical value is around **{kpi['mean']}**  
                • Range varies from **{kpi['min']} to {kpi['max']}**  
                • Variability score: **{kpi['std']}** (higher = more fluctuation)
                """
            )

    # -----------------------------
    # AI-SUGGESTED QUESTIONS
    # -----------------------------
    st.markdown("## ❓ AI-Suggested Questions")

    if AI_MODE == "AGENT" and agent_output:
        for q in agent_output["questions"]:
            st.markdown(f"• {q}")
    else:
        auto_questions = generate_auto_questions(df)
        for q in auto_questions:
            st.markdown(f"• {q}")

    # -----------------------------
    # AI INTELLIGENCE ENGINE
    # -----------------------------
    intent = detect_intent(goal)
    auto_chart = recommend_chart(intent)
    final_chart = chart_choice if chart_choice != "Auto" else auto_chart
    theme = THEMES[theme_choice]

    x = st.selectbox("X-Axis", cat_cols + num_cols)
    y = st.selectbox("Y-Axis", num_cols)

    # -----------------------------
    # MAIN VISUALIZATION
    # -----------------------------
    st.markdown(f"## 🎯 Visualization — {final_chart} Chart ({intent} Intent)")
    fig = create_chart(df, final_chart, x, y, theme)
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # AI DATA INSIGHTS (AGENT-DRIVEN)
    # -----------------------------
    st.markdown("## 🧠 AI Data Insights")

    if AI_MODE == "AGENT" and agent_output:
        for insight in agent_output["insights"]:
            st.info(insight)
    else:
        summary = ai_summary(df, x, y, intent)
        explained_summary = explain_insight(summary, learning_mode)
        st.info(explained_summary)

    # -----------------------------
    # AI FORECASTING
    # -----------------------------
    st.markdown("## 🔮 AI Forecast")

    forecast_insights = ai_forecast(df, ai_context, steps=3)

    if forecast_insights:
        for f in forecast_insights:
            st.success(f)
    else:
        st.info("No suitable numeric trends detected for forecasting.")

    # -----------------------------
    # AI REPORT GENERATION
    # -----------------------------
    st.markdown("## 🧾 AI Executive Report")

    report_data = build_ai_report_data(
              df=df,
              dataset_name=dataset_name,
              ai_context=ai_context,
              agent_output=agent_output,
              forecast_insights=forecast_insights
          )
    

    final_report = write_ai_report(report_data)

    display_ai_report(final_report)

    
    # -----------------------------
    # AI WARNINGS & RISKS
    # -----------------------------
    if AI_MODE == "AGENT" and agent_output and agent_output["warnings"]:
        st.markdown("## ⚠️ AI-Detected Risks")
        for w in agent_output["warnings"]:
            st.warning(w)

    # -----------------------------
    # CORRELATION HEATMAP (BONUS)
    # -----------------------------
    if len(num_cols) > 1:
        st.markdown("## 🔥 Correlation Heatmap")
        corr_fig = px.imshow(df[num_cols].corr(), text_auto=True, color_continuous_scale="RdBu_r", title="Numeric Correlations")
        st.plotly_chart(corr_fig, use_container_width=True)

    # -----------------------------
    # SMART RECOMMENDATIONS
    # -----------------------------
    st.markdown("## 💡 Smart Recommendations")
    suggestions = {
        "Trend": "Try a line or area chart to visualize changes over time.",
        "Comparison": "Bar or column charts highlight category differences effectively.",
        "Composition": "Use pie or stacked bar charts to show parts of a whole.",
        "Distribution": "Histograms or box plots reveal spread and outliers.",
        "Relationship": "Scatter plots uncover variable interactions."
    }
    st.success(suggestions.get(intent, "Use a clear chart and consistent color scheme for best storytelling."))

    # -----------------------------
    # FOOTER
    # -----------------------------
    st.markdown("---")
    st.markdown("<center>✨ Made with ❤️ by <b>Team Innovex Coders</b> | Powered by Streamlit + Plotly</center>", unsafe_allow_html=True)


# ============================================================
# 🔁 COMPARE TAB — STEP 2 (STRUCTURE ONLY)
# ============================================================

with tab_compare:

    st.markdown("## 🧾 Compare Report")

    st.markdown("## 🔁 Dataset Comparison Studio")
    st.caption("Compare multiple datasets, charts, and AI-driven insights side by side.")

    # -----------------------------
    # DATASET SELECTION
    # -----------------------------
    st.markdown("### 📂 Select Datasets to Compare")

    available_sources = list(DATASETS.keys())
    selected_datasets = st.multiselect(
        "Choose one or more datasets",
        options=available_sources,
        default=available_sources[:2]
    )

    if len(selected_datasets) < 2:
        st.info("Select at least two datasets to enable comparison.")
        st.stop()

    # -----------------------------
    # COMPARISON CONFIG
    # -----------------------------
    st.markdown("### ⚙️ Comparison Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        compare_metric = st.selectbox(
            "Metric to Compare",
            options=["Auto"]
        )

    with col2:
        compare_chart = st.selectbox(
            "Chart Type",
            ["Auto", "Bar", "Line", "Box", "Scatter"]
        )

    with col3:
        compare_mode = st.radio(
            "Comparison Mode",
            ["Side-by-Side", "Overlay"],
            horizontal=True
        )

    # -----------------------------
    # PLACEHOLDERS (FOR NEXT STEPS)
    # -----------------------------
    st.markdown("### 📊 Comparison View")
    st.info("Charts will appear here (Step 3)")

    st.markdown("### 🧠 AI Comparison Insights")
    st.info("AI-generated explanations will appear here (Step 4)")

    
    st.markdown("## 🔁 Dataset Comparison Engine")

    st.markdown("### 📂 Select Datasets to Compare")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Dataset A")
        source_A = st.radio(
            "Source A",
            ["Built-in Dataset", "Upload File"],
            key="source_A"
        )

        if source_A == "Built-in Dataset":
            dataset_A_name = st.selectbox(
                "Choose Dataset A",
                list(DATASETS.keys()),
                key="dataset_A_builtin"
            )
            df_A = DATASETS[dataset_A_name]
        else:
            file_A = st.file_uploader(
                "Upload Dataset A (CSV / Excel)",
                type=["csv", "xlsx"],
                key="dataset_A_upload"
            )
            df_A = None
            if file_A:
                df_A = (
                    pd.read_csv(file_A)
                    if file_A.name.endswith(".csv")
                    else pd.read_excel(file_A)
                )

    with colB:
        st.subheader("Dataset B")
        source_B = st.radio(
            "Source B",
            ["Built-in Dataset", "Upload File"],
            key="source_B"
        )

        if source_B == "Built-in Dataset":
            dataset_B_name = st.selectbox(
                "Choose Dataset B",
                list(DATASETS.keys()),
                key="dataset_B_builtin"
            )
            df_B = DATASETS[dataset_B_name]
        else:
            file_B = st.file_uploader(
                "Upload Dataset B (CSV / Excel)",
                type=["csv", "xlsx"],
                key="dataset_B_upload"
            )
            df_B = None
            if file_B:
                df_B = (
                    pd.read_csv(file_B)
                    if file_B.name.endswith(".csv")
                    else pd.read_excel(file_B)
                )

    # -----------------------------
    # PREVIEW SECTION
    # -----------------------------
    if df_A is not None and df_B is not None:
        st.markdown("### 👀 Dataset Preview")

        pcol1, pcol2 = st.columns(2)

        with pcol1:
            st.markdown("**Dataset A Preview**")
            st.write(f"Rows: {df_A.shape[0]} | Columns: {df_A.shape[1]}")
            st.dataframe(df_A.head())

        with pcol2:
            st.markdown("**Dataset B Preview**")
            st.write(f"Rows: {df_B.shape[0]} | Columns: {df_B.shape[1]}")
            st.dataframe(df_B.head())

    # -----------------------------
    # STEP 4: METRIC ALIGNMENT ENGINE
    # -----------------------------
    st.markdown("### 🔗 Metric Alignment")

    def get_numeric_columns(df):
        return df.select_dtypes(include=["number"]).columns.tolist()

    if df_A is not None and df_B is not None:
        numeric_A = get_numeric_columns(df_A)
        numeric_B = get_numeric_columns(df_B)

        common_metrics = sorted(list(set(numeric_A).intersection(set(numeric_B))))

        if not numeric_A or not numeric_B:
            st.warning("One of the datasets does not contain numeric columns.")
        else:
            st.markdown("#### 📊 Select Metrics to Compare")

            if common_metrics:
                metric_mode = st.radio(
                    "Metric selection mode",
                    ["Auto (Recommended)", "Manual"],
                    horizontal=True
                )

                if metric_mode == "Auto (Recommended)":
                    selected_metrics = common_metrics[:3]  # smart default
                    st.success(
                        f"Auto-selected common metrics: {', '.join(selected_metrics)}"
                    )
                else:
                    selected_metrics = st.multiselect(
                        "Choose common numeric metrics",
                        common_metrics,
                        default=common_metrics[:1]
                    )
            else:
                st.warning(
                    "No common numeric metrics found. "
                    "You can manually select separate metrics."
                )

                metric_A = st.selectbox(
                    "Metric from Dataset A",
                    numeric_A,
                    key="metric_A_manual"
                )
                metric_B = st.selectbox(
                    "Metric from Dataset B",
                    numeric_B,
                    key="metric_B_manual"
                )

                selected_metrics = [(metric_A, metric_B)]

        # Store for next steps
        st.session_state["compare_metrics"] = selected_metrics

    # -----------------------------
    # STEP 5: MULTI-CHART COMPARISON ENGINE
    # -----------------------------
    st.markdown("### 📈 Visual Comparison")

    CHART_OPTIONS = ["Bar", "Line", "Area", "Scatter"]

    selected_metrics = st.session_state.get("compare_metrics", [])

    if not selected_metrics:
        st.info("Select metrics above to enable comparison.")
    else:
        chart_type = st.selectbox(
            "Choose chart type for comparison",
            CHART_OPTIONS,
            index=0
        )

        cols = st.columns(len(selected_metrics))

        comparison_payload = []  # for AI reasoning later

        for i, metric in enumerate(selected_metrics):
            with cols[i]:

                # Handle auto vs manual metrics
                if isinstance(metric, tuple):
                    metric_A, metric_B = metric
                else:
                    metric_A = metric
                    metric_B = metric

                st.markdown(f"#### 🔹 {metric_A}")

                try:
                    # Dataset A
                    fig_A = create_chart(
                        df_A,
                        chart_type,
                        x=df_A.select_dtypes(exclude=["number"]).columns[0],
                        y=metric_A,
                        theme=THEMES[theme_choice]
                    )
                    st.plotly_chart(fig_A, use_container_width=True)

                    # Dataset B
                    fig_B = create_chart(
                        df_B,
                        chart_type,
                        x=df_B.select_dtypes(exclude=["number"]).columns[0],
                        y=metric_B,
                        theme=THEMES[theme_choice]
                    )
                    st.plotly_chart(fig_B, use_container_width=True)

                    # Collect structured data for AI
                    comparison_payload.append({
                        "metric": metric_A,
                        "dataset_A": {
                            "mean": round(df_A[metric_A].mean(), 2),
                            "max": round(df_A[metric_A].max(), 2),
                            "min": round(df_A[metric_A].min(), 2)
                        },
                        "dataset_B": {
                            "mean": round(df_B[metric_B].mean(), 2),
                            "max": round(df_B[metric_B].max(), 2),
                            "min": round(df_B[metric_B].min(), 2)
                        }
                    })

                except Exception as e:
                    st.error(f"Unable to compare metric '{metric_A}'")

        # Save for AI explanation step
        st.session_state["comparison_payload"] = comparison_payload
        st.session_state["comparison_labels"] = (dataset_A_name, dataset_B_name)

    # -----------------------------
    # STEP 6: AI COMPARISON REASONING
    # -----------------------------
    st.markdown("### 🧠 AI Comparison Insights")

    comparison_payload = st.session_state.get("comparison_payload", [])

    if not comparison_payload:
        st.info("Visual comparison data not available yet.")
    else:
        insights = []
        recommendations = []

        for item in comparison_payload:
            metric = item["metric"]

            A = item["dataset_A"]
            B = item["dataset_B"]

            diff = round(A["mean"] - B["mean"], 2)

            if diff > 0:
                winner = "Dataset A"
                strength = "higher"
            elif diff < 0:
                winner = "Dataset B"
                strength = "lower"
            else:
                winner = "Both datasets"
                strength = "similar"

            confidence = calculate_confidence(
                mean=max(A["mean"], B["mean"]),
                std=abs(diff) + 1e-6,
                rows=max(df_A.shape[0], df_B.shape[0])
            )

            insight_text = speak_like_ai(
                f"For '{metric}', {winner} shows {strength} average values compared to the other dataset"
            )

            insights.append(f"{insight_text} (Confidence: {confidence}%)")

            if abs(diff) > 0.15 * max(A["mean"], B["mean"], 1):
                recommendations.append(
                    f"Consider investigating factors influencing '{metric}', as the difference is significant."
                )

        # --- DISPLAY ---
        for ins in insights:
            st.info(ins)

        if recommendations:
            st.markdown("### 💡 AI Recommendations")
            for rec in recommendations:
                st.success(rec)

    # -----------------------------
    # UPGRADE 1: AI FINAL VERDICT
    # -----------------------------
    st.markdown("## 🏆 AI Final Verdict")

    if comparison_payload:
        score_A = 0
        score_B = 0
        reasons = []

        for item in comparison_payload:
            A = item["dataset_A"]["mean"]
            B = item["dataset_B"]["mean"]

            if A > B:
                score_A += 1
                reasons.append(f"{item['metric']} favors Dataset A")
            elif B > A:
                score_B += 1
                reasons.append(f"{item['metric']} favors Dataset B")

        if score_A > score_B:
            winner = "Dataset A"
        elif score_B > score_A:
            winner = "Dataset B"
        else:
            winner = "Neither — both datasets perform similarly"

        confidence = calculate_confidence(
            mean=max(score_A, score_B, 1),
            std=abs(score_A - score_B) + 1e-6,
            rows=len(comparison_payload)
        )

        verdict_text = speak_like_ai(
            f"{winner} emerges as the overall stronger dataset based on aggregated metric performance"
        )

        st.success(f"### 🏆 Verdict: {winner}")
        st.info(f"{verdict_text} (Confidence: {confidence}%)")

        with st.expander("📌 Reasoning Breakdown"):
            for r in reasons:
                st.markdown(f"• {r}")

    else:
        st.info("Final verdict will appear after comparisons are generated.")

    # -----------------------------
    # UPGRADE 2.1: MULTI-DATASET SELECTOR
    # -----------------------------
    st.markdown("## 📂 Select Datasets to Compare")

    available_datasets = list(DATASETS.keys())

    selected_datasets = st.multiselect(
        "Choose 2 or more datasets",
        available_datasets,
        default=available_datasets[:2]
    )

    if len(selected_datasets) < 2:
        st.warning("Please select at least 2 datasets to compare.")
        st.stop()

    # -----------------------------
    # UPGRADE 2.2: LOAD & NORMALIZE DATA
    # -----------------------------
    datasets = {}

    for name in selected_datasets:
        df_temp = DATASETS[name].copy()
        numeric_df = df_temp.select_dtypes(include=["number"])

        if numeric_df.empty:
            st.warning(f"{name} has no numeric data and was skipped.")
            continue

        datasets[name] = numeric_df

    # -----------------------------
    # UPGRADE 2.3: COMMON METRICS
    # -----------------------------
    common_metrics = set.intersection(
        *[set(df.columns) for df in datasets.values()]
    )

    if not common_metrics:
        st.error("No common numeric metrics found across selected datasets.")
        st.stop()

    st.success(f"Common metrics detected: {', '.join(common_metrics)}")

    # -----------------------------
    # UPGRADE 2.4: METRIC COMPARISON TABLE
    # -----------------------------
    st.markdown("## 📊 Multi-Dataset Metric Comparison")

    comparison_table = []

    for metric in common_metrics:
        row = {"Metric": metric}
        for name, df_metric in datasets.items():
            row[name] = round(df_metric[metric].mean(), 2)
        comparison_table.append(row)

    comparison_df = pd.DataFrame(comparison_table)
    st.dataframe(comparison_df, use_container_width=True)

    # -----------------------------
    # UPGRADE 2.5: RANKING ENGINE
    # -----------------------------
    ranking_payload = []

    for metric in common_metrics:
        scores = {
            name: round(df[metric].mean(), 2)
            for name, df in datasets.items()
        }

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        ranking_payload.append({
            "metric": metric,
            "ranking": ranked
        })

    with st.expander("🏅 Metric Rankings"):
        for r in ranking_payload:
            st.markdown(f"**{r['metric']}**")
            for i, (name, val) in enumerate(r["ranking"], start=1):
                st.markdown(f"{i}. {name}: {val}")

    # -----------------------------
    # UPGRADE 3.1: TIME COLUMN DETECTION
    # -----------------------------
    TIME_KEYWORDS = ["year", "date", "time", "month"]

    def detect_time_column(df):
        for col in df.columns:
            if any(k in col.lower() for k in TIME_KEYWORDS):
                return col
        return None

    time_columns = {
        name: detect_time_column(df)
        for name, df in DATASETS.items()
        if name in selected_datasets
    }

    valid_time_datasets = {
        name: DATASETS[name]
        for name, col in time_columns.items()
        if col is not None
    }

    if len(valid_time_datasets) < 2:
        st.warning("At least two datasets must contain a time column for time-aligned comparison.")
        st.stop()

    # -----------------------------
    # UPGRADE 3.3: TIME RANGE SELECTOR
    # -----------------------------
    st.markdown("## ⏳ Select Time Range")

    sample_df = next(iter(valid_time_datasets.values()))
    time_col = detect_time_column(sample_df)

    min_time = int(sample_df[time_col].min())
    max_time = int(sample_df[time_col].max())

    selected_range = st.slider(
        "Choose comparison time window",
        min_time,
        max_time,
        (min_time, max_time)
    )

    # -----------------------------
    # UPGRADE 3.4: TIME-ALIGNED AGGREGATION
    # -----------------------------
    aligned_data = []

    for name, df in valid_time_datasets.items():
        t_col = detect_time_column(df)
        df_filtered = df[
            (df[t_col] >= selected_range[0]) &
            (df[t_col] <= selected_range[1])
        ]

        for metric in common_metrics:
            if metric in df_filtered.columns:
                grouped = df_filtered.groupby(t_col)[metric].mean().reset_index()
                grouped["Dataset"] = name
                grouped["Metric"] = metric
                aligned_data.append(grouped)

    aligned_df = pd.concat(aligned_data, ignore_index=True)

    # -----------------------------
    # UPGRADE 3.5: TREND COMPARISON CHART
    # -----------------------------
    st.markdown("## 📈 Time-Aligned Trend Comparison")

    selected_metric = st.selectbox(
        "Select metric to compare over time",
        list(common_metrics)
    )

    fig = px.line(
        aligned_df[aligned_df["Metric"] == selected_metric],
        x=time_col,
        y=selected_metric,
        color="Dataset",
        markers=True,
        title=f"Trend Comparison for {selected_metric}"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # UPGRADE 3.6: AI TREND REASONING
    # -----------------------------
    st.markdown("## 🧠 AI Comparison Insight")

    trend_summary = aligned_df.groupby("Dataset")[selected_metric].agg(
        start="first",
        end="last"
    )

    for dataset, row in trend_summary.iterrows():
        delta = row["end"] - row["start"]
        direction = "increased 📈" if delta > 0 else "decreased 📉"
        st.info(
            f"{dataset}: {selected_metric} {direction} by {round(abs(delta),2)} over the selected period."
        )

    # -----------------------------
    # UPGRADE 4.1: DRIVER DETECTION
    # -----------------------------
    drivers_df = pd.DataFrame()
    st.markdown("## 🔍 Root-Cause Drivers")

    driver_scores = []

    for name, df in valid_time_datasets.items():
        t_col = detect_time_column(df)
    
        for metric in common_metrics:
            if metric not in df.columns:
                continue
    
            series = df.sort_values(t_col)[metric].dropna()
            if len(series) < 5:
                continue
    
            # strength = trend magnitude × volatility signal
            trend = np.polyfit(range(len(series)), series, 1)[0]
            volatility = series.std() / (series.mean() + 1e-6)
    
            driver_score = abs(trend) * volatility
    
            driver_scores.append({
                                  "Dataset": name,
                                  "Metric": metric,
                                  "Trend": round(trend, 3),
                                  "Volatility": round(volatility, 3),
                                  "DriverScore": round(driver_score, 3)
                              })
    drivers_df = pd.DataFrame(driver_scores)


    # -----------------------------
    # UPGRADE 4.2: TOP DRIVERS
    # -----------------------------
    if not drivers_df.empty:
        top_drivers = (
            drivers_df.sort_values("DriverScore", ascending=False)
            .groupby("Dataset")
            .head(2)
        )

        st.dataframe(
            top_drivers[["Dataset", "Metric", "Trend", "Volatility", "DriverScore"]],
            use_container_width=True
        )
    else:
        st.info("No sufficient data for driver analysis.")
    
    if drivers_df.empty:
              st.stop()

    # -----------------------------
    # UPGRADE 4.3: ROOT-CAUSE AI EXPLANATION
    # -----------------------------
    st.markdown("## 🧠 Why These Differences Exist")

    for _, row in top_drivers.iterrows():
        direction = "growth" if row["Trend"] > 0 else "decline"
        stability = "unstable" if row["Volatility"] > 0.6 else "stable"

        explanation = (
            f"In **{row['Dataset']}**, the metric **{row['Metric']}** appears to be a key driver. "
            f"It shows a {direction} trend combined with {stability} behavior, "
            f"suggesting it plays a significant role in shaping overall performance."
        )

        st.info(explanation)

    # -----------------------------
    # UPGRADE 4.4: CROSS-DATASET CAUSE COMPARISON
    # -----------------------------
    st.markdown("## 🔗 Cross-Dataset Cause Analysis")

    cause_matrix = top_drivers.pivot_table(
        index="Metric",
        columns="Dataset",
        values="DriverScore"
    )

    st.dataframe(cause_matrix.fillna(0).round(2), use_container_width=True)

    # -----------------------------
    # UPGRADE 4.5: EXECUTIVE SUMMARY
    # -----------------------------
    st.markdown("## 🧾 AI Executive Summary")

    for metric in cause_matrix.index:
        dominant = cause_matrix.loc[metric].idxmax()
        score = cause_matrix.loc[metric].max()

        st.success(
            f"Across datasets, **{metric}** most strongly influences outcomes in **{dominant}** "
            f"(impact score ≈ {round(score,2)})."
        )

    # -----------------------------
    # COMPARE REPORT
    # -----------------------------

    st.markdown("## 📝 Comparison Report")

    # Button triggers state
    if st.button("🧠 Generate Comparison Report"):
        st.session_state["show_compare_report"] = True

    # UI renders from state (THIS is the fix)
    if st.session_state.get("show_compare_report", False):

        if "comparison_payload" not in st.session_state:
            st.warning("Run a comparison first to generate a report.")
        else:
            label_A, label_B = st.session_state["comparison_labels"]

            report_text = generate_compare_report_text(
                                  dataset_A_name=label_A,
                                  dataset_B_name=label_B,
                                  comparison_payload=st.session_state["comparison_payload"]
                              )

            st.markdown(report_text)

            st.download_button(
                label="⬇️ Download Comparison Report (TXT)",
                data=report_text,
                file_name="comparison_report.txt",
                mime="text/plain"
            )


