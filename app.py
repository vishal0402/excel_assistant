"""
AI-Powered Excel Assistant for Academic Data
---------------------------------------------
A Streamlit demo that lets faculty/admin staff interact with spreadsheets
(attendance, grades, fees, budgets) using plain natural language, powered
by Groq's LLM API.

Setup:
1. pip install -r requirements.txt
2. Create a file at .streamlit/secrets.toml with:
       GROQ_API_KEY = "your_groq_api_key_here"
3. Run: streamlit run app.py
"""

import io
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from groq import Groq

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(page_title="AI Excel Assistant", page_icon="📊", layout="wide")

MODEL = "llama-3.3-70b-versatile"


@st.cache_resource
def get_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in secrets. Add it to .streamlit/secrets.toml")
        st.stop()
    return Groq(api_key=api_key)


def call_groq(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=1500,
    )
    return resp.choices[0].message.content


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def build_context(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    dtypes = df.dtypes.astype(str).to_dict()
    sample = df.head(5).to_string()
    return (
        f"Columns: {cols}\n"
        f"Dtypes: {dtypes}\n"
        f"Total rows: {len(df)}\n"
        f"Sample rows:\n{sample}"
    )


# Very small safety net for the exec sandbox. This is a DEMO tool, not a
# production-grade sandbox — do not expose it to untrusted users as-is.
BLOCKED_TOKENS = [
    "import os", "import sys", "subprocess", "open(", "eval(", "exec(",
    "__import__", "shutil", "socket", ".system(", "input(",
]


def is_code_safe(code: str) -> bool:
    lowered = code.lower()
    return not any(tok.lower() in lowered for tok in BLOCKED_TOKENS)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("📊 AI-Powered Excel Assistant for Academic Data")
st.caption(
    "Upload a spreadsheet — attendance, grades, fees, budgets — and ask "
    "questions in plain English. No formulas, no VLOOKUPs, no pivot tables."
)

# --------------------------------------------------------------------------
# Sidebar: upload + quick prompts
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Upload your data")
    uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])

    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                xls = pd.ExcelFile(uploaded)
                sheet = st.selectbox("Sheet", xls.sheet_names)
                df = pd.read_excel(xls, sheet_name=sheet)
            st.session_state.df = df
            st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.divider()
    st.header("2. No file handy?")
    if st.button("Load sample academic dataset"):
        rng = np.random.default_rng(42)
        n = 40
        names = [f"Student {i+1}" for i in range(n)]
        sample_df = pd.DataFrame({
            "Student Name": names,
            "Roll No": [f"R{1000+i}" for i in range(n)],
            "Subject": rng.choice(["Data Structures", "DBMS", "OS", "Networks"], n),
            "Marks": rng.integers(35, 100, n),
            "Attendance %": rng.integers(50, 100, n),
            "Fees Paid": rng.choice(["Yes", "No"], n, p=[0.75, 0.25]),
        })
        st.session_state.df = sample_df
        st.success("Sample dataset loaded!")

    st.divider()
    st.header("3. Quick prompts")
    quick_action = st.selectbox("Try a demo question", [
        "-- pick one --",
        "Show students with attendance below 75%",
        "Calculate average marks per subject",
        "Highlight the top 10 scorers",
        "Show students who haven't paid fees",
        "Find duplicate entries",
        "Show summary statistics of all numeric columns",
        "Plot a bar chart of average marks by subject",
    ])

if st.session_state.df is None:
    st.info("👈 Upload a file or load the sample dataset to get started.")
    st.stop()

df = st.session_state.df

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🗂 Data Preview", "💬 Ask Assistant", "🧹 Clean & Anomalies", "🧮 Formula Doctor"]
)

# ---- Tab 1: Data preview -------------------------------------------------
with tab1:
    st.subheader("Data Preview")
    st.dataframe(df, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isna().sum().sum()))

# ---- Tab 2: Ask assistant -------------------------------------------------
with tab2:
    st.subheader("Ask a question about your data")
    default_q = quick_action if quick_action != "-- pick one --" else ""
    query = st.text_input(
        "e.g. 'Show me students below 75% attendance'", value=default_q
    )
    run = st.button("Run", type="primary")

    if run and query.strip():
        with st.spinner("Thinking..."):
            system_prompt = (
                "You are a data assistant that converts a natural language question about a "
                "pandas DataFrame called `df` into a single Python code snippet.\n"
                "Rules:\n"
                "- Only use pandas (pd), numpy (np), and matplotlib.pyplot (plt) — already imported.\n"
                "- Assign the final answer to a variable named `result`.\n"
                "- `result` should be a DataFrame, Series, or a plain Python value.\n"
                "- If a chart is requested, build it with matplotlib and assign the figure to `fig`; "
                "still set `result` to a short text description.\n"
                "- If asked to clean/fix data, assign the cleaned DataFrame to `result`.\n"
                "- Never read/write files, never import os/sys/subprocess, never call eval/exec.\n"
                "- Return ONLY a python code block, no prose."
            )
            user_prompt = f"DataFrame info:\n{build_context(df)}\n\nQuestion: {query}"
            raw = call_groq(system_prompt, user_prompt)
            code = extract_code(raw)

        with st.expander("Show generated code"):
            st.code(code, language="python")

        if not is_code_safe(code):
            st.error("Generated code failed the safety check and was not run.")
        else:
            local_env = {"df": df.copy(), "pd": pd, "np": np, "plt": plt}
            try:
                exec(code, {}, local_env)
                result = local_env.get("result")
                fig = local_env.get("fig")

                if fig is not None:
                    st.pyplot(fig)
                if isinstance(result, (pd.DataFrame, pd.Series)):
                    st.dataframe(result, use_container_width=True)
                elif result is not None:
                    st.write(result)
                else:
                    st.info("The assistant ran successfully but returned no `result`.")

                st.session_state.history.append({"query": query, "code": code})
            except Exception as e:
                st.error(f"Couldn't execute the generated code: {e}")
                st.info("Try rephrasing your question, or check the generated code above.")

    if st.session_state.history:
        with st.expander("🕘 Query history"):
            for h in reversed(st.session_state.history[-10:]):
                st.markdown(f"**Q:** {h['query']}")
                st.code(h["code"], language="python")

# ---- Tab 3: Clean & anomalies ---------------------------------------------
with tab3:
    st.subheader("Clean data & flag anomalies")
    st.caption("Get an AI-generated report on missing values, duplicates, and suspicious entries.")
    if st.button("🔍 Detect anomalies & data issues"):
        with st.spinner("Analyzing..."):
            system_prompt = (
                "You are a data quality assistant for academic records. Given summary "
                "information about a pandas DataFrame, identify likely data quality issues: "
                "missing values, duplicate rows, inconsistent formatting, outliers in numeric "
                "columns, and suspicious values (e.g. negative marks, attendance over 100%). "
                "Respond with a concise bullet-point list, no code."
            )
            missing = df.isna().sum()
            dup = df.duplicated().sum()
            desc = df.describe(include="all").to_string()
            context = (
                f"{build_context(df)}\n"
                f"Missing values per column:\n{missing}\n"
                f"Duplicate rows: {dup}\n"
                f"Describe:\n{desc}"
            )
            report = call_groq(system_prompt, context)
        st.markdown(report)

# ---- Tab 4: Formula doctor -------------------------------------------------
with tab4:
    st.subheader("Formula Doctor — paste a broken Excel formula")
    formula = st.text_area(
        "Paste your Excel formula here",
        placeholder="=VLOOKUP(A2,Sheet2!A:B,2,FALSE)",
    )
    if st.button("🩺 Diagnose formula"):
        if formula.strip():
            with st.spinner("Diagnosing..."):
                system_prompt = (
                    "You are an Excel formula expert. Explain what the given formula does, "
                    "step by step, in plain English. If it looks broken or likely to error, "
                    "explain why and suggest a corrected formula. Keep it concise and "
                    "beginner-friendly, and format the corrected formula in a code block."
                )
                explanation = call_groq(system_prompt, formula)
            st.markdown(explanation)
        else:
            st.warning("Paste a formula first.")

st.divider()
st.caption("Demo build — AI-generated code runs in a lightly sandboxed environment. Not for production use on sensitive data without further hardening.")
