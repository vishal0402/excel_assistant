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
if "not_registered_df" not in st.session_state:
    st.session_state.not_registered_df = None
if "drive_messages" not in st.session_state:
    st.session_state.drive_messages = None

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
    SAMPLE_FILE = "demo_academic_data.xlsx"
    if st.button("Load sample academic dataset"):
        try:
            # Bundled demo file — realistic records with a few intentional
            # data-quality issues (duplicates, missing values, bad entries)
            # so the Clean & Anomalies tab has something to catch.
            sample_df = pd.read_excel(SAMPLE_FILE)
        except FileNotFoundError:
            # Fallback if demo_academic_data.xlsx isn't deployed alongside app.py
            rng = np.random.default_rng(42)
            n = 40
            sample_df = pd.DataFrame({
                "Student Name": [f"Student {i+1}" for i in range(n)],
                "Roll No": [f"R{1000+i}" for i in range(n)],
                "Subject": rng.choice(["Data Structures", "DBMS", "OS", "Networks"], n),
                "Marks": rng.integers(35, 100, n),
                "Attendance %": rng.integers(50, 100, n),
                "Fees Paid": rng.choice(["Yes", "No"], n, p=[0.75, 0.25]),
            })
            sample_df["Registered for Drive"] = rng.choice(
                ["Yes", "No"], n, p=[0.65, 0.35]
            )
            st.warning(f"'{SAMPLE_FILE}' not found — loaded a generated fallback dataset instead.")
        st.session_state.df = sample_df
        st.success(f"Sample dataset loaded! ({sample_df.shape[0]} rows)")

    st.divider()
    st.header("3. Quick prompts")
    quick_action = st.selectbox("Try a demo question", [
        "-- pick one --",
        "Show students with attendance below 75%",
        "Calculate average marks per subject",
        "Highlight the top 10 scorers",
        "Show students who haven't paid fees",
        "List students with fees due over 10000",
        "Show department-wise average attendance",
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🗂 Data Preview", "💬 Ask Assistant", "🧹 Clean & Anomalies",
     "🧮 Formula Doctor", "📢 Drive Notifications"]
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

# ---- Tab 5: Drive notifications -------------------------------------------
with tab5:
    st.subheader("Notify students who haven't registered for a drive")
    st.caption(
        "Pick the column that tracks registration status, tell the assistant what "
        "counts as 'registered', and it will find everyone who hasn't signed up and "
        "draft a reminder message for each of them."
    )

    columns = df.columns.tolist()

    def _guess_index(candidates, options, default=0):
        for i, col in enumerate(options):
            if any(c in col.lower() for c in candidates):
                return i
        return default

    c1, c2 = st.columns(2)
    with c1:
        reg_col = st.selectbox(
            "Registration status column",
            columns,
            index=_guess_index(["regist", "drive", "placement"], columns),
        )
    with c2:
        name_col = st.selectbox(
            "Student name column",
            columns,
            index=_guess_index(["name"], columns),
        )

    unique_vals = df[reg_col].dropna().astype(str).str.strip().unique().tolist()
    registered_value = st.selectbox(
        "Which value means the student IS registered?",
        unique_vals if unique_vals else ["Yes"],
    )

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        drive_name = st.text_input("Drive / event name", placeholder="e.g. TCS Campus Placement Drive")
    with dcol2:
        deadline = st.text_input("Registration deadline (optional)", placeholder="e.g. 25 July 2026")

    extra_details = st.text_area(
        "Any other details to include (optional)",
        placeholder="e.g. Registration link, eligibility criteria, venue",
    )

    find = st.button("🔎 Find students who haven't registered", type="primary")

    if find:
        normalized = df[reg_col].astype(str).str.strip().str.lower()
        target = str(registered_value).strip().lower()
        not_registered_mask = df[reg_col].isna() | (normalized != target)
        st.session_state.not_registered_df = df.loc[not_registered_mask].copy()
        st.session_state.reg_name_col = name_col
        st.session_state.drive_messages = None  # reset previously generated messages

    if "not_registered_df" in st.session_state and st.session_state.not_registered_df is not None:
        nr_df = st.session_state.not_registered_df
        name_col = st.session_state.reg_name_col

        st.metric("Students not yet registered", len(nr_df))
        if len(nr_df) > 0:
            st.dataframe(nr_df, use_container_width=True)

            if st.button("✉️ Generate reminder messages"):
                with st.spinner("Drafting messages..."):
                    system_prompt = (
                        "You are a student affairs assistant. Write a short, warm, and clear "
                        "reminder notification for a student who has NOT yet registered for a "
                        "college placement/event drive. Use the placeholder {student_name} "
                        "exactly once where the student's name should go. Keep it under 80 "
                        "words, friendly but urgent, and end with a clear call to action. "
                        "Return ONLY the message text — no subject line, no extra commentary."
                    )
                    details = (
                        f"Drive/event name: {drive_name or 'the upcoming drive'}\n"
                        f"Registration deadline: {deadline or 'not specified'}\n"
                        f"Other details: {extra_details or 'none'}"
                    )
                    template = call_groq(system_prompt, details, temperature=0.4).strip()
                    st.session_state.drive_messages = template

        else:
            st.success("Everyone in this sheet is already registered — nothing to send!")

    if st.session_state.get("drive_messages"):
        template = st.session_state.drive_messages
        nr_df = st.session_state.not_registered_df
        name_col = st.session_state.reg_name_col

        st.markdown("**Message template generated:**")
        st.info(template)

        if "{student_name}" in template:
            personalized = nr_df[[name_col]].copy()
            personalized["Message"] = personalized[name_col].apply(
                lambda n: template.replace(
                    "{student_name}", str(n) if pd.notna(n) else "Student"
                )
            )
        else:
            personalized = nr_df[[name_col]].copy()
            personalized["Message"] = template

        st.markdown("**Personalized messages, ready to send:**")
        with st.expander("Preview all personalized messages", expanded=False):
            for _, row in personalized.iterrows():
                st.markdown(f"**{row[name_col]}:** {row['Message']}")

        csv_bytes = personalized.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download reminder list (CSV)",
            data=csv_bytes,
            file_name="drive_registration_reminders.csv",
            mime="text/csv",
        )

st.divider()
st.caption("Demo build — AI-generated code runs in a lightly sandboxed environment. Not for production use on sensitive data without further hardening.")
