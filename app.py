"""
AI-Powered Excel Assistant for Academic Data
---------------------------------------------
A Streamlit demo that lets faculty/admin staff interact with spreadsheets
(attendance, grades, fees, budgets) using plain natural language, powered
by Groq's LLM API.

Includes a genuinely agentic "Agent Mode": a tool-calling loop where the
LLM autonomously decides which action to take (query data, check for
anomalies, diagnose a formula, find unregistered students, draft a
notification), observes the result of each action, and chains further
steps on its own until the user's goal is met — rather than the user
picking a single fixed tool ahead of time.

Setup:
1. pip install -r requirements.txt
2. Create a file at .streamlit/secrets.toml with:
       GROQ_API_KEY = "your_groq_api_key_here"
3. Run: streamlit run app.py
"""

import json
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

MODEL = "llama-3.3-70b-versatile"  # supports Groq tool/function calling
MAX_AGENT_STEPS = 5


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
# Shared system prompts (reused by both the single-shot tabs AND the agent
# tools below, so the agent behaves identically to the manual tabs).
# --------------------------------------------------------------------------
QUERY_SYSTEM_PROMPT = (
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

ANOMALY_SYSTEM_PROMPT = (
    "You are a data quality assistant for academic records. Given summary "
    "information about a pandas DataFrame, identify likely data quality issues: "
    "missing values, duplicate rows, inconsistent formatting, outliers in numeric "
    "columns, and suspicious values (e.g. negative marks, attendance over 100%). "
    "Respond with a concise bullet-point list, no code."
)

FORMULA_SYSTEM_PROMPT = (
    "You are an Excel formula expert. Explain what the given formula does, "
    "step by step, in plain English. If it looks broken or likely to error, "
    "explain why and suggest a corrected formula. Keep it concise and "
    "beginner-friendly, and format the corrected formula in a code block."
)

REMINDER_SYSTEM_PROMPT = (
    "You are a student affairs assistant. Write a short, warm, and clear "
    "reminder notification for a student who has NOT yet registered for a "
    "college placement/event drive. Use the placeholder {student_name} "
    "exactly once where the student's name should go. Keep it under 80 "
    "words, friendly but urgent, and end with a clear call to action. "
    "Return ONLY the message text — no subject line, no extra commentary."
)


# --------------------------------------------------------------------------
# Agent Mode: tool definitions + autonomous tool-calling loop
#
# This is what makes the app agentic rather than a single-shot Q&A tool:
#   1. The model is given several tools (not just one) and DECIDES on its
#      own which one(s) the request needs — the user does not pick a tab.
#   2. Each tool call's result (the "observation") is fed back to the model
#      before it decides its next move.
#   3. The model can chain multiple tool calls in sequence to satisfy a
#      single request ("find anomalies, then show me the top scorers")
#      without the user breaking the task into steps themselves.
# --------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_dataframe",
            "description": (
                "Run pandas code against the uploaded DataFrame `df` to answer "
                "questions, filter/sort/aggregate data, or build a matplotlib "
                "chart. Use this for anything about the actual data: attendance, "
                "marks, fees, ranking, filtering, counting, plotting, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python using pandas (pd), numpy (np), matplotlib.pyplot "
                            "(plt) and the variable df. Assign the answer to `result` "
                            "(DataFrame/Series/value) and, for charts, a matplotlib "
                            "figure to `fig`."
                        ),
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": (
                "Analyze the DataFrame for data quality issues: missing values, "
                "duplicate rows, outliers, and suspicious values like negative "
                "marks or attendance over 100%. Returns a bullet-point report."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diagnose_formula",
            "description": "Explain what an Excel formula does and suggest a fix if it looks broken.",
            "parameters": {
                "type": "object",
                "properties": {
                    "formula": {
                        "type": "string",
                        "description": "The Excel formula, e.g. =VLOOKUP(A2,Sheet2!A:B,2,FALSE)",
                    }
                },
                "required": ["formula"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_unregistered_students",
            "description": (
                "Find students who have NOT registered for a drive/event, based "
                "on a status column and the value that means 'registered'. Use "
                "the exact column names from the DataFrame info you were given."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "registration_column": {"type": "string"},
                    "registered_value": {"type": "string"},
                    "name_column": {"type": "string"},
                },
                "required": ["registration_column", "registered_value", "name_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_reminder_message",
            "description": (
                "Draft a short reminder notification (with a {student_name} "
                "placeholder) for students who haven't registered for a drive/event. "
                "Call this AFTER find_unregistered_students if the user wants them notified."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drive_name": {"type": "string"},
                    "deadline": {"type": "string"},
                    "details": {"type": "string"},
                },
                "required": ["drive_name"],
            },
        },
    },
]

AGENT_SYSTEM_PROMPT = (
    "You are an autonomous data-assistant agent for a university academic "
    "spreadsheet. You have several tools available and must decide for "
    "yourself which ones the user's request needs, and in what order.\n\n"
    "- Break multi-part requests into separate tool calls and run them one "
    "at a time, using each result to inform the next step.\n"
    "- Do not ask the user to run steps manually — chain the tool calls yourself.\n"
    "- Only respond with plain text (no tool call) once you have everything "
    "needed to give a final answer. In that final answer, briefly summarize "
    "in plain English what each tool call showed you and answer the user's "
    "original request directly.\n\n"
    f"Here is the current data you're working with:\n{{context}}"
)


def describe_result(result) -> str:
    """Turn a tool's raw output into a short text 'observation' the LLM can read."""
    if isinstance(result, pd.DataFrame):
        return f"DataFrame with shape {result.shape}. Preview:\n{result.head(10).to_string()}"
    if isinstance(result, pd.Series):
        return f"Series with {len(result)} values. Preview:\n{result.head(10).to_string()}"
    if result is None:
        return "No `result` variable was set."
    return str(result)[:2000]


def execute_tool(name: str, args: dict, df: pd.DataFrame):
    """Runs one tool call. Returns (observation_text, table_or_None, fig_or_None)."""
    if name == "query_dataframe":
        code = args.get("code", "")
        if not is_code_safe(code):
            return "Blocked: generated code failed the safety check.", None, None
        local_env = {"df": df.copy(), "pd": pd, "np": np, "plt": plt}
        try:
            exec(code, {}, local_env)
            result = local_env.get("result")
            fig = local_env.get("fig")
            table = result if isinstance(result, (pd.DataFrame, pd.Series)) else None
            return describe_result(result), table, fig
        except Exception as e:
            return f"Error running code: {e}. Try a different approach.", None, None

    if name == "detect_anomalies":
        missing = df.isna().sum()
        dup = df.duplicated().sum()
        desc = df.describe(include="all").to_string()
        context = (
            f"{build_context(df)}\nMissing values per column:\n{missing}\n"
            f"Duplicate rows: {dup}\nDescribe:\n{desc}"
        )
        report = call_groq(ANOMALY_SYSTEM_PROMPT, context)
        return report, None, None

    if name == "diagnose_formula":
        formula = args.get("formula", "")
        explanation = call_groq(FORMULA_SYSTEM_PROMPT, formula)
        return explanation, None, None

    if name == "find_unregistered_students":
        reg_col = args.get("registration_column")
        reg_val = args.get("registered_value", "Yes")
        name_col = args.get("name_column")
        if reg_col not in df.columns or name_col not in df.columns:
            return f"Column not found. Available columns: {df.columns.tolist()}", None, None
        normalized = df[reg_col].astype(str).str.strip().str.lower()
        target = str(reg_val).strip().lower()
        mask = df[reg_col].isna() | (normalized != target)
        nr_df = df.loc[mask, [name_col, reg_col]].copy()
        obs = f"{len(nr_df)} students not registered. Names: {nr_df[name_col].head(15).tolist()}"
        return obs, nr_df, None

    if name == "generate_reminder_message":
        drive_name = args.get("drive_name", "the upcoming drive")
        deadline = args.get("deadline", "not specified")
        details = args.get("details", "none")
        detail_text = (
            f"Drive/event name: {drive_name}\n"
            f"Registration deadline: {deadline}\n"
            f"Other details: {details}"
        )
        template = call_groq(REMINDER_SYSTEM_PROMPT, detail_text, temperature=0.4).strip()
        return template, None, None

    return f"Unknown tool: {name}", None, None


def run_agent(user_request: str, df: pd.DataFrame, max_steps: int = MAX_AGENT_STEPS):
    """
    The autonomous loop: ask the model what to do -> if it wants a tool,
    run it and feed the observation back -> repeat -> stop once the model
    answers in plain text instead of calling a tool.
    """
    client = get_client()
    trace = []
    last_table, last_fig = None, None

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT.format(context=build_context(df))},
        {"role": "user", "content": user_request},
    ]

    for _ in range(max_steps):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=1500,
            )
        except Exception as e:
            trace.append({"type": "final", "content": f"Agent error: {e}"})
            return trace, f"Agent error: {e}", last_table, last_fig

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            trace.append({"type": "final", "content": msg.content or ""})
            return trace, msg.content or "", last_table, last_fig

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}

            observation, table, fig = execute_tool(name, args, df)
            if table is not None:
                last_table = table
            if fig is not None:
                last_fig = fig

            trace.append({"type": "action", "tool": name, "args": args, "observation": observation})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": str(observation)[:4000],
            })

    trace.append({"type": "final", "content": "Reached the step limit without a final answer."})
    return trace, "Reached the step limit without a final answer.", last_table, last_fig


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
        # Attendance
        "Show students with attendance below 75%",
        "Show students with attendance below 60%",
        "Show department-wise average attendance",
        "Which students are eligible for exams (attendance 75% or above)?",
        "Rank students by attendance, lowest first",
        "Plot attendance distribution as a histogram",
        # Marks / academics
        "Calculate average marks per subject",
        "Calculate average marks per department",
        "Highlight the top 10 scorers",
        "Highlight the bottom 5 scorers who need support",
        "Show subject-wise pass and fail counts (pass mark is 40)",
        "Rank students by marks within each subject",
        "Which subject has the widest spread of scores?",
        "Plot a bar chart of average marks by subject",
        # Fees
        "Show students who haven't paid fees",
        "List students with fees due over 10000",
        "What is the total fees due across all students?",
        "Plot a pie chart of fees paid vs not paid",
        "Which department has the highest total fees pending?",
        # Placement drive
        "Show students who haven't registered for the drive",
        "Count how many students registered for the drive, department-wise",
        "Show students eligible for the drive (75%+ attendance) but not registered",
        # At-risk / combined
        "Show students with both low attendance and low marks who need attention",
        "List semester-wise student counts",
        "Show students with attendance above 90% and marks above 85%",
        # Data quality
        "Find duplicate entries",
        "Show all rows with missing values",
        "Show summary statistics of all numeric columns",
        "Show the correlation between attendance and marks",
    ])

    st.divider()
    st.header("4. Agent examples")
    agent_examples = [
        "-- pick one --",
        "Find any data quality issues, then show me students with attendance below 75%",
        "Check who hasn't registered for the drive and draft a reminder message for them",
        "Show me average marks per subject, then highlight which subject has the most failures",
        "Diagnose this formula =VLOOKUP(A2,Sheet2!A:B,2,FALSE) and also show me duplicate rows",
    ]
    agent_quick = st.selectbox("Try a multi-step agent request", agent_examples)

if st.session_state.df is None:
    st.info("👈 Upload a file or load the sample dataset to get started.")
    st.stop()

df = st.session_state.df

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab1, tab_agent, tab2, tab3, tab4, tab5 = st.tabs([
    "🗂 Data Preview", "🤖 Agent Mode", "💬 Ask Assistant",
    "🧹 Clean & Anomalies", "🧮 Formula Doctor", "📢 Drive Notifications",
])

# ---- Tab 1: Data preview ---------------------------------------------------
with tab1:
    st.subheader("Data Preview")
    st.dataframe(df, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isna().sum().sum()))

# ---- Tab Agent: autonomous multi-step assistant ----------------------------
with tab_agent:
    st.subheader("🤖 Agent Mode — autonomous, multi-step, tool-calling assistant")
    st.caption(
        "Give it a goal instead of a single instruction. The agent decides which "
        "tools it needs, in what order, observes each result, and chains further "
        "steps on its own until it can answer you — you don't pick a tab."
    )
    with st.expander("What makes this 'agentic'? (vs. the other tabs)"):
        st.markdown(
            "- **Autonomous tool selection** — the model itself decides whether to "
            "query data, check for anomalies, diagnose a formula, or draft a "
            "notification, based on what you asked.\n"
            "- **Multi-step planning** — it can chain several tool calls in sequence "
            "for one request (e.g. *find anomalies, then show me the top scorers*) "
            "without you breaking the task into steps yourself.\n"
            "- **Observes before acting** — after every tool call, the result is fed "
            "back to the model before it decides its next move, so later steps can "
            "depend on earlier ones.\n\n"
            "The other tabs (Ask Assistant, Clean & Anomalies, Formula Doctor, Drive "
            "Notifications) are single-shot: one action per click, chosen by you. "
            "This tab wraps the *same* underlying tools into one autonomous loop."
        )

    agent_default = agent_quick if agent_quick != "-- pick one --" else ""
    agent_query = st.text_area(
        "What do you need?",
        value=agent_default,
        height=90,
        placeholder="e.g. Find anomalies in the data, then show me the top 10 scorers as a table",
    )
    run_agent_btn = st.button("🚀 Run Agent", type="primary")

    if run_agent_btn and agent_query.strip():
        with st.spinner("Agent is planning and taking actions..."):
            trace, final_text, last_table, last_fig = run_agent(agent_query, df)

        st.markdown("### 🧭 Agent trace")
        for i, step in enumerate(trace, start=1):
            if step["type"] == "action":
                with st.expander(f"Step {i}: called `{step['tool']}`", expanded=False):
                    if step["args"]:
                        st.json(step["args"])
                    st.markdown("**Observation:**")
                    st.write(step["observation"])
            else:
                st.markdown(f"**Step {i}: final reasoning**")

        st.markdown("### ✅ Result")
        if last_fig is not None:
            st.pyplot(last_fig)
        if isinstance(last_table, (pd.DataFrame, pd.Series)):
            st.dataframe(last_table, use_container_width=True)
        st.markdown(final_text if final_text else "_No final summary returned._")

# ---- Tab 2: Ask assistant (single-shot, for comparison) -------------------
with tab2:
    st.subheader("Ask a question about your data")
    st.caption("Single-shot version: one question → one action. Compare this with Agent Mode above.")
    default_q = quick_action if quick_action != "-- pick one --" else ""
    query = st.text_input(
        "e.g. 'Show me students below 75% attendance'", value=default_q
    )
    run = st.button("Run", type="primary")

    if run and query.strip():
        with st.spinner("Thinking..."):
            user_prompt = f"DataFrame info:\n{build_context(df)}\n\nQuestion: {query}"
            raw = call_groq(QUERY_SYSTEM_PROMPT, user_prompt)
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
            missing = df.isna().sum()
            dup = df.duplicated().sum()
            desc = df.describe(include="all").to_string()
            context = (
                f"{build_context(df)}\n"
                f"Missing values per column:\n{missing}\n"
                f"Duplicate rows: {dup}\n"
                f"Describe:\n{desc}"
            )
            report = call_groq(ANOMALY_SYSTEM_PROMPT, context)
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
                explanation = call_groq(FORMULA_SYSTEM_PROMPT, formula)
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
                    details = (
                        f"Drive/event name: {drive_name or 'the upcoming drive'}\n"
                        f"Registration deadline: {deadline or 'not specified'}\n"
                        f"Other details: {extra_details or 'none'}"
                    )
                    template = call_groq(REMINDER_SYSTEM_PROMPT, details, temperature=0.4).strip()
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
st.caption(
    "Demo build — AI-generated code runs in a lightly sandboxed environment. "
    "Not for production use on sensitive data without further hardening."
)
