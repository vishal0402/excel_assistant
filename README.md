# AI-Powered Excel Assistant (Demo)

A Streamlit demo where staff/faculty upload a spreadsheet (attendance, grades,
fees, budgets) and ask questions in plain English. Groq's LLM turns each
question into pandas code, runs it, and shows the result — table, number, or
chart. Includes a data-quality/anomaly checker and a "Formula Doctor" that
explains or fixes broken Excel formulas.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Add your Groq API key

Create `.streamlit/secrets.toml` in the project folder:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Get a free key at https://console.groq.com/keys

## 3. Run the app

```bash
streamlit run app.py
```

## What's inside

- **Data Preview** — quick look at the uploaded sheet (row/column counts, missing values).
- **Ask Assistant** — type a question like *"show students below 75% attendance"* or
  *"plot average marks by subject"*; the assistant writes and runs the pandas code for you.
- **Clean & Anomalies** — one click gets an AI report on missing data, duplicates, and
  suspicious values (e.g. attendance over 100%).
- **Formula Doctor** — paste any Excel formula and get a plain-English explanation
  plus a suggested fix if it looks broken.

A "Load sample academic dataset" button in the sidebar generates fake
student/marks/attendance/fees data if you don't have a file handy — good for
a quick demo.

## Notes for a demo/hackathon setting

- The generated pandas code runs via `exec()` with a basic keyword blocklist
  (no file I/O, no `os`/`sys`/`subprocess`, no `eval`/`exec`). This is fine for
  a live demo with a trusted presenter, but it is **not** a hardened sandbox —
  don't expose this build to untrusted public users or real student PII
  without further security work.
- Model used: `llama-3.3-70b-versatile` on Groq. Swap `MODEL` in `app.py` if
  you want a different Groq-hosted model.
