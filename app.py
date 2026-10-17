"""Verified Data Analyst Agent - web app.   Run:  python -m streamlit run app.py"""
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from dataagent import pipeline
from dataagent.llm import BadKey, ModelUnavailable, clean_key, describe, get_llm
from dataagent.report import BADGE, html_report, markdown

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data" / "heart_disease.csv"
SAMPLE_DESC = (ROOT / "data" / "heart_disease_description.txt").read_text(encoding="utf-8")
ICON = {"plan": "🧭", "check": "🔎", "question": "❓", "run": "▶️", "repair": "🛠️", "verify": "⚖️", "label": "🏷️", "done": "🏁"}

st.set_page_config(page_title="Verified Data Analyst Agent", layout="wide")
ss = st.session_state


def friendly(e: Exception) -> str:
    if isinstance(e, (BadKey, ModelUnavailable)):
        return str(e)
    if "connection" in type(e).__name__.lower() or "timeout" in type(e).__name__.lower():
        return "Could not reach Groq. Check your internet connection and try again."
    return f"Something went wrong: {type(e).__name__}: {str(e)[:300]}"


def save_key(key: str) -> None:
    env = ROOT / ".env"
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    lines = [l for l in lines if not l.startswith("GROQ_API_KEY=")] + [f"GROQ_API_KEY={key}"]
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")


with st.sidebar:
    st.header("Settings")
    if not clean_key(os.getenv("GROQ_API_KEY", "")):
        key = clean_key(st.text_input("Groq API key (free at console.groq.com)", type="password"))
        if key:
            os.environ["GROQ_API_KEY"] = key
            save_key(key)
            ss.pop("llm", None)
            st.rerun()
    elif st.button("Change key"):
        os.environ["GROQ_API_KEY"] = ""
        save_key("")
        ss.pop("llm", None)
        st.rerun()
    if "llm" not in ss:
        ss.llm = get_llm()
    st.caption(f"Model: **{describe(ss.llm)}**")
    if ss.llm is None:
        st.info("Demo mode: questions and analyses come from built-in statistical templates. "
                "Add a Groq key to use the AI agents.")
    n_q = st.slider("Number of questions", 1, 6, 3)
    st.divider()
    st.caption("Agents: Business Consultant -> Data Scientist -> Critic (CrewAI). Every number in an answer is "
               "checked against the real code output.")

st.title("Verified Data Analyst Agent")
st.write("Upload any CSV. AI agents ask the right questions, write and run the analysis code, and a critic checks "
         "every claim against the actual numbers before it reaches your report.")

# ---------------- 1. data
st.subheader("1. Data")
c1, c2 = st.columns([2, 1])
up = c1.file_uploader("CSV file", type=["csv"])
use_sample = c2.button("Use sample: heart disease data")
# The uploader keeps returning the same file on every click, so only a NEW file (or the sample button)
# starts a new analysis; otherwise every button press would wipe the questions and results.
new_src = None
if use_sample:
    new_src, desc, ss.upload_id = str(SAMPLE), SAMPLE_DESC, None
elif up is not None and ss.get("upload_id") != (up.name, up.size):
    tmp = Path(tempfile.mkdtemp()) / up.name
    tmp.write_bytes(up.getvalue())
    new_src, desc, ss.upload_id = str(tmp), "", (up.name, up.size)
if new_src:
    ss.src, ss.desc_default = new_src, desc
    for k in ("session", "questions", "findings", "chat"):
        ss.pop(k, None)

if "src" in ss:
    desc = st.text_area("What is this data about? (optional - leave empty and the agent writes a description)",
                        ss.get("desc_default", ""), height=110)
    if st.button("Load and profile", type="primary") or "session" not in ss:
        out = ROOT / "output" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
        try:
            ss.session = pipeline.prepare(ss.src, str(out), desc)
        except Exception as e:
            st.error(f"Could not read this file: {e}")
            st.stop()
        for k in ("questions", "findings", "chat"):
            ss.pop(k, None)

if "session" in ss:
    s = ss.session
    with st.expander(f"Data profile: {s.profile['rows']} rows, {len(s.profile['columns'])} columns", expanded=False):
        st.dataframe(pd.DataFrame([{"column": c, "type": i["kind"], "missing": i["missing"], "unique": i["unique"]}
                                   for c, i in s.profile["columns"].items()]), hide_index=True, width="stretch")
        st.write("**Cleaning steps:** " + " ".join(s.cleaning_log))
        st.dataframe(s.df.head(), width="stretch")

    # ---------------- 2. questions (human in the loop)
    st.subheader("2. Questions")
    if st.button("Suggest questions"):
        with st.spinner("Business Consultant agent is reading the data..."):
            try:
                ss.questions = pipeline.plan(s, ss.llm, n_q)
                ss.qver = ss.get("qver", 0) + 1     # new boxes, so the new questions are shown, not the old text
            except Exception as e:  # BadKey, ModelUnavailable, network errors
                st.error(friendly(e))
    if "questions" in ss:
        st.caption("Edit, remove or add questions before the analysis runs.")
        edited = []
        v = ss.get("qver", 0)
        for i, q in enumerate(ss.questions):
            a, b, c = st.columns([5, 3, 1])
            text = a.text_input(f"Question {i + 1}", q["question"], key=f"qt{v}_{i}",
                                placeholder="Type your own question, e.g. Did women survive more than men?")
            cols = b.multiselect("Columns", list(s.df.columns), [x for x in q["columns"] if x in s.df.columns],
                                 key=f"qc{v}_{i}", max_selections=2)
            keep = c.checkbox("Use", True, key=f"qk{v}_{i}")
            if keep and text.strip():
                edited.append({**q, "question": text, "columns": cols or pipeline.guess_columns(s.df, text)})
        if st.button("Add a question"):
            ss.questions.append({"question": "", "columns": [], "method": ""})
            st.rerun()
        if st.button(f"Run analysis on {len(edited)} question(s)", type="primary", disabled=not edited):
            with st.status("Agents working...", expanded=True) as status:
                try:
                    ss.findings = pipeline.run_all(s, edited, ss.llm, lambda k, t: st.write(f"{ICON.get(k, '')} {t}"))
                    status.update(label="Analysis complete", state="complete")
                except Exception as e:  # BadKey, ModelUnavailable, network errors
                    status.update(label="Stopped", state="error")
                    st.error(friendly(e))

# ---------------- 3. report
def show(f, i):
    icon, _ = BADGE[f.label]
    with st.container(border=True):
        st.markdown(f"**{i}. {f.question}**  \n{icon} **{f.label}**"
                    + (f"  ·  code attempts: {f.code_attempts}" if f.code_attempts else "")
                    + (f"  ·  revisions: {f.revisions}" if f.revisions else "")
                    + ("  ·  built-in template" if f.used_fallback else ""))
        st.write(f.summary)
        for m in f.issues:
            st.error(m)
        for m in f.warnings:
            st.warning(m)
        for img in f.images:
            st.image(img, width=520)
        with st.expander("Code and raw output"):
            st.code(f.code, language="python")
            st.code(f.output)


if ss.get("findings"):
    st.subheader("3. Report")
    fs = ss.findings
    m = st.columns(3)
    for col, label in zip(m, BADGE):
        col.metric(f"{BADGE[label][0]} {label}", sum(f.label == label for f in fs))
    for i, f in enumerate(fs, 1):
        show(f, i)
    model = describe(ss.llm)
    d1, d2 = st.columns(2)
    d1.download_button("Download report (HTML)", html_report(ss.session, fs, model), "analysis_report.html")
    d2.download_button("Download report (Markdown)", markdown(ss.session, fs, model), "analysis_report.md")

    # ---------------- 4. ask your data
    st.subheader("4. Ask your data")
    ss.setdefault("chat", [])
    for i, f in enumerate(ss.chat, 1):
        show(f, f"Q{i}")
    question = st.chat_input("Ask a follow-up question, e.g. 'Is tobacco use higher for people with family history?'")
    if question:
        with st.status("Answering...", expanded=False):
            try:
                ss.chat.append(pipeline.ask(ss.session, question, ss.llm, 100 + len(ss.chat),
                                            lambda k, t: st.write(f"{ICON.get(k, '')} {t}")))
            except Exception as e:  # BadKey, ModelUnavailable, network errors
                st.error(friendly(e))
        st.rerun()
