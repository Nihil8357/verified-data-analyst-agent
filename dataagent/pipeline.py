"""Runs the whole analysis.

  profile & clean -> Quest crew (questions) -> [human can edit/approve] ->
  for each question:  EDA crew writes code -> sandbox runs it -> on error: fix (max 3)
                      -> EDA crew writes the answer -> Verifier + Critic -> revise (max 2)
                      -> trust label (Verified / Partly verified / Rejected)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from . import crews
from .profiler import auto_description, clean, data_card, load_csv, profile
from .sandbox import extract_code, run_code
from .templates import demo_questions, template_code, template_summary
from .verifier import verify

MAX_CODE_TRIES = 3
MAX_REVISIONS = 2
Emit = Callable[[str, str], None]


@dataclass
class Finding:
    question: str
    columns: list[str]
    method: str
    code: str = ""
    output: str = ""
    images: list[str] = field(default_factory=list)
    summary: str = ""
    label: str = "Rejected"
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    code_attempts: int = 0
    revisions: int = 0
    used_fallback: bool = False
    critic_notes: str = ""


@dataclass
class Session:
    """Everything about one dataset: cleaned file, profile and the text card the agents see."""
    source: str
    clean_path: str
    df: pd.DataFrame
    profile: dict
    card: str
    description: str
    cleaning_log: list[str]
    out_dir: str


def prepare(csv_path: str, out_dir: str, description: str = "") -> Session:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = clean(load_csv(csv_path))
    clean_path = out / "clean_data.csv"
    res.df.to_csv(clean_path, index=False)
    prof = profile(res.df)
    desc = description.strip() or auto_description(prof)
    return Session(csv_path, str(clean_path), res.df, prof, data_card(prof, desc), desc, res.log, str(out))


def valid_questions(qs: list[dict], df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """Drop questions that use columns the dataset does not have (LLMs invent column names)."""
    cols = {c.lower(): c for c in df.columns}
    keep, dropped = [], []
    for q in qs:
        fixed = [cols.get(str(c).lower()) for c in q.get("columns", [])]
        if q.get("question") and fixed and all(fixed):
            keep.append({**q, "columns": fixed})
        else:
            dropped.append(q.get("question", "?"))
    return keep, dropped


def plan(session: Session, llm, n: int = 3, emit: Emit = lambda k, t: None) -> list[dict]:
    if llm is None:
        qs = demo_questions(session.df, n)
        emit("plan", f"Demo mode: {len(qs)} questions chosen by built-in rules")
        return qs
    emit("plan", "Business Consultant agent is reading the data profile...")
    raw = crews.plan_questions(llm, session.card, n)
    qs, dropped = valid_questions(raw, session.df)
    for d in dropped:
        emit("check", f"Dropped a question that used a column not in the data: {d}")
    if len(qs) < n:
        used = {frozenset(x["columns"]) for x in qs}
        extra = [q for q in demo_questions(session.df, n * 3) if frozenset(q["columns"]) not in used]
        qs += extra[: n - len(qs)]
    emit("plan", f"{len(qs)} questions ready")
    return qs[:n]


def analyse(session: Session, q: dict, llm, idx: int, emit: Emit = lambda k, t: None) -> Finding:
    f = Finding(q["question"], q.get("columns", []), q.get("method", ""))
    img_dir = str(Path(session.out_dir) / f"q{idx}")
    emit("question", f"Q{idx}: {f.question}")

    # 1. code: write -> run -> repair
    feedback = ""
    result = None
    if llm is not None:
        for attempt in range(1, MAX_CODE_TRIES + 1):
            f.code_attempts = attempt
            f.code = extract_code(crews.write_code(llm, session.card, q, feedback))
            result = run_code(f.code, session.clean_path, img_dir)
            if result.ok:
                emit("run", f"code ran on attempt {attempt}" + (f", {len(result.images)} chart(s)" if result.images else ""))
                break
            emit("repair", f"attempt {attempt} failed: {result.error.splitlines()[-1][:150] if result.error else 'error'}")
            feedback = (f"Your previous code failed with this error:\n{result.error}\nPrevious code:\n{f.code}\n"
                        "Fix the problem and return the full corrected script.")
    if result is None or not result.ok:
        f.used_fallback = True
        f.code = template_code(q, session.df)
        result = run_code(f.code, session.clean_path, img_dir)
        emit("run", "using the built-in statistical template" + ("" if llm is None else " after 3 failed attempts"))
    f.output, f.images = result.stdout, result.images
    if not result.ok:
        f.issues = [f"Analysis could not be run: {result.error[:200]}"]
        return f

    # 2. answer: write -> verify + critic -> revise
    cols = list(session.df.columns)
    if llm is None or f.used_fallback:
        f.summary = template_summary(f.output)
    else:
        f.summary = crews.interpret(llm, q, f.output)
    v = verify(f.summary, f.output, f.question, cols, f.images, need_plot=True)
    if llm is not None and not f.used_fallback:
        for rnd in range(MAX_REVISIONS + 1):
            verdict = crews.critique(llm, q, f.output, f.summary, v.issues)
            f.critic_notes = verdict.feedback
            if not v.issues and verdict.approved:
                emit("verify", f"approved by the Critic, {v.matched_numbers}/{v.checked_numbers} numbers matched")
                break
            if rnd == MAX_REVISIONS:
                break
            problems = v.issues + ([verdict.feedback] if not verdict.approved else [])
            emit("verify", "sent back for revision: " + "; ".join(problems)[:200])
            f.revisions += 1
            f.summary = crews.interpret(llm, q, f.output, "Your previous answer was rejected: " + "; ".join(problems)
                                        + f"\nPrevious answer: {f.summary}\nWrite a corrected answer.")
            v = verify(f.summary, f.output, f.question, cols, f.images, need_plot=True)
        if not verdict.approved and not v.issues:
            v.warnings.append(f"Critic still had concerns: {verdict.feedback[:200]}")
    f.issues, f.warnings, f.label = v.issues, v.warnings, v.label
    emit("label", f"{f.label}" + (f" - {'; '.join(v.issues + v.warnings)[:160]}" if v.issues or v.warnings else ""))
    return f


def run_all(session: Session, questions: list[dict], llm, emit: Emit = lambda k, t: None) -> list[Finding]:
    t0 = time.perf_counter()
    findings = [analyse(session, q, llm, i, emit) for i, q in enumerate(questions, 1)]
    emit("done", f"{len(findings)} questions in {time.perf_counter() - t0:.1f}s")
    return findings


def ask(session: Session, text: str, llm, idx: int, emit: Emit = lambda k, t: None) -> Finding:
    """'Ask your data': any follow-up question in plain English, answered with verified numbers."""
    import re
    mentioned = [c for c in session.df.columns if re.search(rf"\b{re.escape(c)}\b", text, re.I)]
    q = {"question": text.strip(), "columns": mentioned[:2], "method": ""}
    if llm is None and not mentioned:
        f = Finding(q["question"], [], "")
        f.issues = ["Demo mode can only answer questions that name one or two columns, e.g. 'Does age differ by chd?'. "
                    "Add a Groq key for free-form questions."]
        return f
    return analyse(session, q, llm, idx, emit)
