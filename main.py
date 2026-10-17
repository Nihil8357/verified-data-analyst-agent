"""Command line:
    python main.py data/heart_disease.csv --describe data/heart_disease_description.txt --questions 3
    python main.py my_data.csv --demo          # no AI model, built-in statistical templates
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

from dataagent import pipeline
from dataagent.llm import describe, get_llm
from dataagent.report import html_report, markdown

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Verified Data Analyst Agent")
    ap.add_argument("csv")
    ap.add_argument("--describe", default="", help="text file describing the dataset (optional)")
    ap.add_argument("--questions", type=int, default=3)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    llm = None if a.demo else get_llm()
    out = Path("output") / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    desc = Path(a.describe).read_text(encoding="utf-8") if a.describe else ""
    s = pipeline.prepare(a.csv, str(out), desc)
    emit = lambda k, t: print(f"[{k:8s}] {t}", flush=True)
    print(f"Model: {describe(llm)}\nCleaning: {' '.join(s.cleaning_log)}\n")
    qs = pipeline.plan(s, llm, a.questions, emit)
    findings = pipeline.run_all(s, qs, llm, emit)
    print()
    for i, f in enumerate(findings, 1):
        print(f"{i}. {f.question}\n   [{f.label}] {f.summary}")
        for m in f.issues + f.warnings:
            print(f"   ! {m}")
    (out / "report.md").write_text(markdown(s, findings, describe(llm)), encoding="utf-8")
    (out / "report.html").write_text(html_report(s, findings, describe(llm)), encoding="utf-8")
    print(f"\nReport: {out / 'report.html'}")
