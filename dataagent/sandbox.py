"""Run AI-written analysis code safely-ish: separate process, time limit, own folder, no network/shell modules.

The original project ran the LLM's code with exec() inside the app process, so a bad or
endless piece of code could crash or freeze everything. This is a practical guard, not a
full security sandbox (that would need containers)."""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

BLOCKED = re.compile(
    r"^\s*(?:import|from)\s+(subprocess|socket|requests|urllib|http|ftplib|smtplib|shutil|ctypes|multiprocessing)\b"
    r"|\bos\.(?:system|popen|remove|unlink|rmdir|removedirs|kill|exec\w*|spawn\w*)\s*\("
    r"|\b(?:eval|exec|__import__)\s*\(", re.M)

PRELUDE = """import os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
DATA_PATH = {data!r}
OUT_DIR = {out!r}
"""


@dataclass
class RunResult:
    ok: bool
    stdout: str
    error: str = ""
    images: list[str] = field(default_factory=list)
    timed_out: bool = False


def extract_code(text: str) -> str:
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    return max(m, key=len).strip() if m else (text or "").strip()


def check_code(code: str) -> str | None:
    m = BLOCKED.search(code)
    return f"Blocked for safety: '{m.group(0).strip()}' is not allowed in analysis code." if m else None


def run_code(code: str, data_path: str, out_dir: str, timeout: int = 60, max_output: int = 6000) -> RunResult:
    problem = check_code(code)
    if problem:
        return RunResult(False, "", problem)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in out.glob("*.png")}
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "analysis.py"
        script.write_text(PRELUDE.format(data=str(Path(data_path).resolve()), out=str(out.resolve())) + code,
                          encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", str(script)], cwd=str(out), capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            partial = (e.stdout or b"")
            partial = partial.decode(errors="ignore") if isinstance(partial, bytes) else partial
            return RunResult(False, partial[:max_output], f"Timed out after {timeout} seconds.", timed_out=True)
    images = sorted(str(p) for p in out.glob("*.png") if p.name not in before)
    if proc.returncode != 0:
        err = proc.stderr.strip().splitlines()
        short = "\n".join(err[-8:])                      # the useful end of the traceback
        return RunResult(False, proc.stdout[:max_output], short, images)
    return RunResult(True, proc.stdout[:max_output], "", images)
