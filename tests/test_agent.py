"""Tests for the Verified Data Analyst Agent.   Run:  python -m pytest -v
No API key or internet needed: ScriptedLLM plays the agents' parts inside the real CrewAI crews."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ["AGENT_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_llm import ScriptedLLM  # noqa: E402

from dataagent import pipeline  # noqa: E402
from dataagent.profiler import clean, column_kind, data_card, profile  # noqa: E402
from dataagent.report import html_report, markdown  # noqa: E402
from dataagent.sandbox import extract_code, run_code  # noqa: E402
from dataagent.verifier import verify  # noqa: E402

HEART = str(ROOT / "data" / "heart_disease.csv")
HEART_DESC = (ROOT / "data" / "heart_disease_description.txt").read_text()
GOOD_CODE = """```python
import pandas as pd
from scipy import stats
df = pd.read_csv(DATA_PATH)
a = df[df.chd == 1].age; b = df[df.chd == 0].age
t, p = stats.ttest_ind(a, b)
print("mean age chd=1:", round(a.mean(), 2)); print("mean age chd=0:", round(b.mean(), 2)); print("p-value =", p)
df.boxplot(column="age", by="chd"); plt.savefig(os.path.join(OUT_DIR, "chart.png"))
```"""
GOOD_ANSWER = ("People with heart disease are older on average (50.29 vs 38.85 years), and the difference is "
               "statistically significant (p < 0.001).")
Q_AGE = {"question": "Does age differ by chd?", "columns": ["age", "chd"], "method": "t-test"}


@pytest.fixture
def session(tmp_path):
    return pipeline.prepare(HEART, str(tmp_path / "run"), HEART_DESC)


@pytest.fixture
def messy_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({"Unnamed: 0": range(n), "Monthly Spend": [f"{x:,.0f}" for x in rng.normal(1500, 300, n)],
                       "plan": rng.choice(["basic", "pro", "team"], n), "tenure months": rng.integers(1, 60, n),
                       "churned": rng.choice(["yes", "no"], n)})
    df = pd.concat([df, df.iloc[:5]])                                   # 5 duplicate rows
    p = tmp_path / "customers.csv"
    df.to_csv(p, index=False)
    return str(p)


# ------------------------------------------------------------ data handling
def test_cleaning_on_messy_csv(messy_csv, tmp_path):
    s = pipeline.prepare(messy_csv, str(tmp_path / "m"))
    log = " ".join(s.cleaning_log)
    assert "Unnamed:_0" not in s.df.columns and "Found 5 identical rows; kept them" in log
    assert len(s.df) == 125                                            # duplicates flagged, not deleted
    assert "Monthly_Spend" in s.df.columns and column_kind(s.df["Monthly_Spend"]) == "numeric"   # "1,523" -> 1523
    assert "A table of 125 rows" in s.description                      # auto description when none given


def test_heart_profile_and_card(session):
    assert "row.names" not in session.df.columns
    assert session.profile["columns"]["famhist"]["kind"] == "binary"
    assert "- ldl [numeric]" in session.card and "Dataset description" in session.card


# ------------------------------------------------------------ sandbox
def test_sandbox_runs_code_and_finds_chart(tmp_path):
    r = run_code("import pandas as pd\ndf = pd.read_csv(DATA_PATH)\nprint(len(df))\ndf.age.hist()\n"
                 "plt.savefig(os.path.join(OUT_DIR, 'a.png'))", HEART, str(tmp_path))
    assert r.ok and r.stdout.strip() == "462" and len(r.images) == 1


def test_sandbox_reports_errors_and_timeouts(tmp_path):
    r = run_code("import pandas as pd\ndf = pd.read_csv(DATA_PATH)\ndf['nope']", HEART, str(tmp_path))
    assert not r.ok and "KeyError" in r.error
    r = run_code("import time\nprint('start')\ntime.sleep(10)", HEART, str(tmp_path), timeout=2)
    assert r.timed_out and "Timed out" in r.error


@pytest.mark.parametrize("bad", ["import subprocess\nsubprocess.run('dir')", "import os\nos.system('del x')",
                                 "import requests", "eval('1+1')"])
def test_sandbox_blocks_dangerous_code(tmp_path, bad):
    r = run_code(bad, HEART, str(tmp_path))
    assert not r.ok and r.error.startswith("Blocked for safety")


def test_extract_code():
    assert extract_code("Here:\n```python\nprint(1)\n```\nDone") == "print(1)"


# ------------------------------------------------------------ verifier
OUT = "Correlation sbp vs chd: 0.1923\nT-test: t = 4.12, p-value = 0.00005\nmean age: 50.29\n45.8% have family history"


def test_verifier_accepts_rounded_numbers_and_p_bounds():
    v = verify("Weak correlation (r = 0.19), statistically significant (p < 0.001). Mean age 50.3; 45.8% have it.",
               OUT, columns=["sbp", "chd"])
    assert v.label == "Verified" and v.matched_numbers == 3


@pytest.mark.parametrize("summary,expected", [
    ("Strong correlation r = 0.72.", "0.72 does not appear"),
    ("The result is significant.", None),                                           # p=0.00005 -> fine
    ("Significant at p < 0.00001.", "smallest printed p-value"),
    ("High `cholesterol` matters, r = 0.19.", "column `cholesterol`"),
])
def test_verifier_catches_wrong_claims(summary, expected):
    v = verify(summary, OUT, columns=["sbp", "chd", "age"])
    if expected:
        assert any(expected in i for i in v.issues), v.issues
    else:
        assert not v.issues


def test_verifier_significance_and_causal_language():
    assert "not < 0.05" in verify("This is significant.", "p-value = 0.21").issues[0]
    assert verify("No significant difference (p = 0.21).", "p-value = 0.21").label == "Verified"
    w = verify("Smoking causes heart disease (r = 0.19).", OUT).warnings
    assert any("causal wording" in x for x in w)


# ------------------------------------------------------------ agents (CrewAI crews with a scripted model)
def test_invalid_columns_are_dropped_and_replaced(session):
    llm = ScriptedLLM({"questions": {"questions": [Q_AGE, {"question": "Does cholesterol matter?",
                                                           "columns": ["cholesterol", "chd"], "method": "t-test"}]}})
    events = []
    qs = pipeline.plan(session, llm, 2, lambda k, t: events.append(t))
    assert qs[0]["question"] == Q_AGE["question"] and len(qs) == 2
    assert all(set(q["columns"]) <= set(session.df.columns) for q in qs)
    assert frozenset(qs[1]["columns"]) != frozenset(qs[0]["columns"])            # no duplicate fill-in
    assert any("cholesterol" in e for e in events)


def test_code_repair_and_critic_revision(session):
    llm = ScriptedLLM({"code": ["```python\nimport pandas as pd\nprint(pd.read_csv(DATA_PATH)['agee'].mean())\n```",
                                GOOD_CODE],
                       "answer": ["Mean age is 55.0 for the chd group, which causes disease.", GOOD_ANSWER],
                       "critique": [{"approved": False, "feedback": "55.0 is wrong"}, {"approved": True}]})
    f = pipeline.analyse(session, Q_AGE, llm, 1)
    assert f.code_attempts == 2 and f.revisions == 1 and f.label == "Verified" and len(f.images) == 1
    assert llm.log == ["code", "code", "answer", "critique", "answer", "critique"]


def test_fallback_template_after_three_failed_attempts(session):
    llm = ScriptedLLM({"code": "```python\nraise ValueError('broken')\n```"})
    f = pipeline.analyse(session, Q_AGE, llm, 1)
    assert f.code_attempts == 3 and f.used_fallback and f.label == "Verified" and "50.29" in f.summary


def test_persistently_wrong_answer_is_rejected(session):
    llm = ScriptedLLM({"code": GOOD_CODE, "answer": "Mean age is 61.4 years.",
                       "critique": {"approved": False, "feedback": "61.4 not in output"}})
    f = pipeline.analyse(session, Q_AGE, llm, 1)
    assert f.label == "Rejected" and f.revisions == 2 and "61.4" in f.issues[0]


def test_critic_concern_without_number_errors_is_a_warning(session):
    llm = ScriptedLLM({"code": GOOD_CODE, "answer": GOOD_ANSWER,
                       "critique": {"approved": False, "feedback": "does not mention group sizes"}})
    f = pipeline.analyse(session, Q_AGE, llm, 1)
    assert f.label == "Partly verified" and "group sizes" in f.warnings[-1]


# ------------------------------------------------------------ end to end
def test_demo_mode_end_to_end_all_verified(session):
    qs = pipeline.plan(session, None, 4)
    fs = pipeline.run_all(session, qs, None)
    assert len(fs) == 4 and all(f.label == "Verified" for f in fs) and all(f.images for f in fs)


def test_demo_mode_on_any_csv(messy_csv, tmp_path):
    s = pipeline.prepare(messy_csv, str(tmp_path / "m"))
    fs = pipeline.run_all(s, pipeline.plan(s, None, 3), None)
    assert len(fs) == 3 and all(f.label != "Rejected" for f in fs)


def test_ask_your_data(session):
    f = pipeline.ask(session, "Is tobacco different by famhist?", None, 1)
    assert f.columns == ["tobacco", "famhist"] and f.label == "Verified"
    assert "Demo mode can only answer" in pipeline.ask(session, "What is interesting here?", None, 2).issues[0]


def test_reports(session):
    fs = pipeline.run_all(session, pipeline.plan(session, None, 2), None)
    md = markdown(session, fs, "Demo")
    assert "✅ Verified: 2" in md and "```python" in md
    assert "data:image/png;base64," in html_report(session, fs, "Demo")


# ------------------------------------------------------------ Groq adapter
def test_groq_switches_model_on_daily_limit(monkeypatch):
    from dataagent.llm import GroqLLM

    class Msg:
        content = "hello"

    class Resp:
        choices = [type("C", (), {"message": Msg()})()]
        usage = type("U", (), {"total_tokens": 10})()

    calls = []

    def create(model, **kw):
        calls.append(model)
        if model == "openai/gpt-oss-120b":
            raise RuntimeError("Rate limit reached on tokens per day (TPD): Limit 200000, Used 200000")
        return Resp()
    llm = GroqLLM(api_key="gsk_test")
    monkeypatch.setattr(llm._client.chat.completions, "create", create)
    assert llm.call("hi") == "hello" and calls == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert llm.current == "openai/gpt-oss-20b"


def test_titanic_codes_are_categories_and_rows_are_kept(tmp_path):
    s = pipeline.prepare(str(ROOT / "data" / "titanic.csv"), str(tmp_path / "t"))
    assert len(s.df) == 891                                            # no passengers silently deleted
    assert s.profile["columns"]["pclass"]["kind"] == "categorical"     # class 1/2/3 is not a quantity
    qs = pipeline.plan(s, None, 4)
    assert all(not (q["columns"] == ["pclass", "survived"] and "t-test" in q["method"]) for q in qs)


@pytest.mark.parametrize("name", ["titanic", "penguins", "restaurant_tips"])
def test_sample_datasets_all_verified(name, tmp_path):
    desc = (ROOT / "data" / f"{name}_description.txt").read_text()
    s = pipeline.prepare(str(ROOT / "data" / f"{name}.csv"), str(tmp_path / name), desc)
    fs = pipeline.run_all(s, pipeline.plan(s, None, 3), None)
    assert len(fs) == 3 and all(f.label == "Verified" for f in fs)
