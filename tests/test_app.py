"""Click through the Streamlit app (demo mode) with an uploaded CSV, the way a user does.

Regression test: an uploaded file stays in the uploader, and earlier every click was treated as a new
upload, which wiped the questions ("Add a question" made them disappear)."""
import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

WRAPPER = f'''
import runpy, streamlit as st

class Upload:                      # what st.file_uploader returns once a file is chosen
    name = "titanic.csv"
    def __init__(self):
        self.data = open(r"{ROOT / "data" / "titanic.csv"}", "rb").read()
        self.size = len(self.data)
    def getvalue(self):
        return self.data

from streamlit.delta_generator import DeltaGenerator
DeltaGenerator.file_uploader = lambda *a, **k: Upload()   # the app calls it on a column
runpy.run_path(r"{ROOT / "app.py"}", run_name="__main__")
'''


def button(at, label):
    return next(b for b in at.button if b.label.startswith(label))


def test_upload_questions_survive_clicks_and_analysis_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OFFLINE", "1")          # demo mode: no API key or internet needed
    script = tmp_path / "wrapper.py"
    script.write_text(WRAPPER, encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=120).run()
    assert not at.exception
    assert any("891 rows" in e.label for e in at.expander)

    button(at, "Suggest questions").click().run()
    assert len(at.session_state["questions"]) == 3

    button(at, "Add a question").click().run()           # used to wipe everything
    assert len(at.session_state["questions"]) == 4
    boxes = [t for t in at.text_input if t.label.startswith("Question")]
    assert len(boxes) == 4 and boxes[0].value            # suggested questions still shown

    boxes[3].input("Did sex affect survived?").run()     # own question, no columns picked
    assert len(at.session_state["questions"]) == 4

    button(at, "Run analysis on 4").click().run()
    findings = at.session_state["findings"]
    assert len(findings) == 4
    assert findings[3].columns == ["sex", "survived"]    # columns taken from the typed question
    assert all(f.label != "Rejected" for f in findings)

    at.chat_input[0].set_value("Does fare differ by survived?").run()
    assert len(at.session_state["chat"]) == 1
    assert len(at.session_state["findings"]) == 4        # report kept after a follow-up question
