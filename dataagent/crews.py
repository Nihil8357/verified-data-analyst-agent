"""The CrewAI crews (as in the DataHour project), plus a Critic agent for the Reflection pattern.

  Quest crew    : Business Consultant -> analysis questions (structured output)
  EDA crew      : Data Scientist      -> analysis code, then a plain-language answer
  Critic crew   : Analysis Critic     -> approves or rejects the answer (Reflection)
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")     # must be set before crewai is imported
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import Agent, Crew, Process, Task  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

CONFIG = Path(__file__).resolve().parent / "config"
AGENTS = yaml.safe_load((CONFIG / "agents.yaml").read_text(encoding="utf-8"))
TASKS = yaml.safe_load((CONFIG / "tasks.yaml").read_text(encoding="utf-8"))


class Question(BaseModel):
    question: str
    columns: list[str] = Field(default_factory=list)
    method: str = ""


class QuestionPlan(BaseModel):
    questions: list[Question]


class CriticVerdict(BaseModel):
    approved: bool
    feedback: str = "ok"


def _agent(name: str, llm) -> Agent:
    cfg = AGENTS[name]
    return Agent(role=cfg["role"], goal=cfg["goal"], backstory=cfg["backstory"], llm=llm,
                 allow_delegation=False, verbose=False, max_iter=3)


def _run(agent_name: str, task_name: str, llm, inputs: dict, output_pydantic=None):
    agent = _agent(agent_name, llm)
    cfg = TASKS[task_name]
    task = Task(description=cfg["description"], expected_output=cfg["expected_output"], agent=agent,
                output_pydantic=output_pydantic)
    out = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff(inputs=inputs)
    return out.pydantic if output_pydantic is not None and out.pydantic is not None else out.raw


def plan_questions(llm, data_card: str, how_many: int) -> list[dict]:
    plan = _run("business_consultant", "generate_questions", llm,
                {"data_card": data_card, "how_many": how_many}, QuestionPlan)
    if isinstance(plan, str):
        plan = QuestionPlan.model_validate_json(plan[plan.find("{"):plan.rfind("}") + 1])
    return [q.model_dump() for q in plan.questions]


def write_code(llm, data_card: str, q: dict, feedback: str = "") -> str:
    return _run("data_scientist", "write_code", llm,
                {"data_card": data_card, "question": q["question"], "columns": ", ".join(q.get("columns", [])) or "any",
                 "method": q.get("method") or "your choice", "feedback": feedback})


def interpret(llm, q: dict, output: str, feedback: str = "") -> str:
    return _run("data_scientist", "interpret", llm,
                {"question": q["question"], "output": output[:5000], "feedback": feedback}).strip()


def critique(llm, q: dict, output: str, summary: str, issues: list[str]) -> CriticVerdict:
    v = _run("critic", "critique", llm, {"question": q["question"], "output": output[:5000], "summary": summary,
                                          "issues": "; ".join(issues) or "none"}, CriticVerdict)
    if isinstance(v, str):
        v = CriticVerdict.model_validate_json(v[v.find("{"):v.rfind("}") + 1])
    return v
