"""Claim verification: the core accuracy idea of this project.

An LLM writing "strong correlation (r = 0.72)" is only trustworthy if 0.72 was actually computed.
These checks compare the written summary with the real output of the executed code.
They are deterministic, so they can be trusted and tested."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?%?")
PVAL = re.compile(r"\bp(?:[\s_-]?val(?:ue)?)?\s*[=:<]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.I)
SIGNIF = re.compile(r"\b(statistically\s+)?significant(ly)?\b", re.I)
NOT_SIGNIF = re.compile(r"\b(not|no|non|n't|isn't|wasn't|insignificant|lack of)\s+(\w+\s+){0,3}significan", re.I)
P_BOUND = re.compile(r"\bp(?:[\s_-]?val(?:ue)?s?)?\s*(?:<|≤|<=|below|under|less than)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.I)
THRESHOLDS = {"0.05", "0.01", "0.001", "0.1", "5%", "1%", "95%", "99%"}
CAUSAL = re.compile(r"\b(causes?|caused|leads? to|results? in|due to|because of|drives?|impact of .{0,20} on)\b", re.I)


@dataclass
class Verdict:
    issues: list[str] = field(default_factory=list)      # must be fixed
    warnings: list[str] = field(default_factory=list)    # shown to the reader
    checked_numbers: int = 0
    matched_numbers: int = 0

    @property
    def label(self) -> str:
        if self.issues:
            return "Rejected"
        return "Partly verified" if self.warnings else "Verified"


def _to_float(tok: str) -> float | None:
    try:
        return float(tok.replace(",", "").rstrip("%"))
    except ValueError:
        return None


def output_numbers(output: str) -> list[float]:
    vals = []
    for tok in NUM.findall(output or ""):
        v = _to_float(tok)
        if v is not None:
            vals.append(v)
    return vals


def number_supported(tok: str, values: list[float]) -> bool:
    """A claimed number is supported if it equals an output value after rounding to the same
    number of decimals (0.312 printed -> '0.31' claimed is fine), or is that value as a percentage."""
    claim = _to_float(tok)
    if claim is None:
        return True
    decimals = len(tok.rstrip("%").split(".")[1]) if "." in tok else 0
    for v in values:
        for cand in (v, v * 100) if tok.endswith("%") or abs(claim) > 1 >= abs(v) else (v,):
            if round(cand, decimals) == round(claim, decimals):
                return True
            if decimals == 0 and abs(cand) >= 10 and abs(cand - claim) / abs(cand) < 0.005:
                return True
    return False


def verify(summary: str, output: str, question: str = "", columns: list[str] | None = None,
           images: list[str] | None = None, need_plot: bool = False) -> Verdict:
    v = Verdict()
    values = output_numbers(output)
    ignore = set(NUM.findall(question or "")) | {str(c) for c in (columns or [])}

    # 1a. 'p < 0.001' style bounds: some printed p-value must be below the bound
    pvals = [float(m) for m in PVAL.findall(output or "")]
    bounds = set()
    for b in P_BOUND.findall(summary or ""):
        bounds.add(b)
        if pvals and min(pvals) >= float(b):
            v.issues.append(f"Says p < {b} but the smallest printed p-value is {min(pvals):g}.")

    # 1b. every other number in the summary must come from the output
    for tok in NUM.findall(summary or ""):
        if tok in bounds or tok in THRESHOLDS:
            continue
        bare = tok.rstrip("%")
        if tok in ignore or bare in ignore:
            continue
        if re.fullmatch(r"\d", bare):                   # single digits: list markers, counts like '2 groups'
            continue
        if re.fullmatch(r"(19|20)\d\d", bare):           # years
            continue
        v.checked_numbers += 1
        if number_supported(tok, values):
            v.matched_numbers += 1
        else:
            v.issues.append(f"The number {tok} does not appear in the code output.")

    # 2. significance claims must match the printed p-value
    says_sig = bool(SIGNIF.search(summary or ""))
    says_not = bool(NOT_SIGNIF.search(summary or ""))
    if says_sig and not says_not:
        if not pvals:
            v.issues.append("Claims a significant result but the output prints no p-value.")
        elif min(pvals) >= 0.05:
            v.issues.append(f"Claims significance but the smallest printed p-value is {min(pvals):g} (not < 0.05).")
    if says_not and pvals and max(pvals) < 0.05 and not says_sig:
        v.issues.append(f"Says 'not significant' but all printed p-values are below 0.05.")

    # 3. causal wording on an observational analysis
    m = CAUSAL.search(summary or "")
    if m and not re.search(r"cannot|can't|does not|doesn't|not imply|no evidence", summary[max(0, m.start() - 60):m.end() + 60], re.I):
        v.warnings.append(f"Uses causal wording ('{m.group(0)}'); this analysis only shows association.")

    # 4. columns written in backticks must exist
    if columns:
        known = {c.lower() for c in columns}
        for name in re.findall(r"`([A-Za-z_][\w.]*)`", summary or ""):
            if name.lower() not in known and not re.match(r"(df|data|pd|np|plt|sns)$", name):
                v.issues.append(f"Mentions column `{name}`, which is not in the dataset.")

    if need_plot and not images:
        v.warnings.append("No chart was produced for this question.")
    if v.checked_numbers == 0 and values:
        v.warnings.append("The summary quotes no numbers from the output; add the key figures.")
    return v
