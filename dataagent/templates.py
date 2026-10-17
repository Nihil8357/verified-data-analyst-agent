"""Built-in statistical analyses.

Used (1) in Demo mode when no AI model is available, and (2) as a safe fallback when the AI's
code still fails after 3 repair attempts. Each template prints its results and a RESULT line,
so the summary is written from computed numbers only."""
from __future__ import annotations

import json
import re

import pandas as pd

from .profiler import column_kind


def pick_target(df: pd.DataFrame) -> str | None:
    binaries = [c for c in df.columns if column_kind(df[c]) == "binary"]
    for c in reversed(binaries):                      # outcome columns are usually last ('chd', 'churn')
        return c
    return None


def demo_questions(df: pd.DataFrame, n: int = 3) -> list[dict]:
    target = pick_target(df)
    numeric = [c for c in df.columns if column_kind(df[c]) == "numeric"]
    cats = [c for c in df.columns if column_kind(df[c]) in ("categorical", "binary") and c != target]
    qs: list[dict] = []
    if target:
        y = pd.factorize(df[target])[0]
        ranked = sorted(numeric, key=lambda c: -abs(pd.Series(df[c]).corr(pd.Series(y))) if df[c].notna().any() else 0)
        for c in ranked[:2]:
            qs.append({"question": f"Does {c} differ between the {target} groups?", "columns": [c, target],
                       "method": "group comparison (t-test)"})
        for c in cats[:1]:
            qs.append({"question": f"Is {c} associated with {target}?", "columns": [c, target],
                       "method": "chi-square test"})
    if len(numeric) >= 2:
        corr = df[numeric].corr().abs()
        pairs = sorted(((corr.loc[a, b], a, b) for i, a in enumerate(numeric) for b in numeric[i + 1:]), reverse=True)
        if pairs:
            _, a, b = pairs[0]
            qs.append({"question": f"How strongly are {a} and {b} correlated?", "columns": [a, b],
                       "method": "Pearson correlation"})
    for c in numeric:
        qs.append({"question": f"What does the distribution of {c} look like?", "columns": [c], "method": "distribution"})
    return qs[:n]


def template_code(question: dict, df: pd.DataFrame) -> str:
    cols = [c for c in question.get("columns", []) if c in df.columns]
    kinds = [column_kind(df[c]) for c in cols]
    safe = re.sub(r"\W+", "_", "_".join(cols))[:40] or "plot"
    if len(cols) == 2 and set(kinds) <= {"numeric"}:
        a, b = cols
        return f"""import pandas as pd
from scipy import stats
df = pd.read_csv(DATA_PATH)
d = df[[{a!r}, {b!r}]].dropna()
r, p = stats.pearsonr(d[{a!r}], d[{b!r}])
print(f"n = {{len(d)}}")
print(f"Pearson r = {{r:.3f}}")
print(f"p-value = {{p:.4g}}")
ax = d.plot.scatter(x={a!r}, y={b!r}, alpha=0.5, title={a!r} + " vs " + {b!r})
plt.savefig(os.path.join(OUT_DIR, "{safe}.png"), bbox_inches="tight")
print("RESULT " + __import__("json").dumps({{"kind": "corr", "a": {a!r}, "b": {b!r}, "r": round(r, 3), "p": float(p), "n": len(d)}}))
""".replace('__import__("json")', "json").replace("import pandas as pd", "import json\nimport pandas as pd")
    if len(cols) == 2 and "numeric" in kinds:
        num = cols[kinds.index("numeric")]
        grp = cols[1 - kinds.index("numeric")]
        return f"""import json
import pandas as pd
from scipy import stats
df = pd.read_csv(DATA_PATH)
d = df[[{num!r}, {grp!r}]].dropna()
groups = {{str(k): g[{num!r}] for k, g in d.groupby({grp!r})}}
means = {{k: round(float(v.mean()), 2) for k, v in groups.items()}}
for k, v in groups.items():
    print(f"{grp} = {{k}}: n = {{len(v)}}, mean {num} = {{v.mean():.2f}}")
vals = list(groups.values())
if len(vals) == 2:
    t, p = stats.ttest_ind(vals[0], vals[1], equal_var=False)
    test = "Welch t-test"
else:
    t, p = stats.f_oneway(*vals)
    test = "one-way ANOVA"
print(f"{{test}}: statistic = {{t:.3f}}, p-value = {{p:.4g}}")
d.boxplot(column={num!r}, by={grp!r})
plt.suptitle("")
plt.title({num!r} + " by " + {grp!r})
plt.savefig(os.path.join(OUT_DIR, "{safe}.png"), bbox_inches="tight")
print("RESULT " + json.dumps({{"kind": "groups", "num": {num!r}, "grp": {grp!r}, "means": means, "p": float(p), "test": test}}))
"""
    if len(cols) == 2:
        a, b = cols
        return f"""import json
import pandas as pd
from scipy import stats
df = pd.read_csv(DATA_PATH)
tab = pd.crosstab(df[{a!r}], df[{b!r}])
print(tab)
chi2, p, dof, _ = stats.chi2_contingency(tab)
rates = (pd.crosstab(df[{a!r}], df[{b!r}], normalize="index") * 100).round(1)
print(rates)
print(f"Chi-square = {{chi2:.3f}}, dof = {{dof}}, p-value = {{p:.4g}}")
rates.plot.bar(title="% of " + {b!r} + " by " + {a!r})
plt.savefig(os.path.join(OUT_DIR, "{safe}.png"), bbox_inches="tight")
print("RESULT " + json.dumps({{"kind": "chi2", "a": {a!r}, "b": {b!r}, "chi2": round(float(chi2), 3), "p": float(p)}}))
"""
    c = cols[0] if cols else df.select_dtypes("number").columns[0]
    return f"""import json
import pandas as pd
df = pd.read_csv(DATA_PATH)
s = df[{c!r}].dropna()
print(s.describe().round(2))
print(f"skewness = {{s.skew():.2f}}")
s.plot.hist(bins=30, title="Distribution of " + {c!r})
plt.savefig(os.path.join(OUT_DIR, "{safe}.png"), bbox_inches="tight")
print("RESULT " + json.dumps({{"kind": "dist", "c": {c!r}, "mean": round(float(s.mean()), 2), "median": round(float(s.median()), 2), "skew": round(float(s.skew()), 2)}}))
"""


def template_summary(output: str) -> str:
    m = re.search(r"^RESULT (\{.*\})$", output or "", re.M)
    if not m:
        return "The analysis ran, but no structured result was produced."
    r = json.loads(m.group(1))
    sig = lambda p: ("statistically significant" if p < 0.05 else "not statistically significant") + f" (p-value = {p:.4g})"
    if r["kind"] == "corr":
        strength = "weak" if abs(r["r"]) < 0.3 else "moderate" if abs(r["r"]) < 0.6 else "strong"
        return (f"{r['a']} and {r['b']} show a {strength} {'positive' if r['r'] > 0 else 'negative'} correlation "
                f"(Pearson r = {r['r']}, n = {r['n']}), which is {sig(r['p'])}. Correlation does not imply causation.")
    if r["kind"] == "groups":
        parts = ", ".join(f"{k}: {v}" for k, v in r["means"].items())
        return (f"Mean {r['num']} by {r['grp']} is {parts}. The difference ({r['test']}) is {sig(r['p'])}.")
    if r["kind"] == "chi2":
        return (f"The association between {r['a']} and {r['b']} (chi-square = {r['chi2']}) is {sig(r['p'])}. "
                f"See the percentage table above for how the rates differ.")
    return (f"{r['c']} has mean {r['mean']} and median {r['median']}; skewness is {r['skew']}, so the distribution is "
            f"{'roughly symmetric' if abs(r['skew']) < 0.5 else 'right-skewed' if r['skew'] > 0 else 'left-skewed'}.")
