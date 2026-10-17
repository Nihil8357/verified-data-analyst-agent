"""Load, clean and profile any CSV. Deterministic: this is the agent's ground truth about the data."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CleanResult:
    df: pd.DataFrame
    log: list[str] = field(default_factory=list)


def load_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read the file as CSV.")


def clean(df: pd.DataFrame) -> CleanResult:
    """Basic, explainable cleaning (the 'data cleaning crew' the original project left as TODO)."""
    log: list[str] = []
    df = df.copy()

    new_cols = [re.sub(r"\s+", "_", str(c).strip()) for c in df.columns]
    if new_cols != list(df.columns):
        log.append("Trimmed column names and replaced spaces with underscores.")
    df.columns = new_cols

    # index-like columns: 'Unnamed: 0', 'row.names', 'id' that is 1..n
    for c in list(df.columns):
        is_counter = pd.api.types.is_integer_dtype(df[c]) and df[c].is_monotonic_increasing and df[c].is_unique
        if re.fullmatch(r"(unnamed:?_?\d*|row\.?names|index)", c, re.I) or (c.lower() == "id" and is_counter):
            df = df.drop(columns=c)
            log.append(f"Dropped index-like column '{c}'.")

    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            stripped = df[c].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "NA": np.nan, "?": np.nan})
            numeric = pd.to_numeric(stripped.str.replace(",", "", regex=False), errors="coerce")
            if stripped.notna().sum() and numeric.notna().sum() / stripped.notna().sum() >= 0.95:
                df[c] = numeric
                log.append(f"Converted '{c}' from text to numbers.")
            else:
                df[c] = stripped.where(df[c].notna(), np.nan)

    dups = int(df.duplicated().sum())
    if dups:
        df = df.drop_duplicates().reset_index(drop=True)
        log.append(f"Removed {dups} duplicate rows.")

    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        df = df.drop(columns=empty)
        log.append(f"Dropped empty columns: {', '.join(empty)}.")

    missing = {c: int(n) for c, n in df.isna().sum().items() if n}
    if missing:
        log.append("Missing values kept (not guessed): " + ", ".join(f"{c}={n}" for c, n in missing.items()) + ".")
    if not log:
        log.append("No cleaning needed: no duplicates, index columns or text-encoded numbers found.")
    return CleanResult(df, log)


def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "binary"
    if pd.api.types.is_numeric_dtype(s):
        return "binary" if s.dropna().nunique() == 2 else "numeric"
    n = s.dropna().nunique()
    if n == 2:
        return "binary"
    return "categorical" if n <= max(20, len(s) * 0.05) else "text"


def profile(df: pd.DataFrame) -> dict:
    cols = {}
    for c in df.columns:
        s = df[c]
        info = {"kind": column_kind(s), "missing": int(s.isna().sum()), "unique": int(s.nunique())}
        if info["kind"] == "numeric":
            d = s.describe()
            info.update({k: round(float(d[k]), 3) for k in ("mean", "std", "min", "50%", "max")})
        else:
            info["top"] = {str(k): int(v) for k, v in s.value_counts().head(5).items()}
        cols[c] = info
    return {"rows": len(df), "columns": cols}


def data_card(prof: dict, description: str = "") -> str:
    """Compact text the LLM receives: exact column names, types and ranges, so it cannot invent columns."""
    lines = [f"Rows: {prof['rows']}. Columns ({len(prof['columns'])}):"]
    for c, i in prof["columns"].items():
        if i["kind"] == "numeric":
            extra = f"mean {i['mean']}, min {i['min']}, max {i['max']}"
        else:
            extra = "values " + ", ".join(f"{k} ({v})" for k, v in list(i["top"].items())[:4])
        miss = f", {i['missing']} missing" if i["missing"] else ""
        lines.append(f"- {c} [{i['kind']}]: {extra}{miss}")
    if description.strip():
        lines.append("\nDataset description from the user:\n" + description.strip()[:1500])
    return "\n".join(lines)


def auto_description(prof: dict) -> str:
    """Used when the user gives no description (the original project needed a hand-written metadata file)."""
    kinds = {}
    for c, i in prof["columns"].items():
        kinds.setdefault(i["kind"], []).append(c)
    parts = [f"A table of {prof['rows']} rows."]
    for k in ("numeric", "binary", "categorical", "text"):
        if kinds.get(k):
            parts.append(f"{k.capitalize()} columns: {', '.join(kinds[k])}.")
    return " ".join(parts)
