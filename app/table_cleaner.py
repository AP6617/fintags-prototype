from __future__ import annotations
import pandas as pd

def _dedupe(cols):
    seen = {}
    out = []
    for c in cols:
        name = (str(c) or "col").strip()
        if name not in seen:
            seen[name] = 1
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
    return out

def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    head = df.iloc[0].tolist()
    if sum(1 for x in head if str(x).strip()) >= 2:
        df = df.copy()
        df.columns = _dedupe([str(x).strip() for x in head])
        df = df.iloc[1:].reset_index(drop=True)
    else:
        df = df.copy()
        df.columns = _dedupe([f"col_{i}" for i in range(df.shape[1])])
    df = df.dropna(how="all", axis=1)
    df = df.replace("", pd.NA).dropna(how="all", axis=0).fillna("")
    return df
