import pandas as pd
from typing import Dict, Any

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def profile_schema(df: pd.DataFrame) -> Dict[str, Any]:
    summary = {}
    for col in df.columns:
        s = df[col]
        summary[col] = {
            "dtype": str(s.dtype),
            "nulls": int(s.isna().sum()),
            "unique": int(s.nunique()),
            "example": None if s.dropna().empty else s.dropna().iloc[0]
        }
    return {"rows": len(df), "cols": len(df.columns), "columns": summary}
