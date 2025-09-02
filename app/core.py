from typing import Optional
import pandas as pd
import re

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False


class CSVLlmAssistant:

    _AGG_ALIASES = {
        "max": {"max", "maximum", "largest", "highest"},
        "min": {"min", "minimum", "smallest", "lowest"},
        "mean": {"mean", "average", "avg"},
        "sum": {"sum", "total"},
        "median": {"median", "middle"},
        "std": {"std", "stdev", "standard deviation"},
        "count_distinct": {"distinct", "unique"},
        "describe": {"describe", "summary", "stats"},
    }

   # regular expression to detect keywords
    _AGG_PATTERN = re.compile(
        r"\b(" +
        r"|".join(
            [re.escape(x) for xs in _AGG_ALIASES.values() for x in xs]
        ) +
        r")\b\s*(?:of\s*)?(?P<col>[A-Za-z0-9_\-\s]+?)\s*[?\.]?\s*$",
        re.IGNORECASE,
    )

    def __init__(self, df: pd.DataFrame, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.df = df
        self.model_name = model_name
        self._model = None
        self._tok = None
        if _HF_AVAILABLE:
            try:
                self._tok = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name, dtype=getattr(torch, "float16", None)
                )
            except Exception:
                self._model = None
                self._tok = None

    # helps to answer in offline
    def _resolve_col(self, raw: str) -> Optional[str]:
        target = raw.strip().strip("? .").lower()
        for c in self.df.columns:
            if c.lower() == target:
                return c
        return None

    def _is_numeric(self, col: str) -> bool:
        return pd.api.types.is_numeric_dtype(self.df[col])

    # rule-based answers
    def _try_rule_based(self, question: str) -> Optional[str]:
        q = question.strip().lower()
        df = self.df

        # columns
        if any(k in q for k in ["column", "columns", "schema", "fields"]):
            return f"Columns: {list(df.columns)}"

        # Row count
        if any(k in q for k in ["row count", "how many rows", "number of rows", "count rows", "rows?"]):
            return f"Row count: {len(df)}"

        # Aggregations
        m = self._AGG_PATTERN.search(question)
        if m:
            agg_token = m.group(1).lower()
            col_name = self._resolve_col(m.group("col"))
            if not col_name:
                return f"A column '{m.group('col').strip()}' was not found. See the availables: {list(df.columns)}"

            # parse to the keyword
            keyword = None
            for key, aliases in self._AGG_ALIASES.items():
                if agg_token in aliases:
                    keyword = key
                    break

            if keyword in {"max", "min", "mean", "sum", "median", "std"}:
                if not self._is_numeric(col_name):
                    return f"Column '{col_name}' is not numeric; cannot compute {keyword}."
                series = df[col_name]
                if keyword == "max":
                    return f"max({col_name}) = {series.max()}"
                if keyword == "min":
                    return f"min({col_name}) = {series.min()}"
                if keyword == "mean":
                    return f"mean({col_name}) = {series.mean()}"
                if keyword == "sum":
                    return f"sum({col_name}) = {series.sum()}"
                if keywordn == "median":
                    return f"median({col_name}) = {series.median()}"
                if keyword == "std":
                    return f"std({col_name}) = {series.std()}"

            if keyword == "count_distinct":
                n = df[col_name].nunique(dropna=True)
                sample_vals = df[col_name].dropna().unique().tolist()[:10]
                return f"distinct({col_name}) = {n}; first values: {sample_vals}"

            if keyword == "describe":
                s = df[col_name]
                info = {
                    "dtype": str(s.dtype),
                    "nulls": int(s.isna().sum()),
                    "unique": int(s.nunique()),
                }
                if self._is_numeric(col_name):
                    info.update({
                        "min": s.min(),
                        "max": s.max(),
                        "mean": s.mean(),
                        "median": s.median(),
                    })
                return f"describe({col_name}) = {info}"

        return None

    # LLM
    def _context(self, head_n: int = 5) -> str:
        head = self.df.head(head_n).to_csv(index=False)
        cols = ", ".join(list(self.df.columns))
        return (
            "You are a helpful data analyst and your job is to answer questions about a CSV. "
            f"Columns: {cols} "
            f"Sample (first {head_n} rows): {head} "
            "If the answer requires calculations, show steps briefly. And add an explanation why you think that is the answer even though it is not explicitly asked."
        )
    
    def _ask_llm(self, prompt: str) -> str:
        if self._model is None or self._tok is None:
            raise RuntimeError("LLM model not available. Install transformers + model weights or rely on rule-based answers.")
        device = "cuda" if hasattr(self._model, "to") and torch.cuda.is_available() else "cpu"
        self._model.to(device)
        inputs = self._tok(prompt, return_tensors="pt").to(device)
        out = self._model.generate(**inputs, max_new_tokens=200)
        out_text = self._tok.decode(out[0], skip_special_tokens=True)

        if "Answer:" in out_text:
            out_text = out_text.split("Answer:", 1)[1].strip()

        return out_text
        
    def answer(self, question: str) -> str:
        rb = self._try_rule_based(question)
        if rb is not None:
            return f"**Q:** {question}\n\n[Rule-based] {rb}"
        prompt = self._context() + "\nQuestion: " + question + " Why?\nAnswer:"
        try:
            return f"**Q:** {question}\n\n[LLM] {self._ask_llm(prompt)}"
        except Exception as e:
            return f"**Q:** {question}\n\n(LLM unavailable) Error: {e}"

