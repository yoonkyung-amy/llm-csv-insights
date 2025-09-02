# LLM CSV Insights

This is a tool that lets you ask questions about CSV files in natural language.  
It combines fast rule-based analytics with an optional LLM fallback **(TinyLLaMA)** for open-ended reasoning.

---

## Features
- Upload any CSV and query it in plain English
- Rule-based engine for precise queries (examples: average, max, min, distinct)
- Fallback to an open-source LLM (TinyLLaMA) for more complex reasoning
- Streamlit UI for interactive demo
- Automatic checks (style + tests run on GitHub)
- Easy to run anywhere with Docker

---

## 🔧 Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate
pip install -r requirements.txt
python -m app.demo --csv sample.csv --q "average price"
```
If you would like to use Streamlit UI,
```
streamlit ui.py
```

## Example screenshot using Steamlit
<img width="590" height="637" alt="image" src="https://github.com/user-attachments/assets/ce90a5c0-bb1d-4788-a69e-8899e5ca7477" />

