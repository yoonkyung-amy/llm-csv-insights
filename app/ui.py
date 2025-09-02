import streamlit as st
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.core import CSVLlmAssistant

st.set_page_config(page_title="LLM CSV Insights", layout="wide")

st.title("LLM CSV Insights")
st.write("Upload a CSV and ask questions in natural language!")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview", df.head())

    bot = CSVLlmAssistant(df)

    q = st.text_input("Ask a question about your data:")
    if q:
        with st.spinner("Thinking... generating answer..."):
            answer = bot.answer(q)
        st.session_state.history.append(answer)

    # Show history
    if st.session_state.history:
        st.write("### Chat History")
        for msg in st.session_state.history:
            st.markdown(msg)
            st.markdown("---")