import streamlit as st

st.set_page_config(page_title="Lessons — Chaouat Economics Lab", page_icon="🧠", layout="wide")
st.title("Lessons")
st.caption("Curriculum-first lessons (coming next).")

st.info("Next build: lesson templates + auto-graded quizzes + tutor assignment links.")

st.subheader("Planned lesson tracks")
st.markdown("""
**Micro (Foundations)**
- Opportunity cost, comparative advantage
- Supply/demand, incidence, elasticity
- Externalities and Pigouvian taxes

**Macro (Policy)**
- Inflation measurement
- Unemployment and labor force participation
- Monetary policy transmission
- Fiscal multipliers and automatic stabilizers

**International**
- Exchange rates
- Balance of payments basics
""")

