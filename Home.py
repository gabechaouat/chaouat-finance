import streamlit as st

st.set_page_config(page_title="Chaouat Economics Lab", page_icon="📘", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
:root{
  --primary:#007BA7;
  --accent:#00B0FF;
  --bg:#F7FAFC;
  --card:#FFFFFF;
  --text:#0F172A;
  --muted:#64748B;
}
html, body, * { font-family: 'Montserrat', sans-serif !important; }
.cf-hero{
  background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
  color: #fff;
  padding: 34px 36px;
  border-radius: 20px;
  box-shadow: 0 8px 24px rgba(0,123,167,.25);
  margin: 8px 0 18px 0;
}
.cf-brand{ font-weight: 800; font-size: 44px; letter-spacing: .3px; }
.cf-sub{ margin-top: 10px; opacity: .95; font-size: 16px; line-height: 1.5; max-width: 900px; }
.cf-card{
  background: var(--card);
  border: 1px solid #E2E8F0;
  border-radius: 18px;
  padding: 18px 18px;
  box-shadow: 0 8px 30px rgba(15,23,42,.06);
}
small { color: var(--muted); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cf-hero">
  <div class="cf-brand">Chaouat Economics Lab</div>
  <div class="cf-sub">
    A tutoring-first economics and economic policy platform. Interactive policy labs, lesson plans, and teaching decks—
    built with methodological transparency and explicit assumptions.
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown('<div class="cf-card"><h3>Policy Lab</h3><p>Interactive economic policy experiments (monetary policy, fiscal multipliers, debt dynamics) with explicit assumptions.</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="cf-card"><h3>Teaching Material</h3><p>Slide decks and worksheets designed for tutors and classrooms. Structured, printable, and reusable.</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="cf-card"><h3>Tools (Appendix)</h3><p>Market data tools for teaching risk and uncertainty. Not investment advice; strictly pedagogical.</p></div>', unsafe_allow_html=True)

st.markdown("")
st.info("Start with **Policy Lab** (left menu), then use **Teaching Material** to run sessions. Tools remain available under **Tools** as an appendix.")
st.caption("© Chaouat Economics Lab · Built with Python/Streamlit · Educational use only")
