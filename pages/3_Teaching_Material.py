import os
import streamlit as st

st.set_page_config(page_title="Teaching Material — Chaouat Economics Lab", page_icon="📚", layout="wide")
st.title("Teaching Material")
st.caption("Slide decks and worksheets for tutors and classrooms.")

st.info("Add your PDF decks to a folder named **materials/** at the repo root. This page will list them automatically.")

MATERIALS_DIR = "materials"

def list_pdfs(folder: str):
    if not os.path.exists(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".pdf")])

pdfs = list_pdfs(MATERIALS_DIR)

if not pdfs:
    st.warning("No PDFs found yet. Create the folder **materials/** and add PDF files.")
else:
    st.subheader("Available decks")
    for f in pdfs:
        path = os.path.join(MATERIALS_DIR, f)
        with open(path, "rb") as fp:
            data = fp.read()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{f}**")
            st.caption("Educational use only. If you cite or reuse, keep the methodology constraints.")
        with col2:
            st.download_button(
                label="Download",
                data=data,
                file_name=f,
                mime="application/pdf",
                key=f"dl_{f}",
            )

st.divider()
st.subheader("Suggested structure for a deck")
st.markdown("""
- **1 slide**: Objective (what students will learn)
- **2–4 slides**: Core concept + common mistake
- **1–2 slides**: Worked example
- **1 slide**: Practice prompt
- **1 slide**: Extension question (policy trade-off)
""")

