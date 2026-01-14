import os
import streamlit as st
import os
import streamlit as st

@st.cache_data(ttl=6*60*60, show_spinner=False, max_entries=200)
def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def human_kb(path: str) -> int:
    return int(os.path.getsize(path) / 1024)
# ---------- Page setup ----------
st.set_page_config(page_title="Presentations", page_icon="📑", layout="wide")

# ---------- Style (matches your aesthetic) ----------
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
  --border:#E2E8F0;
}
html, body, * { font-family: 'Montserrat', sans-serif !important; }

.cf-hero{
  background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
  color:#fff;
  padding: 26px 28px;
  border-radius: 20px;
  box-shadow: 0 8px 24px rgba(0,123,167,.25);
  margin: 8px 0 22px 0;
}
.cf-brand{ font-weight: 800; font-size: 40px; letter-spacing:.2px; }
.cf-sub{ margin-top: 8px; opacity:.95; font-size: 14.5px; max-width: 900px; line-height: 1.5; }

.deck-card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: 0 10px 28px rgba(15,23,42,.06);
  height: 100%;
}
.deck-title{ font-weight: 800; font-size: 16px; margin: 8px 0 2px 0; color: var(--text); }
.deck-meta{ color: var(--muted); font-size: 12.5px; margin-bottom: 10px; }
.deck-chip{
  display:inline-block;
  background:#E0F2FE;
  border:1px solid #BAE6FD;
  color:#075985;
  padding: 4px 8px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  margin-top: 6px;
}
.preview-box{
  width: 100%;
  background: #F1F5F9;
  border: 1px dashed #CBD5E1;
  border-radius: 14px;
  padding: 18px 14px;
  text-align: center;
  color: #64748B;
  font-size: 12.5px;
}
.stButton>button, .stDownloadButton>button{
  background: var(--primary) !important;
  color: #fff !important;
  border: 0 !important;
  border-radius: 12px !important;
  padding: 10px 14px !important;
}
.stButton>button:hover, .stDownloadButton>button:hover{
  filter: brightness(0.95);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cf-hero">
  <div class="cf-brand">Presentations</div>
  <div class="cf-sub">
    Curated slide decks and handouts. View in browser or download for classroom use.
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Where your PDFs live ----------
MATERIALS_DIR = "materials"

def list_pdfs(folder: str):
    if not os.path.exists(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".pdf")])

pdfs = list_pdfs(MATERIALS_DIR)

if not pdfs:
    st.warning("No PDFs found. Add files to the 'materials' folder in GitHub.")
    st.stop()

# Optional: a simple search box
q = st.text_input("Search decks", placeholder="Type keywords…").strip().lower()
if q:
    pdfs = [p for p in pdfs if q in p.lower()]

# A nice 3-column gallery
cols = st.columns(3, gap="large")

for i, filename in enumerate(pdfs):
    path = os.path.join(MATERIALS_DIR, filename)
    size_kb = int(os.path.getsize(path) / 1024)

    # For display
    title = filename.replace(".pdf", "").replace("_", " ").replace("-", " ")

    with cols[i % 3]:
        st.markdown('<div class="deck-card">', unsafe_allow_html=True)

        # Preview placeholder (clean + minimal). If you later add thumbnails, replace this area.
        cover_png = os.path.join(MATERIALS_DIR, "covers", filename.replace(".pdf", ".png"))
        cover_jpg = os.path.join(MATERIALS_DIR, "covers", filename.replace(".pdf", ".jpg"))
        cover_jpeg = os.path.join(MATERIALS_DIR, "covers", filename.replace(".pdf", ".jpeg"))

        if os.path.exists(cover_png):
          st.image(cover_png, use_container_width=True)
        elif os.path.exists(cover_jpg):
          st.image(cover_jpg, use_container_width=True)
        elif os.path.exists(cover_jpeg):
          st.image(cover_jpeg, use_container_width=True)
        else:
          st.markdown(f"""
            <div class="preview-box">
              No cover image found<br/>
              <span style="font-size:11px;">Add: materials/covers/{filename.replace(".pdf",".png")}</span>
            </div>
          """, unsafe_allow_html=True)


        # Buttons: View + Download
        c1, c2 = st.columns(2)
        with c1:
            # "View" in Streamlit: show inline embed below when clicked
            view = st.button("View", key=f"view_{filename}")
        with c2:
            with open(path, "rb") as f:
                data = f.read()
            st.download_button(
                "Download",
                data=data,
                file_name=filename,
                mime="application/pdf",
                key=f"dl_{filename}"
            )

        if view:
            # Inline PDF viewer (works in most browsers)
            import base64
            b64 = base64.b64encode(data).decode("utf-8")
            st.markdown(
                f"""
                <iframe
                    src="data:application/pdf;base64,{b64}"
                    width="100%"
                    height="520"
                    style="border:1px solid #E2E8F0;border-radius:14px;margin-top:10px;"
                ></iframe>
                """,
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)
