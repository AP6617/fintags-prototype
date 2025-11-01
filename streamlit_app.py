import os
import json
import requests
import pandas as pd
import streamlit as st

# -----------------------------
# Page / Backend Config
# -----------------------------
st.set_page_config(page_title="FinTags – UI", layout="wide")

# Robust backend URL discovery (env -> secrets -> default)
BACKEND = os.getenv("FINTAGS_BACKEND_URL")
if not BACKEND:
    try:
        BACKEND = st.secrets["backend_url"]
    except Exception:
        BACKEND = "http://127.0.0.1:8000"

# -----------------------------
# HTTP helpers
# -----------------------------
def _get_keywords():
    try:
        r = requests.get(f"{BACKEND}/config/keywords", timeout=10)
        r.raise_for_status()
        return r.json().get("keywords", {})
    except Exception as e:
        st.error(f"Failed to load keywords: {e}")
        return {}

def _post_keywords(payload: dict):
    try:
        r = requests.post(f"{BACKEND}/config/keywords", json=payload, timeout=20)
        r.raise_for_status()
        return r.json().get("keywords", {})
    except Exception as e:
        st.error(f"Failed to update keywords: {e}")
        return None

def _process_pdf(uploaded_file):
    files = {
        "file": (uploaded_file.name, uploaded_file.getbuffer(), uploaded_file.type or "application/pdf")
    }
    r = requests.post(f"{BACKEND}/process/upload", files=files, timeout=120)
    r.raise_for_status()
    return r.json()

def _download_bytes(pathname: str):
    r = requests.get(f"{BACKEND}/download/{pathname}", timeout=60)
    r.raise_for_status()
    return r.content

# -----------------------------
# Styles (chips + layout)
# -----------------------------
st.markdown(
    """
    <style>
    .chip-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .chip {
        position: relative;
        display: inline-flex;
        align-items: center;
        padding: 10px 14px;
        border-radius: 18px;
        background: #E8F3FF;         /* soft blue bg */
        color: #0B5CAD;              /* deep blue text */
        font-weight: 600;
        font-size: 13px;
        line-height: 1;
        user-select: none;
    }
    .chip .closebtn{
        position: absolute;
        top: -6px;
        right: -6px;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: #0B5CAD;
        color: white;
        font-size: 12px;
        line-height: 18px;
        text-align: center;
        cursor: pointer;
    }
    .chip .closebtn:hover{ opacity: 0.9; }
    .chip-title { padding-right: 2px; }
    .section-box { padding: 14px; border: 1px solid #e6e6e6; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar (Reset to defaults)
# -----------------------------
with st.sidebar:
    st.markdown("### FinTags Controls")
    st.write("Manage **keywords**, upload PDFs, preview results, and download outputs.")
    if st.button("Reset keywords to defaults", type="primary", use_container_width=True):
        # Clear all, then rely on backend defaults on next fetch
        updated = _post_keywords({"clear_defaults": True, "add": {}})
        # After clearing, fetch again (backend is expected to restore defaults on GET)
        st.session_state.pop("keywords", None)
        st.rerun()

# -----------------------------
# Session bootstrap
# -----------------------------
if "keywords" not in st.session_state:
    st.session_state.keywords = _get_keywords()
if "resp" not in st.session_state:
    st.session_state.resp = None

# -----------------------------
# Title
# -----------------------------
st.markdown("## FinTags – Financial Tagging UI")
st.caption("Upload → Tag → Preview → Download")

# -----------------------------
# Keyword Manager (chips show ONLY concepts, not synonyms)
# -----------------------------
st.markdown("### Keyword Manager")
with st.container():
    st.markdown('<div class="section-box">', unsafe_allow_html=True)

    # Show concept chips horizontally (no synonyms listed)
    st.markdown('<div class="chip-wrap">', unsafe_allow_html=True)
    for concept in list(st.session_state.keywords.keys()):
        # Render chip
        st.markdown(
            f'''
            <div class="chip">
                <span class="chip-title">{concept}</span>
            </div>
            ''',
            unsafe_allow_html=True
        )
        # Small remove button next to the chip (Streamlit needs a real widget to capture clicks)
        # We'll render a tiny button right after each chip; visually it's close enough to "top-right"
        if st.button("×", key=f"rm_concept_{concept}"):
            new_kw = _post_keywords({"remove_concepts": [concept]})
            if new_kw is not None:
                st.session_state.keywords = new_kw
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**Add a new concept (synonyms are applied automatically in backend)**")
    c1, c2 = st.columns([1, 2])
    with c1:
        new_concept = st.text_input("Concept", placeholder="e.g., Stock Price")
    with c2:
        new_terms = st.text_input("Seed terms (optional, comma-separated)", placeholder="e.g., stock price")

    if st.button("Add / Merge", type="primary"):
        if not new_concept.strip():
            st.warning("Please type a concept name.")
        else:
            payload = {"add": {new_concept.strip(): []}, "auto_synonyms": True}
            # if user gave seed terms, add them (backend will auto-expand)
            if new_terms.strip():
                terms = [t.strip() for t in new_terms.split(",") if t.strip()]
                payload = {"add": {new_concept.strip(): terms}, "auto_synonyms": True}
            new_kw = _post_keywords(payload)
            if new_kw is not None:
                st.session_state.keywords = new_kw
                st.success(f"Concept '{new_concept.strip()}' added.")
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Upload & Process
# -----------------------------
st.markdown("### Upload & Process")
with st.container():
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    up = st.file_uploader("Upload a financial PDF", type=["pdf"], accept_multiple_files=False)
    go = st.button("Submit", type="primary")
    if go and up is not None:
        with st.spinner("Processing…"):
            try:
                resp = _process_pdf(up)
                st.session_state.resp = resp
                st.success("Done.")
            except Exception as e:
                st.error(f"Processing failed: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Preview Sentences
# -----------------------------
st.markdown("### Preview Sentences")
with st.container():
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    if st.session_state.resp:
        preview = st.session_state.resp.get("preview_sentences", [])
        if preview:
            for s in preview:
                st.markdown("- " + s)
        else:
            st.caption("No preview sentences returned.")
    else:
        st.caption("Upload a PDF to see preview sentences.")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Detected Tables (if backend returns them)
# -----------------------------
st.markdown("### Detected Tables")
with st.container():
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    if st.session_state.resp and "tables_preview" in st.session_state.resp:
        tprev = st.session_state.resp["tables_preview"] or []
        if not tprev:
            st.caption("No tables detected.")
        else:
            # Combine rows from each table into small dataframes
            for i, tb in enumerate(tprev, start=1):
                st.markdown(f"**Table {i} – Page {tb.get('page', '?')}**")
                rows = tb.get("rows", [])
                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.caption("Empty table.")
    else:
        st.caption("Your backend is not returning table previews yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Downloads
# -----------------------------
st.markdown("### Downloads")
with st.container():
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    if st.session_state.resp:
        csv_name = st.session_state.resp.get("csv")
        pdf_name = st.session_state.resp.get("highlighted_pdf")

        c1, c2 = st.columns(2)
        with c1:
            if csv_name:
                try:
                    data = _download_bytes(csv_name)
                    st.download_button(
                        label="Download CSV",
                        data=data,
                        file_name=csv_name,
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"CSV download failed: {e}")
            else:
                st.caption("No CSV yet.")

        with c2:
            if pdf_name:
                try:
                    data = _download_bytes(pdf_name)
                    st.download_button(
                        label="Download Highlighted PDF",
                        data=data,
                        file_name=pdf_name,
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"PDF download failed: {e}")
            else:
                st.caption("No PDF yet.")
    else:
        st.caption("Process a PDF to enable downloads.")
    st.markdown('</div>', unsafe_allow_html=True)
