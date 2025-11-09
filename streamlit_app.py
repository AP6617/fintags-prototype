# streamlit_app.py
from __future__ import annotations

import io
import os
API = os.getenv("FINTAGS_API", "http://127.0.0.1:8000")

import time
import json
import base64
import pandas as pd
import streamlit as st
import requests
from collections import Counter

# -------------------------
# CONFIG
# -------------------------
API_BASE = os.getenv("FINTAGS_API", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="FinTags AI",
    page_icon="🧾",
    layout="wide",
)

# -------------------------
# LIGHT THEME TWEAKS (to mimic your UI)
# -------------------------
st.markdown(
    """
    <style>
      .big-title { font-size: 40px; font-weight: 800; letter-spacing: .2px; }
      .subtle { color:#667085; }
      .pill { display:inline-flex; align-items:center; gap:.5rem; background:#EEF2FF; color:#3538CD;
              padding:.45rem .75rem; border-radius:999px; font-weight:600; }
      .card { border:1px solid #EAECF0; border-radius:14px; padding:20px; background:white; }
      .metric { font-size:40px; font-weight:800; }
      .metric-label { color:#667085; font-weight:600; }
      .good-btn { background:#16a34a; color:white; padding:.6rem 1rem; border-radius:10px; font-weight:700; }
      .blue-btn { background:#1D4ED8; color:white; padding:.6rem 1rem; border-radius:10px; font-weight:700; }
      .icon { font-weight:900; margin-right:.4rem; }
      .tabcont { padding-top:.6rem; }
      .table thead tr th { font-weight:700; }
      .chip { background:#EEF2FF; color:#3538CD; padding:.35rem .7rem; border-radius:999px; }
      .trend { background:#E7F8ED; color:#067647; padding:.2rem .6rem; border-radius:999px; font-weight:700; }
      .del { cursor:pointer; color:#667085; }
      .del:hover { color:#ef4444; }
      .kw-chip { background:#EEF2FF; color:#3538CD; padding:.45rem .8rem; border-radius:999px; display:inline-flex; align-items:center; gap:.5rem; margin:.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# SESSION STATE
# -------------------------
if "result" not in st.session_state:
    st.session_state.result = None       # API /process/upload response
if "df" not in st.session_state:
    st.session_state.df = None           # main table dataframe
if "csv_bytes" not in st.session_state:
    st.session_state.csv_bytes = None
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "keywords" not in st.session_state:
    st.session_state.keywords = None     # dict from backend (concept -> list of terms)
if "process_time" not in st.session_state:
    st.session_state.process_time = None


def _get(url: str, **kwargs):
    r = requests.get(url, timeout=120, **kwargs)
    r.raise_for_status()
    return r


def _post(url: str, **kwargs):
    r = requests.post(url, timeout=600, **kwargs)
    r.raise_for_status()
    return r


def _download_to_bytes(filename: str) -> bytes:
    resp = _get(f"{API_BASE}/download/{filename}")
    return resp.content


def _load_keywords():
    try:
        data = _get(f"{API_BASE}/config/keywords").json()
        st.session_state.keywords = data.get("keywords", {})
    except Exception as e:
        st.error(f"Failed to load keywords: {e}")


def _display_header():
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown('<div class="big-title">FinTags AI</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtle">Automated Financial Metrics Extraction & Analysis</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.write("")
        st.write("")
        st.markdown('<div class="subtle" style="text-align:right;">Powered by FinBERT & AI</div>', unsafe_allow_html=True)


def _metric_card(label: str, value: str | int):
    st.markdown(f"""
        <div class="card">
          <div class="metric">{value}</div>
          <div class="metric-label">{label}</div>
        </div>
    """, unsafe_allow_html=True)


def _render_stat_row(df: pd.DataFrame):
    cols = st.columns([1,1,1,1])
    with cols[0]:
        _metric_card("Metrics Found", len(df))
    with cols[1]:
        pages = sorted(set(df["Page"].astype(str).tolist())) if "Page" in df else ["—"]
        _metric_card("Pages Analyzed", len(pages))
    with cols[2]:
        concepts = sorted(set(df["Concept"])) if "Concept" in df else []
        _metric_card("Concepts Identified", len(concepts))
    with cols[3]:
        t = f"{st.session_state.process_time}s" if st.session_state.process_time else "—"
        _metric_card("Processing Time", t)


def _render_table(df: pd.DataFrame):
    # order columns exactly like your screenshot
    show = df.copy()
    # Pretty confidence as %
    if "Confidence" in show:
        show["Confidence"] = (show["Confidence"].astype(float) * 100).round().astype(int).astype(str) + "%"
    # trend pill
    if "Trend" in show:
        show["Trend"] = show["Trend"].fillna("").apply(lambda x: f'<span class="trend">{x}</span>' if x else "")
    # page pretty
    if "Page" in show:
        show["Page"] = show["Page"].apply(lambda x: f"Page {int(x)}" if pd.notnull(x) else "—")

    order = ["Concept", "Value", "Trend", "Page", "Confidence"]
    for c in order:
        if c not in show.columns:
            show[c] = ""
    show = show[order]

    # render as html table (for the green trend pills)
    st.markdown(
        show.to_html(escape=False, index=False, classes="table"),
        unsafe_allow_html=True
    )


def _render_downloads():
    c1, c2 = st.columns([1,1])
    with c1:
        if st.session_state.csv_bytes:
            st.download_button(
                "⬇️  Download CSV",
                data=st.session_state.csv_bytes,
                file_name=(st.session_state.result or {}).get("csv", "fintags.csv"),
                mime="text/csv",
                type="primary",
                use_container_width=False,
            )
    with c2:
        if st.session_state.pdf_bytes:
            st.download_button(
                "⬇️  Download Highlighted PDF",
                data=st.session_state.pdf_bytes,
                file_name=(st.session_state.result or {}).get("highlighted_pdf", "highlighted.pdf"),
                mime="application/pdf",
                type="primary",
                use_container_width=False,
            )


# -------------------------
# UI HEADER
# -------------------------
_display_header()
st.write("")

# -------------------------
# TABS (match your screenshots)
# -------------------------
tabs = st.tabs(["📤 Upload & Process", "🧾 Analysis Results", "📊 Analytics", "⚙️ Keywords"])

# --------------------------------------------------------
# TAB 1 — Upload & Process
# --------------------------------------------------------
with tabs[0]:
    st.markdown('<div class="tabcont">', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload Financial Report")
    up = st.file_uploader(" ", type=["pdf"], label_visibility="collapsed")

    analyze = st.button("Analyze Document", type="primary")
    if analyze:
        if not up:
            st.warning("Please upload a PDF first.")
        else:
            try:
                t0 = time.time()
                files = {"file": (up.name, up.read(), "application/pdf")}
                res = _post(f"{API_BASE}/process/upload", files=files).json()

                # store result, load CSV into df, fetch files for downloads
                st.session_state.result = res
                st.session_state.process_time = res.get("processing_time", max(1, int(time.time() - t0)))

                # CSV
                csv_name = res.get("csv")
                csv_bytes = _download_to_bytes(csv_name) if csv_name else None
                st.session_state.csv_bytes = csv_bytes
                df = pd.read_csv(io.BytesIO(csv_bytes)) if csv_bytes else None
                st.session_state.df = df

                # PDF
                pdf_name = res.get("highlighted_pdf")
                st.session_state.pdf_bytes = _download_to_bytes(pdf_name) if pdf_name else None

                st.success("File processed successfully.")
                st.experimental_rerun()  # simple refresh so the other tabs see data

            except Exception as e:
                st.error(f"Processing failed: {e}")

    st.markdown('</div></div>', unsafe_allow_html=True)

# --------------------------------------------------------
# TAB 2 — Analysis Results
# --------------------------------------------------------
with tabs[1]:
    st.markdown('<div class="tabcont">', unsafe_allow_html=True)
    st.subheader("Analysis Results")

    if st.session_state.df is None:
        st.info("Upload a PDF on the **Upload & Process** tab to see results.")
    else:
        # stat cards
        _render_stat_row(st.session_state.df)
        st.write("")
        # buttons
        _render_downloads()
        st.write("")
        # table
        _render_table(st.session_state.df)

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# TAB 3 — Analytics (simple frequency bars)
# --------------------------------------------------------
with tabs[2]:
    st.markdown('<div class="tabcont">', unsafe_allow_html=True)
    st.subheader("Financial Metrics Analytics")

    df = st.session_state.df
    if df is None:
        st.info("Upload a PDF first.")
    else:
        counts = df["Concept"].value_counts().rename_axis("Concept").reset_index(name="Count")
        st.bar_chart(counts.set_index("Concept"))
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# TAB 4 — Keywords (show only canonical names; backend keeps synonyms)
# --------------------------------------------------------
with tabs[3]:
    st.markdown('<div class="tabcont">', unsafe_allow_html=True)
    st.subheader("Financial Keywords Management")

    if st.session_state.keywords is None:
        _load_keywords()

    # Add box
    cA, cB = st.columns([6,1])
    with cA:
        new_kw = st.text_input("Add new financial concept…", label_visibility="collapsed",
                               placeholder="Add new financial concept…")
    with cB:
        if st.button("+  Add", use_container_width=True):
            if not new_kw.strip():
                st.warning("Type a concept name.")
            else:
                try:
                    # Tell backend to add the concept and auto-expand synonyms there.
                    payload = {
                        "add": {new_kw.strip(): [new_kw.strip()]},
                        "auto_synonyms": True
                    }
                    _post(f"{API_BASE}/config/keywords", json=payload)
                    _load_keywords()
                    st.success(f"Added: {new_kw.strip()}")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Add failed: {e}")

    st.write("")
    st.caption("Active Keywords")

    # Show ONLY canonical names (keys), not the synonym list
    if st.session_state.keywords:
        # render chips
        cols = st.columns(4)
        i = 0
        removed = None
        for concept in sorted(st.session_state.keywords.keys()):
            with cols[i % 4]:
                st.markdown(
                    f'<div class="kw-chip">{concept}'
                    f' <span class="del" id="del-{concept}">🗑</span></div>',
                    unsafe_allow_html=True
                )
                # Give each a real button (same row feel)
                if st.button("Delete", key=f"delbtn::{concept}", help=f"Delete {concept}"):
                    removed = concept
            i += 1

        if removed:
            try:
                payload = {"remove_concepts": [removed]}
                _post(f"{API_BASE}/config/keywords", json=payload)
                _load_keywords()
                st.success(f"Removed: {removed}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")
    else:
        st.info("No keywords loaded.")
    st.markdown('</div>', unsafe_allow_html=True)
