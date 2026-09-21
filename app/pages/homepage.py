import streamlit as st

st.set_page_config(
    page_title="Gene Prioritization explorer", page_icon="🧬", layout="centered"
)


# Scree-plot mark in the site's palette (ink navy / signal teal / data amber).
# These hex values are an approximation - swap in the exact ones from the
# CV site's CSS if you want the two to match exactly.
LOGO_SVG = """
<svg width="48" height="48" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <rect x="4"  y="4"  width="10" height="56" rx="2" fill="#1B2A4A"/>
  <rect x="18" y="20" width="10" height="40" rx="2" fill="#17A2A0"/>
  <rect x="32" y="34" width="10" height="26" rx="2" fill="#E8A33D"/>
  <rect x="46" y="44" width="10" height="16" rx="2" fill="#1B2A4A"/>
</svg>
"""

col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown(LOGO_SVG, unsafe_allow_html=True)
with col_title:
    st.title("Gene Prioritization Explorer")
    st.caption("Probabilistic multi-block PLS with EM estimation")
