# import streamlit as st

# st.set_page_config(
#     page_title="Gene Prioritization explorer", page_icon=":material/genetics:", layout="centered"
# )


# # Scree-plot mark in the site's palette (ink navy / signal teal / data amber).
# # These hex values are an approximation - swap in the exact ones from the
# # CV site's CSS if you want the two to match exactly.
# LOGO_SVG = """
# <svg width="48" height="48" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
#   <rect x="4"  y="4"  width="10" height="56" rx="2" fill="#1B2A4A"/>
#   <rect x="18" y="20" width="10" height="40" rx="2" fill="#17A2A0"/>
#   <rect x="32" y="34" width="10" height="26" rx="2" fill="#E8A33D"/>
#   <rect x="46" y="44" width="10" height="16" rx="2" fill="#1B2A4A"/>
# </svg>
# """

# col_logo, col_title = st.columns([1, 8])
# with col_logo:
#     st.markdown(LOGO_SVG, unsafe_allow_html=True)
# with col_title:
#     st.title("Gene Prioritization Explorer")
#     st.caption("Probabilistic multi-block PLS with EM estimation")
import streamlit as st

# Custom subtle card styling
st.markdown(
    """
    <style>
    .metric-box {
        background-color: var(--secondary-background-color);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #2E7D32;
        margin-bottom: 1rem;
    }
    .feature-card {
        background-color: var(--secondary-background-color);
        padding: 1.25rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.18);
        height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Hero Header ---
hero_left, hero_right = st.columns([3, 1], gap="medium")

with hero_left:
    st.title("🧬 Gene Prioritization Explorer")
    st.markdown(
        """
        Integrate multi-cohort omics datasets using **Probabilistic Multi-Block Partial Least Squares 
        with Expectation–Maximization (`mbpls-em`)**. Disentangle disease-wide shared signatures from 
        dataset-specific variation to prioritize target genes with high confidence.
        """
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("📊 View Precomputed Showcase", type="primary", use_container_width=True):
            st.switch_page("pages/showcase.py")
    with c2:
        if st.button("📁 Upload Your Datasets", use_container_width=True):
            st.switch_page("pages/uploaddata.py")

with hero_right:
    with st.container(border=True):
        st.caption("FRAMEWORK")
        st.markdown("**Package:** `mbpls-em`")
        st.markdown("**Status:** Research Software")
        st.markdown("**Grant:** SynOD (ZonMw)")

st.divider()

# --- Pipeline Overview ---
st.subheader("Platform Workflow")

col_step1, col_step2, col_step3 = st.columns(3, gap="medium")

with col_step1:
    with st.container(border=True):
        st.markdown("### 1. Data Ingestion")
        st.markdown(
            """
            Upload $K$ heterogeneous omics blocks $(X_k, Y_k)$ measured across different sample cohorts 
            with shared gene features.
            """
        )

with col_step2:
    with st.container(border=True):
        st.markdown("### 2. EM Decomposition")
        st.markdown(
            """
            Estimate shared latent loadings ($W$) and block-specific effects ($P_k$) via orthonormal EM 
            updates.
            """
        )

with col_step3:
    with st.container(border=True):
        st.markdown("### 3. Prioritization")
        st.markdown(
            """
            Extract top-ranked genes contributing most heavily to shared latent components predicting the 
            phenotype response.
            """
        )

st.divider()

# --- Mathematical Framework ---
st.subheader("Methodological Foundation")

math_col, info_col = st.columns([3, 2], gap="large")

with math_col:
    st.markdown(
        """
        For $K$ independent cohorts $(X_k, Y_k)$ where $X_k \in \mathbb{R}^{N_k \times d}$ represents 
        omics features and $Y_k \in \mathbb{R}^{N_k}$ denotes the clinical phenotype:
        """
    )
    st.latex(r"X_k = T_k W^\top + U_k P_k^\top + E_k")
    st.latex(r"Y_k = T_k \beta_k^\top + U_k \phi_k^\top + \varepsilon_k")
    st.markdown(
        """
        * **$W$**: Orthonormal shared loading structure across all cohorts.
        * **$P_k$**: Block-specific structural variation unique to cohort $k$.
        * **$T_k, U_k$**: Shared and specific latent score matrices ($T_k \in \mathbb{R}^{N_k \times r}$, $U_k \in \mathbb{R}^{N_k \times q_k}$).
        * **$E_k, \varepsilon_k$**: Gaussian noise residuals.
        """
    )

with info_col:
    with st.container(border=True):
        st.markdown("#### Key Method Features")
        st.markdown("- **Cohort Invariance:** Handles different sample sizes ($N_k$) per block.")
        st.markdown("- **EM Maximum Likelihood:** Robust estimation under noisy or missing values.")
        st.markdown("- **Strict Orthogonality:** Ensures clear separation between shared and block noise.")

st.divider()

# --- Funding & Citation ---
st.subheader("Project Context & Funding")
f_col1, f_col2 = st.columns([3, 1], gap="medium")

with f_col1:
    st.markdown(
        """
        This tool was developed as part of **Alpha-Synuclein OMICS to identify Drug-targets (SynOD)** 
        (Grant No. [10510062320007](https://projecten.zonmw.nl/en/project/alpha-synuclein-omics-identify-drug-targets-synod)).
        
        **Author:** Rosember Guerra-Urzola  
        **Inspired by:** Said el Bouhaddani et al., *PLOS Computational Biology* (2024).
        """
    )

with f_col2:
    with st.container(border=True):
        st.caption("CODEBASE")
        st.markdown("[GitHub Repo](https://github.com/RosemberGuerra/mbpls-em)")
        st.markdown("License: MIT")