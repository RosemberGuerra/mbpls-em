import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mbpls_em.estimators import MBPLS_EM
from mbpls_em.simulate import generate_gene_pool, generate_multiblock_mbpls

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


with st.sidebar:
    st.header("Data settings")
    k = st.selectbox("Blocks (K)", [2, 3])
    d = st.number_input("Genes (d)", min_value=10, max_value=1000, value=100, step=10)
    n = st.slider("Samples per block", min_value=20, max_value=300, value=100, step=10)
    r = st.slider("Shared rank (r)", min_value=1, max_value=10, value=5)
    q = st.slider("Specific rank (q)", min_value=1, max_value=20, value=10)
    sig2e = st.slider(
        "Noise (blocks)", min_value=0.0, max_value=1.0, value=0.15, step=0.05
    )
    sig2eps = st.slider(
        "Noise (effects)", min_value=0.0, max_value=1.0, value=0.12, step=0.05
    )
    seed = st.number_input("Seed", value=10123, step=1)
    fit_clicked = st.button("Fit model", use_container_width=True)

if fit_clicked:
    with st.spinner("Fitting model..."):
        # MVP simplification: same sample count and specific rank applied to
        # every block, even though generate_multiblock_mbpls supports a
        # different value per block via N_list / q_list.
        N_list = [n] * k
        q_list = [q] * k

        data, params, latents = generate_multiblock_mbpls(
            K=k,
            N_list=N_list,
            d=d,
            r=r,
            q_list=q_list,
            sig2e=sig2e,
            sig2eps=sig2eps,
            seed=seed,
        )
        gene_names = generate_gene_pool(d)

        params_fit, history = MBPLS_EM(data=data, r=r, q_list=q_list)

        W = params_fit["W"]
        P = params_fit["P"]
        beta = params_fit["beta"]
        phi = params_fit["phi"]

        shared_effect = [W @ beta_k.T for beta_k in beta]
        specific_effect = [P_k @ phi_k.T for phi_k, P_k in zip(phi, P)]

        df_shared = pd.DataFrame(data=np.hstack(shared_effect), index=gene_names)
        df_shared["total_shared"] = np.linalg.norm(df_shared, axis=1)

        df_specific = pd.DataFrame(data=np.hstack(specific_effect), index=gene_names)
        df_specific["total_spec"] = np.linalg.norm(df_specific, axis=1)

        df_effects = pd.concat(
            (df_shared["total_shared"], df_specific["total_spec"]), axis=1
        )
        df_effects["total_effect"] = np.linalg.norm(df_effects, axis=1)
        df_effects_sorted = df_effects.sort_values(by="total_effect", ascending=False)

        # persist across reruns - see explanation above
        st.session_state["df_effects_sorted"] = df_effects_sorted
        st.session_state["history"] = history

st.subheader("Results")

if "df_effects_sorted" in st.session_state:
    df_effects_sorted = st.session_state["df_effects_sorted"]
    top_genes = df_effects_sorted.head(20)

    history = st.session_state["history"]
    st.caption(f"Converged in {history['iters']} iterations")

    fig_conv, ax_conv = plt.subplots()
    ax_conv.plot(history["loglik"], marker="o")
    ax_conv.set_title("Convergence")
    ax_conv.set_xlabel("EM iteration")
    ax_conv.set_ylabel("Log-likelihood")
    fig_conv.tight_layout()
    # st.pyplot(fig_conv)

    fig, ax = plt.subplots()
    top_genes.plot(y=list(top_genes.columns), kind="line", marker="o", ax=ax)
    ax.set_title("Gene effect size (top 20 by total effect)")
    ax.set_xlabel("Genes")
    ax.set_ylabel("Effect size")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)
    st.pyplot(fig_conv)

    st.download_button(
        "Download results (csv)",
        data=df_effects_sorted.to_csv().encode("utf-8"),
        file_name="mbpls_em_effects.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info(
        "Set your parameters in the sidebar and click 'Fit model' to see results here."
    )
