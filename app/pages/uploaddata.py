import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

from mbpls_em.simulate import generate_multiblock_mbpls
from mbpls_em.estimators import MBPLS_EM
from mbpls_em.preprocessing import cleaning_data, input_data_mbpls_em, center_scale

st.set_page_config(
    page_title="Gene Prioritization explorer", page_icon=":material/genetics:", layout="centered"
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


def compute_effects(params_fit, gene_names):
    """Decompose each gene's effect into shared vs. block-specific components."""
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
    return df_effects.sort_values(by="total_effect", ascending=False)



col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown(LOGO_SVG, unsafe_allow_html=True)
with col_title:
    st.title("Gene Prioritization Explorer")
    st.caption("Probabilistic multi-block PLS with EM estimation")

st.caption(
        "Two blocks (K=2). Counts files: CSV, samples as rows, genes as columns, "
        "first column is a sample ID. Response files: CSV with one data column "
        "(plus the same leading ID column), same row order as its counts file."
    )

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Block 1**")
    b1_x_file = st.file_uploader("Omics data ($X_1$)", type="csv", key="b1_x")
    b1_y_file = st.file_uploader("Response $Y_1$", type="csv", key="b1_y")
    b1_transpose = st.checkbox("File is genes × samples (transpose)", key="b1_t")
    b1_symbol = st.checkbox(
        "Convert gene IDs to symbols", key="b1_s",
        help="Calls an external gene-lookup service - slower, needs internet.",
    )
with col2:
    st.markdown("**Block 2**")
    b2_x_file = st.file_uploader("Omics data ($X_2$)", type="csv", key="b2_x")
    b2_y_file = st.file_uploader("Response $Y_2$", type="csv", key="b2_y")
    b2_transpose = st.checkbox("File is genes × samples (transpose)", key="b2_t")
    b2_symbol = st.checkbox(
        "Convert gene IDs to symbols", key="b2_s",
        help="Calls an external gene-lookup service - slower, needs internet.",
    )


st.markdown("**Model settings**")
col_r, col_q1, col_q2 = st.columns(3)
with col_r:
    r_up = st.slider("Shared rank (r)", min_value=1, max_value=10, value=2, key="r_up")
with col_q1:
    q1_up = st.slider("Block 1 rank (q)", min_value=1, max_value=10, value=1, key="q1_up")
with col_q2:
    q2_up = st.slider("Block 2 rank (q)", min_value=1, max_value=10, value=1, key="q2_up")

fit_upload_clicked = st.button("Fit model", use_container_width=True, key="fit_upload")

if fit_upload_clicked:
    if not (b1_x_file and b1_y_file and b2_x_file and b2_y_file):
        st.error("Please upload counts and response files for both blocks.")
    else:
        with st.spinner("Fitting model..."):
            df_x1 = pd.read_csv(b1_x_file, index_col=0)
            df_y1 = pd.read_csv(b1_y_file, index_col=0)
            df_x2 = pd.read_csv(b2_x_file, index_col=0)
            df_y2 = pd.read_csv(b2_y_file, index_col=0)

            if b1_transpose:
                df_x1 = df_x1.T
            if b2_transpose:
                df_x2 = df_x2.T

            if df_y1.shape[1] != 1 or df_y2.shape[1] != 1:
                st.error(
                    "Response files must have exactly one data column "
                    "(besides the leading ID column)."
                )
            else:
                data_b1 = cleaning_data(X=df_x1, Y=df_y1, dropna=True, get_symbol=b1_symbol)
                data_b2 = cleaning_data(X=df_x2, Y=df_y2, dropna=True, get_symbol=b2_symbol)

                input_data, gene_names = input_data_mbpls_em(data_b1, data_b2)
                input_data_scaled = center_scale(input_data)

                params_fit, history = MBPLS_EM(
                    data=input_data_scaled, r=r_up, q_list=[q1_up, q2_up]
                )

                # st.session_state["upload_effects"] = compute_effects(params_fit, gene_names)
                # st.session_state["upload_history"] = history
                st.subheader("Results")
                upload_effect = compute_effects(params_fit, gene_names)
                upload_effect.reset_index(level=0, inplace=True)

                #  Melt to long format for Altair
                df_long = upload_effect.iloc[:200].melt(id_vars="index", var_name="Variable", value_name="Value")
                 # Create chart with preserved order
                chart = alt.Chart(df_long).mark_line(point=True).encode(
                    x=alt.X("index", sort=list(upload_effect["index"])),  # Explicit order
                    y="Value",
                    color="Variable"
                    )

                st.altair_chart(chart, use_container_width=True)
                st.line_chart(history["loglik"], 
                    x_label ="Algorithm Iterations",
                    y_label= "Log-Likelihood")


# render_results("upload_effects", "upload_history")
