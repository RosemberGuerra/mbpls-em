import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from mbpls_em.estimators import MBPLS_EM
from mbpls_em.preprocessing import center_scale, cleaning_data, input_data_mbpls_em

st.title("📂 Upload & Analyze Multi-Cohort Data")
st.caption(
    "Integrate two independent experimental blocks ($K=2$) to decouple consensus regulatory genes from batch artifacts."
)

def compute_effects(params_fit, gene_names):
    """Decompose each gene's effect into shared vs. block-specific components."""
    W = params_fit["W"]
    P = params_fit["P"]
    beta = params_fit["beta"]
    phi = params_fit["phi"]

    shared_effect = [W @ beta_k.T for beta_k in beta]
    specific_effect = [P_k @ phi_k.T for phi_k, P_k in zip(phi, P)]

    df_shared = pd.DataFrame(data=np.hstack(shared_effect), index=gene_names)
    df_shared["Shared Effect"] = np.linalg.norm(df_shared, axis=1)

    df_specific = pd.DataFrame(data=np.hstack(specific_effect), index=gene_names)
    df_specific["Specific Effect"] = np.linalg.norm(df_specific, axis=1)

    df_effects = pd.concat([df_shared["Shared Effect"], df_specific["Specific Effect"]], axis=1)
    df_effects["Total Effect"] = np.linalg.norm(df_effects, axis=1)
    df_sorted = df_effects.sort_values(by="Total Effect", ascending=False).reset_index()
    df_sorted.rename(columns={"index": "Gene"}, inplace=True)
    return df_sorted

# --- File Ingestion ---
with st.container(border=True):
    st.markdown("### 1. Ingest Data Cohorts")
    st.caption("Counts files: CSV with samples as rows and genes as columns. Responses: single continuous phenotype.")
    
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("#### Cohort Block 1")
        b1_x_file = st.file_uploader("Counts Matrix ($X_1$)", type="csv", key="b1_x")
        b1_y_file = st.file_uploader("Response Vector ($Y_1$)", type="csv", key="b1_y")
        b1_col_t, b1_col_s = st.columns(2)
        with b1_col_t:
            b1_transpose = st.checkbox("Transpose (Genes × Samples)", key="b1_t")
        with b1_col_s:
            b1_symbol = st.checkbox("Query Gene Symbols", key="b1_s")

    with col2:
        st.markdown("#### Cohort Block 2")
        b2_x_file = st.file_uploader("Counts Matrix ($X_2$)", type="csv", key="b2_x")
        b2_y_file = st.file_uploader("Response Vector ($Y_2$)", type="csv", key="b2_y")
        b2_col_t, b2_col_s = st.columns(2)
        with b2_col_t:
            b2_transpose = st.checkbox("Transpose (Genes × Samples)", key="b2_t")
        with b2_col_s:
            b2_symbol = st.checkbox("Query Gene Symbols", key="b2_s")

# --- Model Hyperparameters ---
with st.container(border=True):
    st.markdown("### 2. Latent Subspace Dimensionality")
    col_r, col_q1, col_q2 = st.columns(3)
    with col_r:
        r_up = st.slider("Shared Rank ($r$)", min_value=1, max_value=10, value=2, key="r_up")
    with col_q1:
        q1_up = st.slider("Cohort 1 Specific Rank ($q_1$)", min_value=1, max_value=10, value=1, key="q1_up")
    with col_q2:
        q2_up = st.slider("Cohort 2 Specific Rank ($q_2$)", min_value=1, max_value=10, value=1, key="q2_up")

    fit_upload_clicked = st.button("🚀 Run MBPLS-EM Prioritization", type="primary", use_container_width=True)

if fit_upload_clicked:
    if not (b1_x_file and b1_y_file and b2_x_file and b2_y_file):
        st.error("Please provide both feature ($X$) and phenotype ($Y$) files for both cohorts.")
    else:
        with st.spinner("Aligning cohorts, standardizing matrices, and running EM algorithm..."):
            try:
                df_x1 = pd.read_csv(b1_x_file, index_col=0)
                df_y1 = pd.read_csv(b1_y_file, index_col=0)
                df_x2 = pd.read_csv(b2_x_file, index_col=0)
                df_y2 = pd.read_csv(b2_y_file, index_col=0)

                if b1_transpose:
                    df_x1 = df_x1.T
                if b2_transpose:
                    df_x2 = df_x2.T

                if df_y1.shape[1] != 1 or df_y2.shape[1] != 1:
                    st.error("Response files must contain exactly one target feature column.")
                    st.stop()

                data_b1 = cleaning_data(X=df_x1, Y=df_y1, dropna=True, get_symbol=b1_symbol)
                data_b2 = cleaning_data(X=df_x2, Y=df_y2, dropna=True, get_symbol=b2_symbol)

                input_data, gene_names = input_data_mbpls_em(data_b1, data_b2)
                input_data_scaled = center_scale(input_data)

                params_fit, history = MBPLS_EM(
                    data=input_data_scaled, r=r_up, q_list=[q1_up, q2_up]
                )

                st.session_state["upload_effects"] = compute_effects(params_fit, gene_names)
                st.session_state["upload_history"] = history
                st.session_state["upload_dims"] = {
                    "genes": len(gene_names),
                    "n1": df_x1.shape[0],
                    "n2": df_x2.shape[0],
                }
            except Exception as e:
                st.error(f"Execution Error: {e}")

# --- Render Upload Results ---
if "upload_effects" in st.session_state:
    st.divider()
    upload_effect = st.session_state["upload_effects"]
    history = st.session_state["upload_history"]
    dims = st.session_state["upload_dims"]

    st.subheader("🎯 Prioritization Results")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Genes Retained", f"{dims['genes']}")
    m2.metric("Cohort 1 Samples", f"{dims['n1']}")
    m3.metric("Cohort 2 Samples", f"{dims['n2']}")
    m4.metric("Iterations to Converge", f"{history['iters']}")

    tab_plot, tab_table, tab_diag = st.tabs(["📊 Component Breakdown", "🗃 Gene Rankings Table", "📉 EM Diagnostics"])

    with tab_plot:
        top_k_slider = st.slider("Display Top Genes", min_value=10, max_value=min(100, len(upload_effect)), value=25)
        top_subset = upload_effect.head(top_k_slider)

        df_long = top_subset.melt(
            id_vars=["Gene"], 
            value_vars=["Shared Effect", "Specific Effect"], 
            var_name="Component", 
            value_name="Value"
        )

        chart = (
            alt.Chart(df_long)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X("Gene:N", sort=list(top_subset["Gene"]), title="Ranked Gene"),
                y=alt.Y("Value:Q", stack=None, title="Component Norm"),
                color=alt.Color("Component:N", scale=alt.Scale(range=["#1E88E5", "#FB8C00"])),
                tooltip=["Gene", "Component", alt.Tooltip("Value:Q", format=".3f")],
            )
            .properties(height=360)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    with tab_table:
        st.dataframe(
            upload_effect.style.background_gradient(subset=["Total Effect", "Shared Effect"], cmap="Greens"),
            use_container_width=True,
            height=400,
        )
        st.download_button(
            "📥 Download Target Prioritization (CSV)",
            data=upload_effect.to_csv(index=False).encode("utf-8"),
            file_name="mbpls_em_cohort_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab_diag:
        loglik_df = pd.DataFrame({"Iteration": range(1, len(history["loglik"]) + 1), "Log-Likelihood": history["loglik"]})
        line_chart = (
            alt.Chart(loglik_df)
            .mark_line(point=True, color="#00897B")
            .encode(
                x=alt.X("Iteration:Q", title="Iteration"),
                y=alt.Y("Log-Likelihood:Q", scale=alt.Scale(zero=False), title="Observed Log-Likelihood"),
            )
            .properties(height=300)
        )
        st.altair_chart(line_chart, use_container_width=True)