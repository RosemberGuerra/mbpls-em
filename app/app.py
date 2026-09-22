import streamlit as st

st.set_page_config(
    page_title="Gene Prioritization explorer | mbpls-em",
    page_icon=":material/genetics:",
    layout="wide",
    initial_sidebar_state= "expanded"
)

pg = st.navigation(
    [
        st.Page(
            "pages/homepage.py",
            title="Home",
            icon=":material/home:",
            default=True, # Set Home as entry point
            ),
        st.Page(
            "pages/showcase.py",
            title="Showcase",
            icon=":material/dashboard:",
            ),
        st.Page("pages/uploaddata.py", 
                title="Upload Data", 
                icon=":material/upload:",
                ),      
    ]
)
pg.run()

with st.sidebar:
    st.divider()
    st.markdown("### About")
    st.caption(
        "Probabilistic multi-block PLS with EM estimation for multi-cohort gene prioritization."
    )
    st.markdown(
        ":material/code: [GitHub Repository](https://github.com/RosemberGuerra/mbpls-em)"
    )
    st.caption("Alpha-Synuclein OMICS (SynOD) Project")
    # st.caption("Made in :streamlit: by [@andfanilo](https://andfanilo.com)")
  