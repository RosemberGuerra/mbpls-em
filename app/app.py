import streamlit as st

st.set_page_config(
    page_title="Gene Prioritization explorer",
    page_icon=":material/genetics:",
    layout="centered",
)

pg = st.navigation(
    [
        st.Page("pages/homepage.py", title="Home page", icon=":material/home:"),
        st.Page(
            "pages/showcase.py",
            title="Showcase",
            icon=":material/dashboard:",
            default=True,
        ),
        st.Page("pages/uploaddata.py", title="Upload data", icon=":material/upload:"),
        # st.Page("pages/examples.py", title="Examples", icon=":material/code:"),
    ]
)
pg.run()

with st.sidebar:
    st.markdown(
        ":material/code: [mbpls-em](https://github.com/RosemberGuerra/mbpls-em)"
    )
    # st.caption("Made in :streamlit: by [@andfanilo](https://andfanilo.com)")
  