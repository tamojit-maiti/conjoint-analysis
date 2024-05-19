import pandas as pd
import streamlit as st
from utils.utils import plot_feature_importance, plot_part_worth
import time

st.set_page_config(
    page_title="Analyse Results",
    page_icon="🧠",
)

st.write("# Survey Statistics")
if 'conjoint_df' not in st.session_state:
    st.warning(body = 'Please upload relevant data and try again', icon = "🔥")
else:
    conjoint_df = st.session_state['conjoint_df']
    # Reporting Metrics 1
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(label = 'Number of respondents', value = conjoint_df.shape[0])
    with c2:
        st.metric(label = 'Number of features', value = len(set(conjoint_df.columns.map(lambda x: x.split('_')[0]))))
    with c3:
        st.metric(label = 'Number of options', value = conjoint_df.shape[1] - 1)

    # Workflow
    fig1, res = plot_part_worth(conjoint_df)
    fig2 = plot_feature_importance(res)

    st.write("# Relative Feature Importance")
    st.pyplot(fig2)

    st.write("# Parts Worth")
    st.pyplot(fig1)