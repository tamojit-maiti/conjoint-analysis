import pandas as pd
import streamlit as st
import time

st.set_page_config(
    page_title="Upload Data",
    page_icon="⬆️",
)

st.write('# ⬆️ Upload Data')
st.write('### Upload conjoint survey data')
conjoint_file = st.file_uploader("The file must be in CSV format", key = '1')

if conjoint_file is not None:
    if 'conjoint_df' not in st.session_state:
        # sales table
        t0 = time.time()
        with st.spinner(text='Reading  table ...'):
            conjoint_df = pd.read_csv(conjoint_file)
            st.session_state['conjoint_df'] = conjoint_df
        st.success(f'Successfully read the sales table in {"{:.2f}".format(time.time() - t0)}s', icon = '✅')
    else:
        st.success('Successfully read the conjoint table from cache', icon = '✅')
        st.write('## Consolidated table view')
        conjoint_df = st.session_state['conjoint_df']
    st.dataframe(conjoint_df.head())
    