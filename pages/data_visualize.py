import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

def app():
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        # df_analysis = pd.read_csv('data/2015.csv')
        df_analysis = pd.read_csv('data/main_data.csv')
        df = df_analysis.copy()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write("---")
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)		
		
        