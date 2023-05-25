from operator import index 
import streamlit as st 
import plotly.express as px  
import pycaret.regression as preg
import pycaret.classification as pclas
import pandas_profiling
import pandas as pd   
from streamlit_pandas_profiling import st_profile_report
import os
import requests
import json  
from streamlit_lottie import st_lottie

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None 
    return r.json()

data_name = 'dataset.csv'    # insert your datapath here 

animation = load_lottieurl('https://assets8.lottiefiles.com/packages/lf20_uPKZ69ixvZ.json')
if os.path.exists(data_name):
    df = pd.read_csv(data_name, index_col=None)


with st.sidebar:
    st_lottie(animation) 
    st.title('AutoML')
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    type_model = st.selectbox("Type Model", ["Classification", "Regression"])
    

if choice == "Upload":
    st.title("Upload You Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
    

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling":
    if type_model == "Classification":
        chosen_target = st.selectbox("Choose the Target Column", df.columns)
        if st.button('Run Modelling'):
            pclas.setup(df, target=chosen_target)
            setup_df = pclas.pull()
            st.dataframe(setup_df)
            best_model = pclas.compare_models()
            compare_df = pclas.pull()
            st.dataframe(compare_df)
            pclas.save_model(best_model, 'best_model_classification')
    
    elif type_model == "Regression":
        chosen_target = st.selectbox("Chosen the Target Column", df.columns)
        if st.button("Run Modelling"):
            preg.setup(df, target=chosen_target)
            setup_df = preg.pull()
            st.dataframe(setup_df)
            best_model = preg.compare_models()
            compare_df = preg.pull()
            st.dataframe(compare_df)
            preg.save_model(best_model, 'best_model_regression')

if choice == "Download":
    if type_model == "Classification":
        with open('best_model_classification.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name='best_model_classification.pkl')
    elif type_model == "Regression":
        with open('best_model_regression.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name='best_model_regression.pkl')