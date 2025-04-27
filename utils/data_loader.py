import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    """
    Load and cache the dataset from file
    """
    df = pd.read_csv('D:/Kuliah/SEM 6/BahasaAlami/UTS/train_preprocess_stemmed.csv')
    return df
