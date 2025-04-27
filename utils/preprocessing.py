import re
import string
import streamlit as st


@st.cache_data
def preprocess_text(text):
    """
    Preprocess text by removing numbers, punctuation, and converting to lowercase
    """
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text.lower()
