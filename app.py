import streamlit as st
# from sklearn.ensemble import RandomForestClassifier
from PIL import Image # Required to show images
import pandas as pd
import numpy as np

logo = Image.open("Tesla.png")
st.sidebar.image(logo, width=250)

# Text/Title
st.title("Line Detection")

st.sidebar.header("Team Tesla")
st.sidebar.text("Team members")
st.sidebar.write("""
# Pablo
# Dan Adrian
# Harsha""")

uploaded_file = st.file_uploader("Choose a file")
output = pd.read_csv(uploaded_file)
st.write(output)  