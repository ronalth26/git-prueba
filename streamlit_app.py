from collections import namedtuple
import altair as alt
import math
import numpy as np
import pandas as pd
import streamlit as st

@st.cache
def load_data():
  df=pd.read_csv('https://github.com/ronalth26/git-prueba/blob/ccfc74218dfbe9876af6da2e4fc4f4104b37c66d/lista-residuos.csv',sep=';',encoding='iso-8859-1')
  return df

df=load_data()


"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


