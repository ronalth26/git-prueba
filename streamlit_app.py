from collections import namedtuple
import altair as alt
import math
import numpy as np
import pandas as pd
import streamlit as st

@st.cache
df = pd.read_csv('lista-residuos.csv')
  return df

df=load_data()
"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

df = pd.DataFrame(np.random.randn(100, 4),  columns=['A','B','C','D'])
