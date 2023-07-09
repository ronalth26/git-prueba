from collections import namedtuple
import altair as alt
import math
import numpy as np
import pandas as pd
import streamlit as st



# Read the CSV file
airbnb_data = pd.read_csv("lista-residuos.csv")

# View the first 5 rows
airbnb_data.head()


"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


