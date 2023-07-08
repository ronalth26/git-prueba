from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.plot(data['POB_URBANA'],data['QRESIDUOS_DOM'],'.', color='green')
plt.title('Generación de residuos domiciliarios para una población urbana')
plt.xlabel('QRESIDUOS_DOM')
plt.ylabel('POB_URBANA')
plt.show()
