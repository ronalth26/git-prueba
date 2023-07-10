import pandas as pd
import streamlit as st
import plotly.express as px
#from PIL import Image

st.set_page_config(page_title="Encuesta Oficial EPS") # Nombre para configurar la pagina web
st.header('Resultados Encuestas Nacionales EPS Colombia 2022') #Va a ser el titulo de la pagina
st.subheader('Cómo perciben los ciudadanos el servicio de las EPS en Colombia?') #Subtitulo



nombre_variables = ['POB_URBANA', 'QRESIDUOS_DOM']
# Carga de datos desde la web: descarga el CSV y lo carga como DataFrame

data = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/master/lista-residuos.csv',sep=';',encoding='iso-8859-1'
                 )
data

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.plot(data['POB_URBANA'],data['QRESIDUOS_DOM'],'.', color='green')
plt.title('Generación de residuos domiciliarios para una población urbana')
plt.xlabel('QRESIDUOS_DOM')
plt.ylabel('POB_URBANA')
plt.show()
