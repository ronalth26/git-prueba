import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Encuesta Oficial EPS") # Nombre para configurar la pagina web
st.header('Resultados Encuestas Nacionales EPS Colombia 2022') #Va a ser el titulo de la pagina
st.subheader('Cómo perciben los ciudadanos el servicio de las EPS en Colombia?') #Subtitulo

nombre_variables = ['POB_URBANA', 'QRESIDUOS_DOM']

excel_file  = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/master/lista-residuos.csv',sep=';',encoding='iso-8859-1'
                 )
excel_file 

