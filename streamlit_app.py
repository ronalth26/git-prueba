import pandas as pd
import streamlit as st

#from PIL import Image

st.set_page_config(page_title="Datos Abiertos") # Nombre para configurar la pagina web
st.header('Residuos municipalidades generados anualmente') #Va a ser el titulo de la pagina
st.subheader('La GPC de residuos domiciliarios es un dato obtenido de los estudios de caracterización elaborados por las municipalidades provinciales y distritales y se refiere a la generación de residuos sólidos por persona-día.') #Subtitulo

nombre_variables = ['POB_URBANA', 'QRESIDUOS_DOM']

excel_file  = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/master/lista-residuos.csv',sep=';',encoding='iso-8859-1'
                 )
excel_file 

st.subheader('El objetivo sería determinar si existe una relación lineal entre dos variables y predecir la generación de residuos domiciliarios para una población determinada.') #Subtitulo

tabla  = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/master/lista-residuos.csv',sep=';',encoding='iso-8859-1', names=nombre_variables
                 )
tabla 



tabla.head() 
