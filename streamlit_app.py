import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Datos Abiertos") # Nombre para configurar la pagina web
st.header('Residuos municipalidades generados anualmente') #Va a ser el titulo de la pagina
st.subheader('La GPC de residuos domiciliarios es un dato obtenido de los estudios de caracterización elaborados por las municipalidades provinciales y distritales y se refiere a la generación de residuos sólidos por persona-día.') #Subtitulo

nombre_variables = ["POB_URBANA", "QRESIDUOS_DOM"]

df = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/master/lista-residuos.csv',sep=';',encoding='iso-8859-1'
                 )
df 


st.subheader('El objetivo sería determinar si existe una relación lineal entre dos variables y predecir la generación de residuos domiciliarios para una población determinada.') #Subtitulo

data  = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/master/lista-residuos.csv',sep=';',encoding='iso-8859-1',  usecols=["POB_URBANA", "QRESIDUOS_DOM"]                       
                 )
data

st.subheader('Generación de residuos domiciliarios para una población urbana') 

st.image("https://raw.githubusercontent.com/ronalth26/git-prueba/master/img.JPG", caption="Descripción de la imagen")


# Crear el gráfico de dispersión
fig, ax = plt.subplots()
ax.scatter(data['POB_URBANA'], data['QRESIDUOS_DOM'])

# Agregar una línea al gráfico de dispersión
line_x = [min(data['POB_URBANA']), max(data['QRESIDUOS_DOM'])]  # Valores para el eje X de la línea
line_y = [3, 3]  # Valores para el eje Y de la línea
ax.plot(line_x, line_y, color='red', linestyle='--', label='Línea')  # Personalizar la línea según tus necesidades


# Establecer etiquetas y título
ax.set_xlabel('QRESIDUOS_DOM')
ax.set_ylabel('POB_URBANA')
ax.set_title('Gráfico de Dispersión')

# Mostrar el gráfico en Streamlit
st.pyplot(fig)





