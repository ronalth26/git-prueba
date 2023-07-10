import pandas as pd
import streamlit as st
import altair as alt
import numpy as np


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

st.subheader('Gráfico') 

st.area_chart(data)
st.subheader('E')


from streamlit_image_select import image_select

img = image_select(
    label="Select a cat",
    images=[
        "images/cat1.jpeg",
        "https://bagongkia.github.io/react-image-picker/0759b6e526e3c6d72569894e58329d89.jpg",
        Image.open("images/cat3.jpeg"),
        np.array(Image.open("images/cat4.jpeg")),
    ],
    captions=["A cat", "Another cat", "Oh look, a cat!", "Guess what, a cat..."],
)


#opening the image

W = {'img':[misc.imread('img.jpg')]}
df = pd.DataFrame(W)

# This displays the image
plt.imshow(df.img1[0])
plt.show()






