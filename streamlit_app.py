
import streamlit as st
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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

# Establecer etiquetas y título
ax.set_xlabel('QRESIDUOS_DOM')
ax.set_ylabel('POB_URBANA')
ax.set_title('Gráfico de Dispersión')

# Mostrar el gráfico en Streamlit

st.pyplot(fig)
st.subheader('El objetivo sería determinar si existe una relación lineal entre dos variables y predecir la generación de residuos domiciliarios para una población determinada.') 

#prueba#
#----------------------------------------------------------------------------------------------------------------------------------------------

def load_data():
    data = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/master/lista-residuos.csv',sep=';',encoding='iso-8859-1'
                 )  # Reemplaza "dataset.csv" por la ruta a tu archivo CSV
    return data

def train_model(data):
    X = data.drop("DEPARTAMENTO", axis=1)  # Reemplaza "target_column" por el nombre de la columna objetivo
    y = data["target_column"]

    # Entrena tu modelo aquí
    model = RandomForestClassifier()  # Cambia por el modelo que desees usar
    model.fit(X, y)

    return model

def predict(model, input_data):
    # Realiza la predicción usando el modelo
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title("Aplicación de Predicción")

    # Carga el dataset
    data = load_data()

    # Entrena el modelo
    model = train_model(data)

    # Widget de selección con opciones de una columna del dataset
    selected_column = st.selectbox("Seleccionar columna", data.columns)

    # Interfaz de usuario para ingresar las características (features) y obtener la predicción
    st.header("Ingresar características para la predicción")
    input_data = {}
    for column in data.columns:
        if column != "target_column" and column != selected_column:  # Excluye la columna objetivo y la columna seleccionada
            input_data[column] = st.number_input(f"Ingresar {column}", value=0)

    # Convierte las características ingresadas en un DataFrame
    input_df = pd.DataFrame([input_data])

    # Realiza la predicción
    if st.button("Predecir"):
        prediction = predict(model, input_df)
        st.write(f"La predicción es: {prediction[0]}")

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------------------------------------------------------------------------




