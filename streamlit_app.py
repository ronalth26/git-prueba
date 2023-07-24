
import streamlit as st
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm


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

# Entrenar el modelo de regresión polinómica
def train_model(data):
    X = data["POB_TOTAL"].values.reshape(-1, 1)
    y = data["QRESIDUOS_DOM"].values

    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    return model, poly_features

# Realizar la predicción
def predict(model, poly_features, num_personas):
    num_personas_poly = poly_features.transform([[num_personas]])
    prediction = model.predict(num_personas_poly)
    return prediction[0]

def main():
    st.title("Predicción de Residuos Domiciliarios por Región")

    # Cargar el dataset y entrenar el modelo
    data = load_data()
    model, poly_features = train_model(data)

    # Obtener la lista de regiones únicas en el dataset
    regiones = data["REG_NAT"].unique()

    # Widget de selección de región
    selected_region = st.selectbox("Seleccionar Región", regiones)

    # Filtrar el dataset por la región seleccionada
    data_filtered = data[data["REG_NAT"] == selected_region]

    # Interfaz de usuario para ingresar el número de personas
    num_personas = st.number_input("Ingrese el número de personas:", min_value=1, step=1)

    # Realizar la predicción al presionar el botón
    if st.button("Predecir"):
        predicted_residuos = predict(model, poly_features, num_personas)
        st.write(f"El aproximado total de residuos en toneladas al año para la región {selected_region} es: {predicted_residuos:.2f}")

if __name__ == "__main__":
    main()





#----------------------------------------------------------------------------------------------------------------------------------------------




