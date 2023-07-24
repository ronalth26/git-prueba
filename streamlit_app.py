import streamlit as st
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm


import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


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
    df_residuos = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/master/lista-residuos.csv', sep=';', encoding='iso-8859-1')
    return df_residuos

# Entrenar el modelo de regresión polinómica
def train_model(data):
    X = data["POB_TOTAL"].values.reshape(-1, 1)
    y = data["QRESIDUOS_DOM"].values

    # Filtrar filas con valores iguales a 0 en la columna QRESIDUOS_DOM
    mask_nonzero = y != 0
    X = X[mask_nonzero]
    y = y[mask_nonzero]

    # Aplicar log shift (sumar epsilon) para evitar logaritmo de valores iguales a 0
    epsilon = 1e-10
    y_log = np.log(y + epsilon)

    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y_log)

    return model, poly_features

# Realizar la predicción
def predict(model, poly_features, num_personas):
    num_personas_poly = poly_features.transform([[num_personas]])
    prediction_log = model.predict(num_personas_poly)
    prediction = np.exp(prediction_log)
    return prediction[0]

def main():
    st.title("Predicción de Residuos Domiciliarios por Departamento")

    # Cargar el dataset y entrenar el modelo
    data = load_data()
    model, poly_features = train_model(data)

    # Obtener la lista de departamentos únicos en el dataset
    departamentos = data["DEPARTAMENTO"].unique()

    # Widget de selección de departamento
    selected_departamento = st.selectbox("Seleccionar Departamento", departamentos)

    # Filtrar el dataset por el departamento seleccionado
    data_filtered = data[data["DEPARTAMENTO"] == selected_departamento]

    # Interfaz de usuario para ingresar el número de personas
    num_personas = st.number_input("Ingrese el número de personas:", min_value=1, step=1)

    # Realizar la predicción al presionar el botón
    if st.button("Predecir"):
        predicted_residuos = predict(model, poly_features, num_personas)
        st.write(f"El aproximado total de residuos en toneladas al año para el departamento {selected_departamento} es: {predicted_residuos:.2f}")

if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------------------------------------------------------------------------




