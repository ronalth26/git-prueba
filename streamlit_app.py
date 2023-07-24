import streamlit as st
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import re
import warnings
import pickle
import statsmodels.api as sm

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


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

def preparar_datos(df):
    df = df.dropna(subset=['DEPARTAMENTO', 'POB_TOTAL', 'QRESIDUOS_DOM'])  # Eliminar filas con valores faltantes
    df['QRESIDUOS_DOM'] = df['QRESIDUOS_DOM'].str.replace(',', '.').astype(float)  # Reemplazar ',' por '.' y convertir a float
    return df

def train_model(data):
    X = data["POB_TOTAL"].values.reshape(-1, 1)
    y = data["QRESIDUOS_DOM"].values

    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    return model, poly_features

def predict(model, poly_features, num_personas):
    num_personas_poly = poly_features.transform([[num_personas]])
    prediction = model.predict(num_personas_poly)
    return max(prediction[0], 0)  # Asegurar que la predicción no sea menor a cero

def main():
    st.title("Predicción de Residuos Domiciliarios por Departamento")

    # Cargar el dataset y entrenar el modelo
    data = load_data()
    df_residuos_cleaned = preparar_datos(data)
    model, poly_features = train_model(df_residuos_cleaned)

    # Obtener la lista de departamentos únicos en el dataset
    departamentos = df_residuos_cleaned["DEPARTAMENTO"].unique()

    # Widget de selección de departamento
    selected_departamento = st.selectbox("Seleccionar Departamento", departamentos)

    # Filtrar el dataset por el departamento seleccionado
    data_filtered = df_residuos_cleaned[df_residuos_cleaned["DEPARTAMENTO"] == selected_departamento]

    # Interfaz de usuario para ingresar el número de personas
    num_personas = st.number_input("Ingrese el número de personas:", min_value=1, step=1)

    # Realizar la predicción al presionar el botón
    if st.button("Predecir"):
        predicted_residuos = predict(model, poly_features, num_personas)
        st.write(f"El aproximado total de residuos en toneladas al año para el departamento {selected_departamento} es: {predicted_residuos:.2f} toneladas")

    # Gráfico de dispersión interactivo
    st.subheader("Gráfico de Dispersión: Número de Personas vs Cantidad de Residuos")
    scatter_plot = alt.Chart(data_filtered).mark_circle(size=60).encode(
        x=alt.X("POB_TOTAL", title="Número de Personas"),
        y=alt.Y("QRESIDUOS_DOM", title="Cantidad de Residuos (toneladas)"),
        tooltip=["POB_TOTAL", "QRESIDUOS_DOM"]
    ).properties(
        width=600,
        height=400
    ).interactive()

    st.altair_chart(scatter_plot)

    # Gráfico de barras para comparar valor real y valor predicho
    st.subheader("Comparación entre Valor Real y Valor Predicho")
    y_real = data_filtered["QRESIDUOS_DOM"].values
    X_test = np.array([num_personas]).reshape(-1, 1)
    X_test_poly = poly_features.transform(X_test)
    y_pred = model.predict(X_test_poly)
    df_comparacion = pd.DataFrame({"Valor": ["Real", "Predicho"],
                                   "Cantidad de Residuos (toneladas)": [y_real.mean(), max(y_pred[0], 0)]})

    bar_plot = alt.Chart(df_comparacion).mark_bar().encode(
        x="Valor",
        y="Cantidad de Residuos (toneladas)"
    )

    st.altair_chart(bar_plot)

if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------------------------------------------------------------------------




