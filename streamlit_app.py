
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
    df_residuos = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/master/lista-residuos.csv',sep=';',encoding='iso-8859-1'
                 )  # Reemplaza "dataset_residuos.csv" por la ruta a tu archivo CSV
    return data

def preparar_datos(df):
    # Implementa aquí la función para el preprocesamiento y limpieza de datos si es necesario.
    # Asegúrate de convertir las columnas relevantes a valores numéricos y eliminar filas con valores faltantes.

    return df

# Preprocesamiento y limpieza de datos
df_residuos_cleaned = preparar_datos(df_residuos)

# Separar las características (X) y las etiquetas (y)
X = df_residuos_cleaned.drop(['QRESIDUOS_DOM'], axis=1).values
y = df_residuos_cleaned['QRESIDUOS_DOM']

# Escalar las características en un rango de 0 a 1
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# Entrenar el modelo de Regresión de Bosques Aleatorios
model = RandomForestRegressor(max_depth=9, random_state=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluación del modelo
mse = round(metrics.mean_squared_error(y_test, y_pred), 5)
rmse = round(np.sqrt(mse), 3)
r2_value = round(metrics.r2_score(y_test, y_pred), 5)

# Guardar el modelo entrenado en un archivo .pkl
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)



#----------------------------------------------------------------------------------------------------------------------------------------------




