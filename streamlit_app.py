import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#from PIL import Image

st.set_page_config(page_title="Datos Abiertos") # Nombre para configurar la pagina web
st.header('Residuos municipalidades generados anualmente') #Va a ser el titulo de la pagina
st.subheader('La GPC de residuos domiciliarios es un dato obtenido de los estudios de caracterización elaborados por las municipalidades provinciales y distritales y se refiere a la generación de residuos sólidos por persona-día.') #Subtitulo

nombre_variables = ["POB_URBANA", "QRESIDUOS_DOM"]

df  = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/master/lista-residuos.csv',sep=';',encoding='iso-8859-1'
                 )
df 


st.subheader('El objetivo sería determinar si existe una relación lineal entre dos variables y predecir la generación de residuos domiciliarios para una población determinada.') #Subtitulo

data  = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/master/lista-residuos.csv',sep=';',encoding='iso-8859-1',  usecols=["POB_URBANA", "QRESIDUOS_DOM"]                       
                 )
data


st.sidebar.subheader('Input features')
sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.8)
sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.1)
petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 3.8)
petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 1.2)

# Separate to X and y
X = df.drop('Species', axis=1)
y = df.Species

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building
rf = RandomForestClassifier(max_depth=2, max_features=4, n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Apply model to make predictions
y_pred = rf.predict([[sepal_length, sepal_width, petal_length, petal_width]])

# Print EDA
st.subheader('Brief EDA')
st.write('The data is grouped by the class and the variable mean is computed for each class.')
groupby_species_mean = df.groupby('Species').mean()
st.write(groupby_species_mean)
st.line_chart(groupby_species_mean.T)

# Print input features
st.subheader('Input features')
input_feature = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
st.write(input_feature)

# Print prediction output
st.subheader('Output')

    st.caption(QUESTION_OR_FEEDBACK)


