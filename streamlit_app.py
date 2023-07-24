
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
Con estos cambios, estamos utilizando los datos originales en la columna "POB_TOTAL" para entrenar el modelo y realizar la predicción sin aplicar una transformación logarítmica. El modelo de regresión polinómica de grado 2 ahora debería ajustarse mejor a los datos proporcionados y generar predicciones más precisas.

Espero que con estos ajustes puedas obtener predicciones más precisas y evitar valores negativos en la estimación. Si tienes más ejemplos de datos disponibles y deseas mejorar aún más la precisión del modelo, considera agregar más datos al dataset para que el modelo tenga más información para aprender.





User
al ingresar 1 me sale El aproximado total de residuos en toneladas al año para la región SIERRA es: -842.69
ChatGPT
Lamento la confusión. Parece que todavía estamos enfrentando problemas con las predicciones negativas, incluso después de los cambios realizados en el código. El problema podría deberse a las características del dataset o la relación entre las variables.

Una posible solución para evitar predicciones negativas es utilizar un modelo más sofisticado que tenga en cuenta los límites y restricciones de los datos. En lugar de utilizar una regresión lineal o polinómica, podríamos utilizar un modelo de regresión que garantice predicciones no negativas, como el modelo Gamma Generalizado.

Para implementar el modelo Gamma Generalizado, podemos utilizar la biblioteca statsmodels, que proporciona un modelo de regresión con la distribución gamma. Aquí está el código modificado:

python
Copy code
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

# Cargar el dataset
def load_data():
    data = pd.read_csv("dataset_residuos.csv")  # Reemplaza "dataset_residuos.csv" por la ruta a tu archivo CSV
    return data

# Entrenar el modelo de regresión gamma generalizado
def train_model(data):
    X = data["POB_TOTAL"].values
    y = data["QRESIDUOS_DOM"].values

    X_poly = PolynomialFeatures(degree=2).fit_transform(X)

    model = sm.GLM(y, X_poly, family=sm.families.Gamma())
    result = model.fit()

    return result

# Realizar la predicción
def predict(result, num_personas):
    num_personas_poly = PolynomialFeatures(degree=2).fit_transform([[num_personas]])
    prediction = result.predict(num_personas_poly)
    return prediction[0]

def main():
    st.title("Predicción de Residuos Domiciliarios por Región")

    # Cargar el dataset y entrenar el modelo
    data = load_data()
    result = train_model(data)

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
        predicted_residuos = predict(result, num_personas)
        st.write(f"El aproximado total de residuos en toneladas al año para la región {selected_region} es: {predicted_residuos:.2f}")

if __name__ == "__main__":
    main()
Con este cambio, hemos utilizado el modelo de regresión gamma generalizado a través de la biblioteca statsmodels. Este modelo garantiza que las predicciones sean no negativas y puede ser más adecuado para la naturaleza de los datos de residuos. Sin embargo, ten en cuenta que este es solo un enfoque posible y que la elección del modelo siempre debe basarse en una comprensión adecuada de los datos y el problema que estás abordando.

Espero que con esta implementación puedas obtener predicciones más precisas y evites resultados negativos en la estimación. Si sigues enfrentando problemas o tienes más datos disponibles para mejorar el modelo, considera realizar más ajustes o agregar más datos al dataset.





User
me vota el siguiente error "ModuleNotFoundError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/home/appuser/venv/lib/python3.9/site-packages/streamlit/scriptrunner/script_runner.py", line 554, in _run_script
    exec(code, module.__dict__)
File "/app/git-prueba/streamlit_app.py", line 10, in <module>
    import statsmodels.api as sm"
ChatGPT
Lamento que sigas enfrentando problemas. El error "ModuleNotFoundError" indica que la biblioteca statsmodels no está instalada en tu entorno. La biblioteca statsmodels es necesaria para utilizar el modelo de regresión gamma generalizado que mencioné en la respuesta anterior.

Para solucionar este problema, asegúrate de instalar la biblioteca statsmodels en tu entorno antes de ejecutar la aplicación de Streamlit. Puedes instalarla utilizando pip de la siguiente manera:

Copy code
pip install statsmodels
Después de instalar la biblioteca, asegúrate de que tu entorno de desarrollo reconozca la instalación. Si estás utilizando un entorno virtual (como venv o conda), asegúrate de que esté activado y que la biblioteca esté instalada en ese entorno específico.

Una vez que hayas realizado la instalación y configurado correctamente el entorno, vuelve a ejecutar la aplicación de Streamlit para ver si el problema se ha solucionado y si ahora puedes utilizar el modelo de regresión gamma generalizado.

Si sigues enfrentando problemas después de instalar statsmodels, verifica que no haya otros errores o problemas en el código y que los datos sean válidos para el entrenamiento del modelo. Si es posible, proporciona más información sobre el entorno de desarrollo que estás utilizando y cualquier otra información relevante para que pueda ayudarte de manera más específica.





User
estoy en github, no puedo instalar pero tengo mi archivo requirements.txt
ChatGPT
Entiendo que estés utilizando GitHub y que no puedas instalar bibliotecas directamente en ese entorno. Sin embargo, puedes incluir la biblioteca statsmodels en tu archivo requirements.txt para que se instale automáticamente cuando alguien clone o ejecute tu proyecto en su entorno local.

Agrega la siguiente línea al archivo requirements.txt:

Copy code
statsmodels
Esto indicará a las herramientas de gestión de paquetes (por ejemplo, pip) que deben instalar la biblioteca statsmodels junto con otras dependencias especificadas en el archivo requirements.txt.

Luego, asegúrate de que el archivo requirements.txt se encuentre en la raíz de tu proyecto en GitHub. Cuando alguien quiera clonar o ejecutar tu proyecto, puede utilizar el siguiente comando para instalar todas las bibliotecas requeridas:

Copy code
pip install -r requirements.txt
Esto instalará statsmodels y otras dependencias necesarias para tu proyecto. Una vez que las bibliotecas estén instaladas, podrán ejecutar la aplicación de Streamlit sin problemas.

Recuerda que si estás utilizando un entorno de desarrollo local, también debes asegurarte de que tengas Python instalado y que estés utilizando el mismo entorno virtual (si lo tienes) para instalar las bibliotecas requeridas.

Espero que esta solución te ayude a resolver el problema y que puedas seguir adelante con tu proyecto en GitHub sin errores. Si necesitas más ayuda o tienes alguna otra pregunta, no dudes en preguntar. ¡Buena suerte con tu proyecto!





User
me sigue votando el mismo error "ModuleNotFoundError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/home/appuser/venv/lib/python3.9/site-packages/streamlit/scriptrunner/script_runner.py", line 554, in _run_script
    exec(code, module.__dict__)
File "/app/git-prueba/streamlit_app.py", line 10, in <module>
    import statsmodels.api as sm"
ChatGPT
Lamento que sigas enfrentando problemas. Parece que la biblioteca statsmodels aún no está instalada correctamente en tu entorno, a pesar de que la has agregado al archivo requirements.txt.

Es posible que el archivo requirements.txt no se esté utilizando automáticamente para instalar las dependencias cuando se ejecuta la aplicación de Streamlit.

Para asegurarte de que las dependencias se instalen correctamente, sigue estos pasos:

Verifica que el archivo requirements.txt se encuentre en la raíz de tu proyecto en GitHub y que contenga la línea statsmodels.

Asegúrate de que estás utilizando un entorno virtual (como venv o conda) para ejecutar la aplicación de Streamlit. Esto ayudará a mantener las dependencias separadas del sistema global de Python.

Activa tu entorno virtual antes de ejecutar la aplicación de Streamlit. Si estás utilizando venv, puedes activarlo con el siguiente comando (en Windows):

Copy code
venv\Scripts\activate
O en sistemas basados en Unix (Linux/Mac):

bash
Copy code
source venv/bin/activate
Después de activar el entorno virtual, asegúrate de estar en la ubicación correcta de tu proyecto y luego ejecuta el siguiente comando para instalar las dependencias:

Copy code
pip install -r requirements.txt
Esto instalará todas las bibliotecas especificadas en requirements.txt, incluida statsmodels.

Una vez que las dependencias se hayan instalado correctamente, puedes ejecutar la aplicación de Streamlit con el siguiente comando:

arduino
Copy code
streamlit run streamlit_app.py
Esto iniciará la aplicación de Streamlit y debería resolver el error "ModuleNotFoundError" relacionado con statsmodels.

Si sigues enfrentando problemas después de seguir estos pasos, verifica que estás utilizando el entorno virtual adecuado y que las dependencias se instalan correctamente. Si es posible, proporciona más detalles sobre tu entorno de desarrollo y cómo estás ejecutando la aplicación de Streamlit para que pueda ayudarte de manera más específica.








#----------------------------------------------------------------------------------------------------------------------------------------------




