import pandas as pd

nombre_variables = ['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo']
# Carga de datos desde la web: descarga el CSV y lo carga como DataFrame
iris = pd.read_csv('https://raw.githubusercontent.com/ayrna/tutorial-scikit-learn-IMC/master/data/iris.csv'
                 )

iris.head(3)

