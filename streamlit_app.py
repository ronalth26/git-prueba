import pandas as pd

nombre_variables = ['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo', 'ancho_petalo', 'clase']
# Carga de datos desde la web: descarga el CSV y lo carga como DataFrame
iris = pd.read_csv('https://github.com/ronalth26/git-prueba/blob/ccfc74218dfbe9876af6da2e4fc4f4104b37c66d/lista-residuos.csv',
                   names = nombre_variables)
iris




