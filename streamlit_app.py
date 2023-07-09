import pandas as pd

nombre_variables = ['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo']
# Carga de datos desde la web: descarga el CSV y lo carga como DataFrame
iris = pd.read_csv('https://github.com/ronalth26/git-prueba/blob/df87a3cd8bef7f3a91a136afed2b3c5467fa433c/lista-residuos.csv',sep=';',encoding='iso-8859-1'
                 )
iris


