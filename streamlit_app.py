import pandas as pd

nombre_variables = ['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo']
# Carga de datos desde la web: descarga el CSV y lo carga como DataFrame
iris = pd.read_csv('https://raw.githubusercontent.com/ayrna/tutorial-scikit-learn-IMC/master/data/iris.csv',sep=',',encoding='iso-8859-1'
                 )
iris


iris2 = pd.read_csv('https://github.com/ronalth26/git-prueba/blob/c08197d989bdbf1a2e60014d230aa0907aa75f26/corto.csv',sep=';',encoding='iso-8859-1'
                 )
iris2


iris3 = pd.read_csv('https://github.com/ronalth26/git-prueba/blob/c08197d989bdbf1a2e60014d230aa0907aa75f26/corto2.csv',sep=',',encoding='iso-8859-1'
                 )
iris3

iris4 = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/3c2bc32f088cba5752937f6d3ad386c58581a150/corto3.csv',sep=',',encoding='iso-8859-1'
                 )
iris4
