import pandas as pd

nombre_variables = ['POB_URBANA', 'QRESIDUOS_DOM']
# Carga de datos desde la web: descarga el CSV y lo carga como DataFrame
iris = pd.read_csv('https://raw.githubusercontent.com/ayrna/tutorial-scikit-learn-IMC/master/data/iris.csv',sep=',',encoding='iso-8859-1'
                 )
iris


iris2 = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/master/corto.csv',sep=';',encoding='iso-8859-1'
                 )
iris2.head(1)


iris3 = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/master/corto2.csv',sep=',',encoding='iso-8859-1'
                 )
iris3.head(2)

iris4 = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/3c2bc32f088cba5752937f6d3ad386c58581a150/corto3.csv',sep=',',encoding='iso-8859-1',names = nombre_variables
                 )
iris4


iris5 = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/master/lista-residuos.csv',sep=';',encoding='iso-8859-1'
                 )
iris5
