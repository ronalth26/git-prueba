import pandas as pd





nombre_variables = ['POB_URBANA', 'QRESIDUOS_DOM']
# Carga de datos desde la web: descarga el CSV y lo carga como DataFrame

excel_file  = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/master/lista-residuos.csv',sep=';',encoding='iso-8859-1'
                 )
excel_file 
