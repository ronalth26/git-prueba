import pandas as pd

nombre_variables = ['POB_URBANA', 'QRESIDUOS_DOM']
# Carga de datos desde la web: descarga el CSV y lo carga como DataFrame

data = pd.read_csv('https://raw.githubusercontent.com/ronalth26/git-prueba/master/lista-residuos.csv',sep=';',encoding='iso-8859-1'
                 )
data

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.plot(data['POB_URBANA'],data['QRESIDUOS_DOM'],'.', color='green')
plt.title('Generación de residuos domiciliarios para una población urbana')
plt.xlabel('QRESIDUOS_DOM')
plt.ylabel('POB_URBANA')
plt.show()
