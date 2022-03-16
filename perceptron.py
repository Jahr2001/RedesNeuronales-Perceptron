from cProfile import label
from matplotlib import markers
import numpy as np
import math

import matplotlib.pyplot as plt
# #Entrada, tabla de verdad de 2 variables
X = np.array([[1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1],
              ])

Y = [0 for i in range(len(X)-1)]
Y.append(1)

X = X.transpose()
Y = np.array(Y)

def entranamiento_Neurona(n, wk):

    k = 0

    errores = []
    generaciones = []

    while(True):
        k += 1
        uk = np.dot(wk, X)
        yck = np.array([0 if uk[0][i] < 0 else 1 for i in range(len(uk[0]))])
        ek = Y - yck

        temp = np.dot(X, ek) * n

        wt = wk + temp

        cont = 0

        for i in range(len(ek)):
            cont += ek[i]**2

        wk = wt
        errores.append((math.sqrt(cont)))
        generaciones.append(k)
    

        if np.all(yck == Y):
            return errores, generaciones, list(wk[0])
    print(f'K: {k}')


def entrada(wk, ns):
    curvas = []
    for i in range(len(ns)):
        curvas.append(entranamiento_Neurona(ns[i], np.array([tuple(wk)])))
    # figure2 = plt.figure(figsize=(15, 7))
    grafica(ns, curvas)

def grafica(ns, curvas):

    grafica = plt.subplot(1, 2, 1)

    grafica.set_title('Gráfica')

    for i in range(len(curvas)):
        marker = 'o'
        grafica.plot(curvas[i][1], curvas[i][0], marker = marker, linestyle = 'solid', label =  f'η{i+1} = {ns[i]}')

    grafica.legend()

    
    tabla = plt.subplot(1, 2, 2)
    tabla.set_title('Tabla')

    tabla.axis('off')

    table = [['η (Taza de aprendizaje)', 'Ultimos pesos de W']]

    for y in range(len(ns)):
        redondear = []
        for f in range(len(curvas[y][2])):
            redondear.append(round(curvas[y][2][f], 3))

        table.append([ns[y], redondear])

    table = tabla.table(cellText=table, loc='center', cellLoc= 'center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)
    plt.show()