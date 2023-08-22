#!/usr/bin/env python
# coding: utf-8

# # Práctica Final
# 
# Enhorabuena!!! Ya el haber llegado hasta aquí es un logro más en tu camino para ser un experto del Big Data y del Machine Learning!! 
# 
# 
# <img src="./Images/happy.gif" alt="Drawing" style="width: 300px;"/>
# 
# Con esta práctica pondremos en valor todo lo que hemos visto a lo largo del módulo. Vamos allá!! 😄

# ## 1. Multiconjuntos
# 
# Este ejercicio pondrá a prueba tu habilidad resolver un problema usando vectores.
# 
# **Objetivos**:
# - Usar `Python`
# - Asegurar los fundamentos matemáticos detrás de las operaciones con conjuntos.
# 
# **Problema**: Implementar las operaciones de los multiconjuntos (utilizando las librerías y estructuras de datos vistas en el curso).
# 
# **Datos:**
# 
# Un multiconjunto es un conjunto en el que un elemento puede repetirse, es decir, cada elemento posee una multiplicidad (un número natural) que indica cuántas veces el elemento es miembro del conjunto. Por ejemplo, en el multiconjunto `{a, a, b, b, b, c}`, las multiplicidades de los miembros a, b, y c son 2, 3, y 1, respectivamente.
# 
# Al igual que los conjuntos, poseen las siguientes características y operaciones:
# - Cardinalidad: indica el número de elementos del multiconjunto. Por ejemplo, la cardinalidad del multiconjunto `{a, a, b, b, b, c}` es 6 (la suma de sus multiplicidades).
# - Inserción: permite insertar una ocurrencia de un elemento en el multiconjunto.
# - Eliminación: permite eliminar una ocurrencia de un elemento del multiconjunto.
# - Comparación: compara dos multiconjuntos para determinar si son iguales.
# - Pertenencia: determina si un elemento pertenece al multiconjunto.
# - Subconjunto: determina si un multiconjunto es subconjunto de otro.
# - Unión: conjunción de todos los elementos de dos multiconjuntos (sumando sus multiplicidades si un elementos está en los dos).
# - Intersección: elementos que están en los dos multiconjuntos quedándonos con la multiplicidad más pequeña.
# - Diferencia: restar a un multiconjunto los elementos de otro.

# In[6]:


### TODO: Crear una función que dada una lista devuelva un multiconjunto
### El multiconjunto que devuelve puede crearse con la estructura de datos que se quiera (incluso una lista)
### TU RESPUESTA ABAJO


    #Devuelve un multiconjunto con la lista de elementos dada
    
    #Argumentos:
        #elementos -- lista de elementos
        
    #Ejemplo:
        #elementos = [1,1,1,3,3,1,4,5,1,5]


def calcular_frecuencias(lista):
    frecuencias = {}
    
    for elemento in lista:
        if elemento in frecuencias:
            frecuencias[elemento] += 1
        else:
            frecuencias[elemento] = 1
            
    return frecuencias

# Ejemplo de uso
lista = [1, 1, 1, 3, 3, 1, 4, 5, 1, 5]
resultado = calcular_frecuencias(lista)
print(resultado)


# In[14]:


### TODO: Crear una función que dado un multiconjunto devuelva su cardinalidad
### TU RESPUESTA ABAJO
#Devuelve la cardinalidad del multiconjunto dado
    
    #Argumentos:
        #multiconjunto -- multiconjunto devuelto por la función creada anteriormente
        
def conteo(multiconjunto):
    cuentatodo = len(multiconjunto)
    return total_elementos

lista = [1, 1, 1, 3, 3, 1, 4, 5, 1, 5]
cuentatodo = conteo(lista)
print("El total de elementos en el multiconjunto es:", cuentatodo)


# In[15]:


### TODO: Crear una función que dado un multiconjunto y un elemento devuelva el multiconjunto con el elemento insertado
### TU RESPUESTA ABAJO
#    Devuelve el multiconjunto habiendo insertado el elemento dado    
#    Argumentos: multiconjunto -- multiconjunto devuelto por la función creada anteriormente elemento -- elemento a insertar
    
def insertar_elemento(multiconjunto, elemento):
    nuevo_multiconjunto = multiconjunto.copy()  
    
    if elemento in nuevo_multiconjunto:
        nuevo_multiconjunto[elemento] += 1
    else:
        nuevo_multiconjunto[elemento] = 1
        
    return nuevo_multiconjunto

multiconjunto_original = {1: 4, 3: 2, 5: 2, 4: 1}
elemento_insertar = 9
agregadonueve = insertar_elemento(multiconjunto_original, elemento_insertar)
print("Multiconjunto anterior:", multiconjunto_original)
print("Multiconjunto modificado:", agregadonueve)


# In[16]:


### TODO: Crear una función que dado un multiconjunto y un elemento devuelva el multiconjunto con el elemento eliminado
### TU RESPUESTA ABAJO

   # Devuelve el multiconjunto habiendo eliminado una ocurrencia del elemento dado
    
  #  Argumentos:
 #       multiconjunto -- multiconjunto devuelto por la función creada anteriormente
#        elemento -- elemento a eliminar
    
def delete(multiconjunto, elemento):
    nuevo_multiconjunto = multiconjunto.copy()  
    
    if elemento in nuevo_multiconjunto:
        if nuevo_multiconjunto[elemento] == 1:
            del nuevo_multiconjunto[elemento]
        else:
            nuevo_multiconjunto[elemento] -= 1
    else:
        print("El elemento no existe en el multiconjunto.")
        
    return nuevo_multiconjunto

multiconjunto_modificado = {1: 4, 3: 2, 5: 2, 4: 1, 9: 1}
elemento_eliminar = 9
multiconjunto_modificado1 = delete(multiconjunto_modificado, elemento_eliminar)
print("Multiconjunto anterior:", multiconjunto_modificado)
print("Multiconjunto modificado:", multiconjunto_modificado1)


# In[21]:


### TODO: Crear una función que dado un multiconjunto y un elemento devuelva si el elemento pertenece al multiconjunto
### TODO: Crear una función que dados dos multiconjuntos devuelva si el primero es subconjunto del segundo
### TODO: Crear una función que dados dos multiconjuntos devuelva si son iguales
### TU RESPUESTA ABAJO

#############################################################parte1
def pertenece_elemento(multiconjunto, elemento):
    return elemento in multiconjunto


multiconjunto_original = {1: 4, 3: 2, 5: 2, 4: 1}
elemento_verificar = 3
pertenencia = pertenece_elemento(multiconjunto_original, elemento_verificar)

if pertenencia:
    print(f"El elemento {elemento_verificar} pertenece al multiconjunto.")
else:
    print(f"El elemento {elemento_verificar} no pertenece al multiconjunto.")

    #########################################################PARTE 2


    #Devuelve si multiconjunto1 es subconjunto de multiconjunto2
    
  #  Argumentos:
 #       multiconjunto1 -- multiconjunto devuelto por la función creada anteriormente
#        multiconjunto2 -- multiconjunto devuelto por la función creada anteriormente


def es_subconjunto(subconjunto, conjunto):
    for elemento, frecuencia in subconjunto.items():
        if elemento not in conjunto or conjunto[elemento] < frecuencia:
            return False
    return True

# Ejemplo de uso
multiconjunto_1 = {1: 2, 3: 1}
multiconjunto_2 = {1: 4, 3: 2, 5: 2, 4: 1}

if es_subconjunto(multiconjunto_1, multiconjunto_2):
    print("El primer multiconjunto es un subconjunto del segundo.")
else:
    print("El primer multiconjunto no es un subconjunto del segundo.")     
    
    
    
##############################################################PARTE 3    

    
#    Devuelve si multiconjunto1 es igual a multiconjunto2
    
#    Argumentos:
#        multiconjunto1 -- multiconjunto devuelto por la función creada anteriormente
#        multiconjunto2 -- multiconjunto devuelto por la función creada anteriormente

def soniguales(multiconjunto1, multiconjunto2):
    return multiconjunto1 == multiconjunto2

# Ejemplo de uso
multiconjunto_1 = {1: 4, 3: 2, 5: 2, 4: 1}
multiconjunto_2 = {1: 4, 3: 2, 5: 2, 4: 1}

if soniguales(multiconjunto_1, multiconjunto_2):
    print("Los multiconjuntos son iguales.")
else:
    print("Los multiconjuntos no son iguales.")


# In[22]:


### TODO: Crear una función que dados dos multiconjuntos devuelva su unión
### TODO: Crear una función que dados dos multiconjuntos devuelva su intersección
### TODO: Crear una función que dados dos multiconjuntos devuelva su diferencia
### TU RESPUESTA ABAJO

#PARTE 1, UNION

def union(multiconjunto1, multiconjunto2):
    resultado = {}
    
    for e in multiconjunto1:
        resultado[e] = max(multiconjunto1.get(e, 0), multiconjunto2.get(e, 0))
    
    for e in multiconjunto2:
        resultado[e] = max(multiconjunto1.get(e, 0), multiconjunto2.get(e, 0))
    
    return resultado

# Ejemplo de uso
multiconjunto_1 = {1: 5, 3: 2, 4: 1, 5: 2}
multiconjunto_2 = {1: 6, 3: 3, 4: 2}

union_resultante = union(multiconjunto_1, multiconjunto_2)
print("Unión de multiconjuntos:", union_resultante)

#PARTE 2, INTERSECCION

def interseccion_multiconjuntos(multiconjunto1, multiconjunto2):
    interseccion = {}
    
    for elemento in multiconjunto1:
        if elemento in multiconjunto2:
            interseccion[elemento] = min(multiconjunto1[elemento], multiconjunto2[elemento])
    
    return interseccion

multiconjunto_1 = {1: 5, 3: 2, 4: 1, 5: 2}
multiconjunto_2 = {1: 1, 3: 1, 4: 2}

interseccion_resultante = interseccion_multiconjuntos(multiconjunto_1, multiconjunto_2)
print("Intersección de multiconjuntos:", interseccion_resultante)

#PARTE 3, DIF

def diferencia_multiconjuntos(multiconjunto1, multiconjunto2):
    diferencia = {}
    
    for elemento in multiconjunto1:
        if elemento not in multiconjunto2:
            diferencia[elemento] = multiconjunto1[elemento]
        else:
            diferencia[elemento] = max(multiconjunto1[elemento] - multiconjunto2[elemento], 0)
    
    return diferencia

multc1 = {1: 5, 3: 2, 4: 1, 5: 2}
multc2 = {1: 1, 3: 1, 4: 2}

diferencia_resultante = diferencia_multiconjuntos(multc1, multc2)
print("Diferencia de multiconjuntos:", diferencia_resultante)


# ## 2. Singular Value Decomposition
# 
# Este ejercicio pondrá a prueba tu habilidad para usar Singular Value Decomposition para comprimir una imagen.
# 
# **Objetivos**
# - Usar `Python`
# - Entender los fundamentos de `SVD`.
# 
# **Problema:** Usar `SVD` para comprimir una imagen en blanco y negro.

# La imagen que deberas usar es la siguiente:

# In[4]:


import matplotlib.pyplot as plt
from scipy import misc
get_ipython().run_line_magic('matplotlib', 'inline')

# Load image
A = misc.face(gray=True)

plt.imshow(A, cmap=plt.cm.gray)


# Deberas crear tu propia función para calcular el error de reconstrucción, que viene definido por:
# 
# $$SSE =  \sum_{n}^{i=1}  \begin{Vmatrix}x_{i} -  \widehat{x}_i \end{Vmatrix} ^2 $$
# 
# Donde:
# 
# - $x_i$ son los valores de la matriz original X
# - $\widehat{x}_i$ son los valores de la matriz reconstruida

# In[5]:


### TODO: Función para calcular el error de reconstrucción
### TU RESPUESTA ABAJO

import numpy as np

def sse_score(X, X_hat):
    return np.sum((X - X_hat)**2)

X = np.array([[1, 2], [3, 4]])
X_hat = np.array([[1.01, 1.75], [2.81, 3.99]])
sse = sse_score(X, X_hat)
print(sse)


# Una vez que ya tenemos la función `sse` hecha, podemos pasar a construir la función que ejecutará `SVM`.

# In[6]:


### TODO: Función para ejecutar SVM
### Tiene como entrada una matriz X
### Devuelve U, s, Vt

### Hint: S debe ser una matriz diagonal
### TU RESPUESTA ABAJO


def svm(X):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    S = np.diag(s)
    return U, S, Vt

X = np.array([[1, 2], [3, 4]])
U, S, Vt = svm(X)  
#se guardará la funci+on y se imprimen las variables/matrices debajo
    


# In[7]:


U


# In[8]:


S


# In[9]:


Vt


# Como hemos visto en clase, las matrices obtenidas a partir de `SVM` nos sirven para reconstruir la matriz original `X`. Para ello, construye una función que permita reconstruir la matriz original `X` a partir de `U, s, Vt`.

# In[10]:


### TODO: Función para reconstruir la matriz original a partir de U, s, Vt
### Tiene como entrada U, s, Vt
### Devuelve X_hat
### TU RESPUESTA ABAJO


import numpy as np

def reconstruction(U, S, Vt):
    X_hat = U @ S @ Vt
    return X_hat

# Ejemplo de uso
U = np.array([[-0.40455358, -0.9145143 ],
              [-0.9145143 ,  0.40455358]])
S = np.array([[5.4649857 , 0.        ],
              [0.        , 0.36596619]])
Vt = np.array([[-0.57604844, -0.81741556],
               [ 0.81741556, -0.57604844]])

X_hat = reconstruction(U, S, Vt)
print("Matriz X_hat:")
print(X_hat)
    


# Calcula el error de reconstrucción usando la función `sse` que has programado anteriormente.

# In[11]:


wrongreconst = sse_score(X, X_hat)
print("Error de reconstrucción:", wrongreconst)


# Una vez que hemos programado todas las funciones necesarias para realizar `SVM` y medir el error de reconstrucción, podemos proceder a realizar la compresión de la imagen. Esta [página web](http://timbaumann.info/svd-image-compression-demo/) te ayudará a repasar y a entender como calcular la compresión.
# 
# Debes usar la siguiente imagen: 

# In[12]:


# Load image
A = misc.face(gray=True)

plt.imshow(A, cmap=plt.cm.gray)


# In[19]:


### TODO: Función que recibe una imagen A y devuelve la imagen comprimida
### Tiene como entrada A y el número de componentes para realizar la reducción de dimensionalidad
### Devuelve la imagen comprimidad, el error de reconstrucción y el ratio de compresión

### Hint: Usa las funciones anteriormente construidas
### TU RESPUESTA ABAJO

    
def image_compression(A, n_components):
    """
    Función para comprimir una imagen A
    
    Argumentos:
        A -- Imagen original
        n_components -- Número de componentes
        
    Ejemplo:
        A_hat, sse, comp_ratio = image_compression(A, n_components=50)
    """
    
    # Calcular U, s, Vt
    U, S, Vt = svm(A)
    
    A_hat = reconstruction(U[0:U.shape[0], 0:n_components], 
                           S[0:n_components,0:n_components], 
                           Vt[0:n_components, 0:Vt.shape[1]])
            
    sse = np.sum((A - A_hat)**2) 
    comp_ratio = (A.shape[1]*n_components + n_components + A.shape[0]*n_components)/(A.shape[1] * A.shape[0])
    
    return A_hat, sse, comp_ratio

A_hat, sse, comp_ratio = image_compression(A, n_components=50)

print(f"Margen de reconstrucción: {sse}")
print(f"Comprensión: {comp_ratio}")

plt.figure(figsize=(15,10)) 
plt.subplot(121)
plt.imshow(A, cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(122)
plt.imshow(A_hat, cmap=plt.cm.gray)
plt.title('Imagen comprimida')
plt.show()


# Grafica la imagen original `X` y la imagen reconstruida `X_hat`, y imprime el error de reconstrucción `sse` y el `ratio de compresion`.

# ## 3. Linear Regression - Least Squares
# 
# Este ejercicio pondrá a prueba tu habilidad para programar tu propia versión de mínimos cuadrados en Python.
# 
# **Objetivos**:
# - Usar `Python` + `Pandas` para leer y analizar los datos.
# - Asegurar los fundamentos matemáticos detrás del método de los mínimos cuadrados.
# 
# **Problema**: Usando datos sobre el precio de la vivienda, intentaremos predecir el precio de una casa en base a la superficie habitable con un modelo de regresión.
# 
# **Datos:** [Kaggle's House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

# ### Repaso
# 
# Usaremos la versión matricial de la solución de los **métodos de los mínimos cuadrados** para resolver este problema. Como recordatorio, expresamos los coeficientes $w_{LS}$ como un vector, y calculamos ese vector en base a la matriz de entrada $X$ y en base a $y$.<br><br>
# 
# 
# 
# Como mostramos en clase, la matriz $X$ siempre contiene un vector de valores $1$ en la primera columna. En otras palabras:<br><br>
# 
# <center>$
# X = \begin{bmatrix}
# 1 \  x_{11}  \\
# 1 \  x_{21}  \\
# \vdots \ \vdots \\
# 1 \ x_{n1}
# \end{bmatrix} 
# $</center>
# 
# Para dos variables, $X$ tomará esta forma:
#  
# <center>$
# X = \begin{bmatrix}
# 1 \  x_{11} \  x_{12} \\
# 1 \  x_{21} \  x_{22} \\
# \vdots \ \vdots \\
# 1 \ x_{n1} \  x_{n2}
# \end{bmatrix} 
# $</center>
# 
# ### Exploratorio de datos

# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)


# In[34]:


### Leer los datos
tr_path = r'C:\Users\naesp\agc\train.csv'
data = pd.read_csv(tr_path)


# In[28]:


import os

cwd = os.getcwd()

nombre_archivo = 'train.csv'

ruta_completa = os.path.join(cwd, nombre_archivo)

print("Ruta completa del archivo:", ruta_completa)


# In[35]:


### La función .head() muestras las primeras lineas de los datos
data.head()


# In[36]:


### Lista con los nombres de las columnas
data.columns


# In[38]:


### TODO: Numero de columnas 
### Asignar int variable a: ans1
### TU RESPUESTA ABAJO
numcolumn = len(data.columns)
print('Len columns: ', numcolumn)
ans1 = numcolumn


# #### Visualizaciones

# In[39]:


### Podemos graficar los datos price vs living area - Matplotlib

Y = data['SalePrice']
X = data['GrLivArea']

plt.scatter(X, Y, marker = "x")

### Anotaciones
plt.title("Sales Price vs. Living Area (excl. basement)")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice");


# In[40]:


### price vs year - Pandas

data.plot('YearBuilt', 'SalePrice', kind = 'scatter', marker = 'x');


# ### Regresión Lineal
# 
# Ya que conocemos la ecuación para $w_{LS}$ tenemos todo lo necesario para resolver la regresión lineal. Vamos allá!<br><br>
# 
# <center>$w_{LS} = (X^T X)^{-1}X^T y,$</center>
# 

# In[44]:


### TODO: Función para invertir una matriz
### Contruye una función que toma como input una matriz
### Devuelve la inversa de dicha matriz
### TU RESPUESTA ABAJO

def matinversa(mat):
    
#Calcula y devuelve la inversa de la matriz
    
#    Argumentos:
#        mat -- Matriz cuadrada a invertir
        
#    Ejemplo:
#        sample_matrix = [[1, 2], [3, 4]]
#        the_inverse = inverse_of_matrix(sample_matrix)  
        # -> the_inverse = [[-2.   1. ]
        #                   [ 1.5 -0.5]]
    
#Requerimientos:
#        Esta función depende de 'numpy.linalg.inv'
    
        matrizinversa = np.linalg.inv(mat)
        return matrizinversa

sample_matrix = [[1, 2], [3, 4]]
the_inverse = matinversa(sample_matrix)  
the_inverse


# #### Leer los datos
# 
# Lo primero que debemos hacer es leer los datos, para ello construye una función que reciba el directorio de un archivo .csv `file_path` y lo lea usando `pandas`, la función debe devolver el dataframe.

# In[45]:


### TODO: Función para leer un .csv
### La función recibe un file_path y debe devolver el dataframe
### TU RESPUESTA ABAJO

import pandas as pd

def read_to_df(file_path):
    """Leer un archivo .csv"""
    dataframe = pd.read_csv(file_path)

dataframe = read_to_df(tr_path)


# #### Subset del dataframe por columnas
# 
# Queremos construir una función que nos permita obtener los datos de ciertas columnas. Por ello, le pasaremos como argumento un `dataframe` y una lista con los nombres de las columnas que queremos extraer `column_names` y nos devolverá un dataframe con solo esas columnas.

# In[48]:


### TODO: Función para extraer los datos de ciertas columnas
### Como argumentos, recibe un dataframe `data_frame`y una lista con los nombres de las columnas `column_names`
### Devuelve un dataframe con solo las columnas que le hemos especificado
### TU RESPUESTA ABAJO

def select_columns(data_frame, column_names):
    """Devuelve un subset del dataframe en base a los nombres de las columnas
    
    Argumentos:
        data_frame -- Dataframe Object
        column_names -- Lista con los nombres de las columnas a seleccionar
        
    Ejemplo:
        data = read_into_data_frame('train.csv')
        selected_columns = ['SalePrice', 'GrLivArea', 'YearBuilt']
        sub_df = select_columns(data, selected_columns)
    """
    if len(column_names) != 0: 
        df_columns = data_frame[column_names]
        return df_columns
    else:
        print('No columns detected, please, insert at least a column')
        return 
    
data = pd.read_csv(tr_path)
selected_columns = ['SalePrice', 'GrLivArea', 'YearBuilt']
sub_df = select_columns(data, selected_columns)


# #### Subset del dataframe por valores
# 
# El siguiente paso es construir una función que recibe un `data_frame`, el nombre de una columna, un valor mínimo y un valor máximo `cutoffs`. Nos devuelve un dataframe excluyendo las filas donde el valor de la columna indica está fuera de los valores mínimos y máximos que le hemos indicado.

# In[52]:


### TODO: Función para crear un nuevo subset en base a valores
### Como argumento recibe un dataframe y una lista de tuples
### Tuples: (column_name, min_value, max_value)
### Devuelve un dataframe que excluye las filas donde los valores, en la columna que le hemos indicado, exceden los valores
### que le hemos indicado
### No eliminar la fila si los valores son iguales al min/max valor
### TU RESPUESTA ABAJO

def column_cutoff(data_frame, cutoffs):
    """Crea un nuevo dataframe en base a unos límites
    
    Argumentos:
        data_frame -- Dataframe Object
        cutoffs -- Lista de tuples con el siguiente formato:
        (column_name, min_value, max_value)
        
    Ejemplo:
        data_frame = read_into_data_frame('train.csv')
        # Remove data points with SalePrice < $50,000
        # Remove data points with GrLiveAre > 4,000 square feet
        cutoffs = [('SalePrice', 50000, 1e10), ('GrLivArea', 0, 4000)]
        selected_data = column_cutoff(data_frame, cutoffs)
    """
    
    for i in range(len(cutoffs)):
        column = cutoffs[i][0]
        min_value = cutoffs[i][1]
        max_value = cutoffs[i][2]
        df_cutoff = data_frame[(data_frame[column] >= min_value) & (data_frame[column] <= max_value)]

        data_frame = df_cutoff
    return df_cutoff

data_frame = pd.read_csv(tr_path)
columnas = [('SalePrice', 50000, 1e10), ('GrLivArea', 0, 4000)]
salesylivearea = column_cutoff(data_frame, columnas)

salesylivearea 


# #### Mínimos Cuadrados / Least Squares
# 
# Ahora, implementarás la ecuación $w_{LS}$:
# 
# <center>$w_{LS} = (X^T X)^{−1}X^T y,$</center>

# In[55]:


### TODO: Función para resolver la ecuación wLS
### Toma como argumentos dos matrices, una para X y otra para y
### Asumimos que las matrices tienen las dimensiones correctas

### Paso 1: Asegurate que n > d. 
### Es decir, que el número de observaciones es mayor que el número de dimensiones.
### O lo que es lo mismo, que el número de filas de cada matriz sea mayor que el número de columnas
### Si no es así, debes transponer las matrices

### Paso 2: Debes añadir a la matriz X un vector columna del tamaño (n x 1)

### Paso 3: Usa la ecuación de arriba para obtener wLS

### TU RESPUESTA ABAJO


def least_squares_weights(input_x, target_y):
    """Resuelve la ecuación para wLS
    
    Argumentos:
        input_x -- Matriz con los datos de entrenamiento
        target_y -- Vector con los datos de salida
        
    Ejemplo:
        import numpy as np
        training_y = np.array([[208500, 181500, 223500, 
                                140000, 250000, 143000, 
                                307000, 200000, 129900, 
                                118000]])
        training_x = np.array([[1710, 1262, 1786, 
                                1717, 2198, 1362, 
                                1694, 2090, 1774, 
                                1077], 
                               [2003, 1976, 2001, 
                                1915, 2000, 1993, 
                                2004, 1973, 1931, 
                                1939]])
        weights = least_squares_weights(training_x, training_y)
        
        print(weights)  #--> np.array([[-2.29223802e+06],
                        #              [ 5.92536529e+01],
                        #              [ 1.20780450e+03]])
                           
        print(weights[1][0]) #--> 59.25365290008861
    
    Asumimos:
        -- target_y es un vector con el mismo número de observaciones que input_x
    """
    x_shape = input_x.shape
    y_shape = target_y.shape
   
    if x_shape[0] < x_shape[1]:
        input_x = np.transpose(input_x)
    if y_shape[0] < y_shape[1]:
        target_y = np.transpose(target_y)
    num = input_x.shape[0]
    vct = np.ones((num, 1))
#concatenras las los v y d
    input_X = np.concatenate((vct, input_x), axis=1)

    vct1 = np.dot(np.linalg.inv(np.dot(input_X.T, input_X)), np.dot(input_X.T, target_y))
    return vct1

    
import numpy as np
training_y = np.array([[208500, 181500, 223500, 
                        140000, 250000, 143000, 
                        307000, 200000, 129900, 
                        118000]])
training_x = np.array([[1710, 1262, 1786, 
                        1717, 2198, 1362, 
                        1694, 2090, 1774, 
                        1077], 
                       [2003, 1976, 2001, 
                        1915, 2000, 1993, 
                        2004, 1973, 1931, 
                        1939]])
weights = least_squares_weights(training_x, training_y)

print(weights)                 
print(weights[1][0]) 


# #### Testing en datos reales
# 
# Ahora que ya hemos programado todas las funciones necesarias para calcular la regresión lineal vamos a aplicar al conjunto de datos que habíamos seleccionado al principio. 
# 
# **Datos:** [Kaggle's House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
# 
# Si tus funciones están correctamente programadas, la siguiente celda correrá sin problemas 😃

# In[60]:


test_path = 'C:\\Users\\naesp\\agc\\train.csv'
df = pd.read_csv(test_path)
df_sub = select_columns(df, ['SalePrice', 'GrLivArea', 'YearBuilt'])

cutoffs = [('SalePrice', 50000, 1e10), ('GrLivArea', 0, 4000)]
df_sub_cutoff = column_cutoff(df_sub, cutoffs)

X = df_sub_cutoff['GrLivArea'].values
Y = df_sub_cutoff['SalePrice'].values

### reshaping for input into function
training_y = np.array([Y])
training_x = np.array([X])

weights = least_squares_weights(training_x, training_y)
print(weights)


# In[61]:


max_X = np.max(X) + 500
min_X = np.min(X) - 500

### Choose points evenly spaced between min_x in max_x
reg_x = np.linspace(min_X, max_X, 1000)

### Use the equation for our line to calculate y values
reg_y = weights[0][0] + weights[1][0] * reg_x

plt.plot(reg_x, reg_y, color='#58b970', label='Regression Line')
plt.scatter(X, Y, c='k', label='Data')

plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.legend()
plt.show()


# #### Implementación con sklearn
# 
# Podemos comprobar como el resultado de nuestro código es exactamente igual al resultado de `sklearn`. Enhorabuena! Has programado tu propia **regresión lineal!!** 😃

# In[62]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()

### SKLearn requiere un array 2-dimensional X y 1 dimensional y.
### skl_X = (n,1); skl_Y = (n,)
skl_X = df_sub_cutoff[['GrLivArea']]
skl_Y = df_sub_cutoff['SalePrice']

lr.fit(skl_X,skl_Y)
print("Intercept:", lr.intercept_)
print("Coefficient:", lr.coef_)


# ## 4. Linear Regression - Gradient Descent
# 
# En este ejercicio resolveras el mismo problema anterior pero usando **Gradient Descent**
# 
# **Objetivos**:
# - Asegurar los fundamentos matemáticos detrás del Gradient Descent.
# 
# **Problema**: Usando datos sobre el precio de la vivienda, intentaremos predecir el precio de una casa en base a la superficie habitable con un modelo de regresión.
# 
# **Datos:** [Kaggle's House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
# 
# **Repaso:**
# 
# $$ RSS(w) = \sum_{n=1}^{N}[y_n-f(x_n)]^2 =  \sum_{n=1}^{N}[y_n- (w_0 + \sum_{d=1}^{D}w_dx_{nd}) ]^2 .$$
# 
# Loss function:
# 
# $$ RSS(w) = \frac{1}{2}\sum_{n=1}^{N}[y_n-f(x_n)]^2$$
# 
# Y lo que queremos es minimizar esta distancia, para que el modelo se acerque lo máximo posible a los valores verdaderos.
# 
# $$\nabla RSS(w) = X^T(Xw^t-y)$$
# 
# En resumen, el gradient descendiente para una regresión lineal, se basa en resolver esta ecuación de forma iterativa:
# 
# $$w^{t+1} = w^t - \eta * \nabla RSS(w)$$

# #### Leer Datos

# In[68]:


import pandas as pd
import numpy as np

# Leer datos
data = pd.read_csv(r'C:\Users\naesp\agc\train.csv')

# Extraer dichas columnas
newData = data[['GrLivArea','SalePrice']]
print(newData.head())

# Contruir x - y
x = newData['GrLivArea']
y = newData['SalePrice']

# Standarizar los datos
x = (x - x.mean()) / x.std()
x = np.c_[np.ones(x.shape[0]), x] 

print("Shape of X: ", x.shape)
print("Shape of y:", y.shape)


# #### Gradient Descent

# In[73]:


### TODO: Función para encontrar los valores w usando Gradient Descent
### Toma como argumentos: X, y, w, n_iterations, eta
### Completa la función añadiendo la loss función y la updating rule
### TU RESPUESTA ABAJO

def gradient_descent(x, y, w, iterations, eta):
    """Gradient descent
    
    Argumentos:
        x -- Matriz con los datos de entrenamiento
        y -- Vector con los datos de salida
        w -- Vector aleatoriamente inicializado
        iterations -- Número de iteraciones
        eta -- Learning Rate
        
    Ejemplo:
        import numpy as np

        # Learning rate
        eta = 0.01 

        # Número de iteraciones
        iterations = 2000 #No. of iterations

        # Seed para inicializar w
        np.random.seed(123)
        w0 = np.random.rand(2)
        
        training_y = np.array([208500, 181500, 223500, 
                                140000, 250000])
        training_x = np.array([[ 1.        ,  0.37020659],
                               [ 1.        , -0.48234664],
                               [ 1.        ,  0.51483616],
                               [ 1.        ,  0.38352774],
                               [ 1.        ,  1.29888065]])
                            
        weights, loss = gradient_descent(training_x, training_y, w0, iterations, eta)
        
        print(weights[-1])  #--> np.array([183845.82320222  40415.66453324])
    """


    losses = []
    weights = [w]
    n = y.size
    
    for _ in range(iterations):
        prediction = np.dot(x, w)
        error = prediction - y
        
        # Cálculo de la función de pérdida
        loss = 0.5 * np.dot(error.T, error)
        losses.append(loss)
        
        # Cálculo del gradiente
        gradient = np.dot(x.T, error)
        
        # Actualización de los pesos
        w -= eta * gradient
        weights.append(w.copy())
    
    return weights, losses

# Learning rate
eta = 0.01 

# Número de iteraciones
iterations = 2000 #No. of iterations

# Seed para inicializar w
np.random.seed(123)
w0 = np.random.rand(2)

training_y = np.array([208500, 181500, 223500, 
                        140000, 250000])
training_x = np.array([[ 1.        ,  0.37020659],
                       [ 1.        , -0.48234664],
                       [ 1.        ,  0.51483616],
                       [ 1.        ,  0.38352774],
                       [ 1.        ,  1.29888065]])

weights, losses = gradient_descent(training_x, training_y, w0, iterations, eta)

print(weights[-1])  
    
    
    
    


# Una vez construida nuestra función para el Gradient Descent podemos usarla para encontrar los valores optimos de $w$. **Prueba a modificar el learning rate para ver la convergencia del Gradient Descent.**

# In[74]:


import numpy as np

# Learning rate
eta = 0.001 

# Número de iteraciones
iterations = 1000 #No. of iterations

# Seed para inicializar w
np.random.seed(123)
w0 = np.random.rand(2)

weights, loss = gradient_descent(x, y, w0, iterations, eta)

print(weights[-1])


# In[75]:


type(x)


# In[77]:


import matplotlib
print(matplotlib.matplotlib_fname())


# Hemos creado la siguiente función para ver como Gradient Descent encuentra el resultado final - **Tarda un poco**

# In[78]:


import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Definir figure
fig = plt.figure()
ax = plt.axes()
plt.title('Sale Price vs Living Area')
plt.xlabel('Living Area in square feet (normalised)')
plt.ylabel('Sale Price ($)')
plt.scatter(x[:,1], y, color='red')
line, = ax.plot([], [], lw=2)
annotation = ax.text(-1, 700000, '')
annotation.set_animated(True)
plt.close()

# Generar animacion de los datos
def init():
    line.set_data([], [])
    annotation.set_text('')
    return line, annotation

# Función para la animación
def animate(i):
    x = np.linspace(-5, 20, 1000)
    y = weights[i][1]*x + weights[i][0]
    line.set_data(x, y)
    annotation.set_text('loss = %.2f e10' % (loss[i]/10000000000))
    return line, annotation

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=300, interval=0, blit=True)

anim.save('animation.gif', writer='imagemagick', fps = 30)

# Visualizar la animación
import io
import base64
from IPython.display import HTML

filename = 'animation.gif'

video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


# ## (Opcional) - Calculando similitud entre páginas web
# 
# Este ejercicio pondrá a prueba tu capacidad para encontrar la similitud entre vectores usando cosine similarity.
# 
# **Objetivos**:
# - Usar `Python` + `BeautifulSoup` para "scrapear" páginas webs.
# - Asegurar los fundamentos matemáticos detrás del cosine similarity.
# 
# **Problema**: Dadas N páginas web, extraer el texto de ellas y determinar la similitud.
# 
# ### Repaso
# 
# Como recordarás, podemos medir la similitud entre vectores usando la siguiente ecuación:<br>
# 
# <center>$\overrightarrow{u} \cdot \overrightarrow{v} = |\overrightarrow{u}||\overrightarrow{v}| \cos \theta $</center>
# 
# Que podemos reescribir de la siguiente forma:<br>
# 
# <center>$\cos \theta = \frac{\overrightarrow{u} \cdot \overrightarrow{v}}{|\overrightarrow{u}||\overrightarrow{v}|}$</center>
# 
# La **similitud** va a venir dada por el ángulo $\theta$, que nos indicará lo siguiente:
# 
# <img src="./Images/cosine_sim.png" width=70%/>
# 
# ### Web scraping
# 
# La técnica llamada `web scraping` es la utilizada normalmente para extraer contenido de páginas webs y posteriormente procesarlos. Por ejemplo, si quisieramos construir una base de datos para entrenar un modelo con imágenes de ropa para hombres, podríamos intentar "scrapear" dicha sección de la página web del El Corte Inglés para conseguir las imágenes (no es tan fácil como suena).

# In[ ]:


# Estas librerias deben ser instaladas para hacer este ejercicio
get_ipython().system('pip install beautifulsoup4')
get_ipython().system('pip install lxml')


# In[ ]:


import re
import lxml
from bs4 import BeautifulSoup
import urllib
import urllib.request

url = "https://es.wikipedia.org/wiki/Canis_lupus_familiaris"

def parse_from_url(url):
    """
    Función para extraer el contenido (raw text) de una página web
    """
    
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, "html.parser" )
    for script in soup(["script", "style"]):
        script.extract()
        
    text = soup.get_text()
    
    # Eliminar saltos de linea
    text = re.sub('\s+', ' ', text)
    return text

parse_from_url(url)[:1000]


# In[ ]:


### TODO: Escribe una función que reciba una lista de urls
### Aplica web scraping a cada una de ellas para extraer el contenido
### Y devuelva un diccionario con el contenido por cada url

### HINT: Usa la función anterior
### NOTE: Suele tardar un poco en extraer el contenido de las paginas web
### TU RESPUESTA ABAJO

def get_content(url_ls):
    """Extrae el contenido de una lista de urls
    
    Argumentos:
        url_ls -- Lista con urls
        
    Ejemplo:
        url_ls = ['https://es.wikipedia.org/wiki/Canis_lupus_familiaris', 
        'https://es.wikipedia.org/wiki/Canis_lupus',
        'https://es.wikipedia.org/wiki/Felis_silvestris_catus']
        
        url2content = get_content(url_ls)  
    
    Requerimientos:
        Esta función depende de 'parse_from_url()'
    """


# ### Preprocesado
# 
# Como es lógico, no podemos resolver esta ecuación $\cos \theta = \frac{\overrightarrow{u} \cdot \overrightarrow{v}}{|\overrightarrow{u}||\overrightarrow{v}|}$ usando texto sin más, debemos convertir cada página web a un vector.

# In[ ]:


### TODO: Escribe una función que reciba texto
### Y devuelva una lista con el texto separado por espacios
### Además del set de la lista
### "hola que que tal" - ["hola", "que", "que", "tal"], {"hola", "que", "tal"}
### TU RESPUESTA ABAJO

def tokenizer(text):
    """Divide el texto en palabras
    
    Argumentos:
        text -- String
        
    Ejemplo:
        url_ls = "Hola me llamo llamo Alex y estamos aprendiendo Algebra y estamos bien"
        
        tokens_txt, set_txt = tokenizer(url_ls)  
    
    Requerimientos:
        No uses ningún tokenizer ya implementado (nltk, spacy, ...)
    """


# El siguiente paso es crear un conjunto con las palabras de ambas páginas web (unión), por ejemplo:
# 
# - Los perros son maravillosos...
# - Los maravillosos años 80...
# 
# Por tanto, el conjunto para estas dos frases sería `{"los", "perros", "son", "maravillosos", "años", "80"}`. Debemos realizar esto para todas las combinaciones posibles, es decir:
# 
# - web_1
# - web_2
# - web_3
# 
# En este caso, las combinaciones serían (no importa el orden) `[web_1, web_2]`, `[web_1, web_3]`, `[web_2, web_3]`

# In[ ]:


### TODO: Escribe una función que recibe una lista de N páginas web
### Y calcula todas las combinaciones posibles entre ellas, no importa el orden
### [web_1, web_2, web_3, ...]
### Devuelve una lista de tuples con las combinaciones [(web_1, web_2), (web_1, web_3), ...]

# HINT: Puedes implementar esta función como quieras pero la librería itertools 
#       proporciona una función llamada `combinations` para realizar esta tarea.

### TU RESPUESTA ABAJO
import itertools
    
def combinations(url_ls):
    """Calcula todas las combinaciones posibles entre los elementos de una lista
    
    Argumentos:
        url_ls -- Lista de urls
        
    Ejemplo:
        url_ls = ['https://es.wikipedia.org/wiki/Canis_lupus_familiaris', 
        'https://es.wikipedia.org/wiki/Canis_lupus',
        'https://es.wikipedia.org/wiki/Felis_silvestris_catus']
        
        permutation = combinations(url_ls)  
    
    Requerimientos:
        Puedes implementar esta función como quieras pero la librería itertools 
        proporciona una función llamada `combinations` para realizar esta tarea.
    """


# In[ ]:


### TODO: Escribe una función que recibe una lista con tuples
### [({'que', 'hola'}, {'que', 'es', 'guay'}), ({'que', 'hola'}, {'madrid', 'la', 'es'})]
### Y devuelve una lista con la union de los conjuntos
### [({'que', 'hola', 'es', 'guay'}), ({'que', 'hola', 'madrid', 'la', 'es'})]
### TU RESPUESTA ABAJO

def union(comb_ls):
    """Calcula la unión por cada tuple de una lista 
    
    Argumentos:
        comb_ls -- Lista de tuples
        
    Ejemplo:
        comb_ls = [({'que', 'hola'}, {'que', 'es', 'guay'}), ({'que', 'hola'}, {'madrid', 'la', 'es'})]
        
        union_ls = union(comb_ls)  # -> [{'es', 'que', 'hola', 'guay'}, {'es', 'que', 'hola', 'madrid', 'la'}]
    """


# Una vez que tenemos una lista de conjuntos por cada par de páginas web, podemos convertir el texto de la página web a un vector.

# In[ ]:


def set2vector(tokens_web1, tokens_web_2, set_web1, set_web2):
    """
    Función para convertir un conjunto a vector
    
    Argumentos:
        tokens_web1 -- Contenido tokenizado de página web 1
        tokens_web_2 -- Contenido tokenizado de página web 2
        set_web1 -- Conjunto de palabras de la página web 1
        set_web2 -- Conjunto de palabras de la página web 2
        
    Ejemplo:
        tokens_web1 = ["hola", "que", "tal", "soy", "Alex"]
        tokens_web_2 = ["hola", "me", "llamo"]
        set_web1 = {"hola", "que", "tal"}
        set_web2 = {"hola", "me", "llamo"}
        union_ls = set2vector(tokens_web1, tokens_web_2, set_web1, set_web2)  
        
    Requerimientos:
        Depende de la función `union()`
    """
    
    # Unimos los conjuntos
    join_set = union([(set_web1, set_web2)])[0]
    
    web1_array = []
    web2_array = [] 

    for word in join_set:
        if word in tokens_web1:
            web1_array.append(1)
        else:
            web1_array.append(0)
        if word in tokens_web_2:
            web2_array.append(1)
        else:
            web2_array.append(0)

    return web1_array, web2_array

tokens_web1 = ["hola", "que", "tal", "soy", "Alex"]
tokens_web_2 = ["hola", "me", "llamo"]
set_web1 = {"hola", "que", "tal", "soy", "Alex"}
set_web2 = {"hola", "me", "llamo"}
web1_array, web2_array = set2vector(tokens_web1, tokens_web_2, set_web1, set_web2)  


# In[ ]:


print(web1_array, web2_array)


# ### Cosine Similarity
# 
# Por último, ya podemos implementar la ecuación: $\cos \theta = \frac{\overrightarrow{u} \cdot \overrightarrow{v}}{|\overrightarrow{u}||\overrightarrow{v}|}$

# In[ ]:


### TODO: Escribe una función que recibe dos vectores, u y v
### Y devuelva la similaridad entre ambos vectores
###
### Paso 1: Si u y v son listas -> Convertirlo a arrays
###
### Paso 2: Calcula la similaridad entre ambos vectores
### TU RESPUESTA ABAJO

import numpy as np

def cosine_similarity(u, v):
    """Calcula la similaridad entre dos vectores
    
    Argumentos:
        u -- Vector 1
        v -- Vector 2
        
    Ejemplo:
        u = np.array([1, 2, 3])
        v = np.array([3, 2, 1])
        
        similarity = cosine_similarity(u, v)  # -> 0.71428
    """


# In[ ]:


def websites_sim(url_ls):
    """Función para calcular la similaridad entre páginas web
    
    Argumentos:
        url_ls -- Listas de páginas web
        
    Ejemplo:
        url_ls = ['https://es.wikipedia.org/wiki/Canis_lupus_familiaris', 
        'https://es.wikipedia.org/wiki/Canis_lupus',
        'https://es.wikipedia.org/wiki/Felis_silvestris_catus']
        
        similarity_ls = websites_sim(url_ls)  
    """
    
    url2content = get_content(url_ls)
    
    # Creamos un diccionario donde cada url tendrá su contenido tokenizado y su conjunto
    url_dict = {}
    for url, content in url2content.items():
        toks, sets = tokenizer(content)
        url_dict[url] = {'tokens': toks,
                        'unique_tokens': sets}
    
    # Calculamos todas las combinaciones posibles de las direcciones de las páginas web
    comb_ls = combinations(url_ls)

    # Usando comb_ls y la función `set2vector()` convertimos cada página web a vectores
    print("Similaridad: ")
    for el in comb_ls:
        # Obtenemos los tokens y el conjunto para cada página web
        token_1 = url_dict[el[0]]['tokens']
        token_2 = url_dict[el[1]]['tokens']
        set_1 = url_dict[el[0]]['unique_tokens']
        set_2 = url_dict[el[1]]['unique_tokens']
        array_web1, array_web2 = set2vector(token_1, token_2, set_1, set_2)
        similarity = cosine_similarity(array_web1, array_web2)
        
        print("{} vs {} - {}".format(el[0], el[1], round(similarity, 3)))

                      
url_ls = ['https://es.wikipedia.org/wiki/Canis_lupus_familiaris', 
'https://es.wikipedia.org/wiki/Canis_lupus',
'https://es.wikipedia.org/wiki/Felis_silvestris_catus']

similarity_ls = websites_sim(url_ls) 


# In[ ]:




