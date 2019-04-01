
#%%
import tensorflow as tf 
from tensorflow import keras

import numpy as nump
import matplotlib.pyplot as plot


#%%
digitos = keras.datasets.mnist


#%%
(x, y ),  (x_prueba , y_prueba) = digitos.load_data()


#%%
x=x/255.0
x_prueba=x_prueba/255.0


#%%
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.sigmoid),
    keras.layers.Dense(10, activation=tf.nn.sigmoid)
])


#%%
model.compile(optimizer='adam',  loss='sparse_categorical_crossentropy',metrics=['accuracy'])


#%%
model.fit(x, y, epochs=20)


#%%
model.evaluate(x_prueba,y_prueba)


#%%
predicciones = model.predict(x_prueba)


#%%
def predecir(xx):
    plot.figure()
    plot.imshow(x_prueba[xx])
    plot.xlabel(y_prueba[xx])


#%%
len(y_prueba)


#%%
#ingresa un numero // equivale al numero de array de los datos de prueba 
#tamaño del arrya prueba es de 10000
predecir(15)


#%%





#%%


#