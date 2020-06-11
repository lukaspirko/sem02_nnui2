from matplotlib import pyplot as plt
import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, MaxPooling2D, Dense, Flatten, Convolution2D

# tvorba trenovaci mnoziny
path_positive = "..\\Training\\positive\\"
path_negative = "..\\Training\\negative\\"

files_positive = os.listdir(path_positive)
files_negative = os.listdir(path_negative)

n_positive = len(files_positive)
n_negative = len(files_negative)
n_sum = n_positive + n_negative
inputs = np.zeros((n_sum, 51, 51, 3))
targets = np.zeros((n_sum, 2))

for ind in range(n_positive):
    str1 = path_positive + files_positive[ind]
    img = plt.imread(str1)
    inputs[ind, :, :, :] = img
    targets[ind, :] = np.array([1, 0])

for ind in range(n_negative):
    str1 = path_negative + files_negative[ind]
    img = plt.imread(str1)
    inputs[ind + n_positive, :, :, :] = img
    targets[ind + n_positive, :] = np.array([0, 1])

model = Sequential()
# původní řešení s komentářem
# počet filtrů (desítky(mocniny dvou)), velikost filtru(3*3), velikost vstupního obrázku (51*51*3(rgb)), posun vestikálně a horizontálně, řešení krajních hodnot obrázku (same)
model.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(51, 51, 3), strides=(1, 1), padding="same"))
# aktivační funkce - u konvoluční pozitivní lineární aktivační funkce
model.add(Activation("relu"))
# velikost okna pro sdružování, posun sdružování
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# sestaví výstupní vektor
model.add(Flatten())
model.add(Dense(10, activation='tanh'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='rmsprop', loss='mse')

model.summary()
epochs = 100
batch = 16
val = 0.15
hist = model.fit(x=inputs, y=targets, epochs=epochs, batch_size=batch, validation_split=val, verbose=2)
model.save('model.h5')
