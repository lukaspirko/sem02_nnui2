from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import os

# tvorba trenovaci mnoziny
path_positive = "..\\Testing\\positive\\"
path_negative = "..\\Testing\\negative\\"

files_positive = os.listdir(path_positive)
files_negative = os.listdir(path_negative)

n_positive = len(files_positive)
n_negative = len(files_negative)
n_sum = n_positive + n_negative
inputs = np.zeros((n_sum, 51, 51, 3))  # vstup - počet vyzorků, rozměry obrázků 51x51, 3 barvy = 3 matice (rgb)
targets = np.zeros((n_sum, 2))  # výstup - počet vyzorků, počet tříd

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

model = load_model('model.h5')
outputs = model.predict(inputs)

print(outputs[5, :])
trues = 0
falses = 0

for ind in range(n_sum):
    if (outputs[ind, 0] > outputs[ind, 1] and ind < n_positive) or (
            outputs[ind, 0] < outputs[ind, 1] and ind >= n_negative):
        trues += 1
    else:
        falses += 1
print(trues)
print(falses)
print(trues / (trues + falses))
