from matplotlib import pyplot as plt
from skimage import feature
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import os

path_positive = "..\\Training\\positive\\"
path_negative = "..\\Training\\negative\\"

# prepare parameters for next processing
img = plt.imread(path_positive + 'p (10).png')
plt.imshow(img)
plt.show()

# doporucene pole 8x8 a orientace=9
(H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), visualize=True)
plt.imshow(hogImage)
plt.show()

# count parameters of image
n_features = int(''.join(map(str, H.shape)))

# tvorba trenovaci mnoziny

files_positive = os.listdir(path_positive)
files_negative = os.listdir(path_negative)

n_positives = len(files_positive)
# print(n_positives)
n_negatives = len(files_negative)
# print(n_negatives)

n_count = n_positives + n_negatives
inputs = np.zeros((n_count, n_features))
targets = np.zeros((n_count, 2))

for ind in range(len(files_positive)):
    str1 = path_positive + files_positive[ind]
    img = plt.imread(str1)
    inputs[ind, :] = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), visualize=False)
    targets[ind, :] = np.array([1, 0])

for ind in range(len(files_negative)):
    str1 = path_negative + files_negative[ind]
    img = plt.imread(str1)
    inputs[ind + len(files_positive), :] = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), visualize=False)
    targets[ind + len(files_positive), :] = np.array([0, 1])

# add layers
model = Sequential()
model.add(Dense(64, input_dim=n_features, activation='tanh'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='rmsprop', loss='mse')
model.summary()

# 0.9539
epochs = 1000
batch = 16
val = 0.15
hist = model.fit(x=inputs, y=targets, epochs=epochs, batch_size=batch, validation_split=val, verbose=2)
model.save('model.h5')
