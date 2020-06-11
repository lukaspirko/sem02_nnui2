from matplotlib import pyplot as plt
from skimage import feature
from tensorflow.keras.models import load_model
import numpy as np
import os

path_positive = "..\\Testing\\positive\\"
path_negative = "..\\Testing\\negative\\"

# prepare parameters for next processing
img = plt.imread('..\\Testing\\positive\\p (9).png')
plt.imshow(img)
plt.show()

# doporucene pole 8x8 a orientace=9
(H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), visualize=True)
plt.imshow(hogImage)
plt.show()

# count parameters of image
n_features = int(''.join(map(str, H.shape)))

files_positive = os.listdir(path_positive)
files_negative = os.listdir(path_negative)

n_positives = len(files_positive)
# print(n_positives)
n_negatives = len(files_negative)
# print(n_negatives)

n_count = n_positives + n_negatives
inputs = np.zeros((n_count, n_features))
targets = np.zeros((n_count, 2))

for ind in range(n_positives):
    str1 = path_positive + files_positive[ind]
    img = plt.imread(str1)
    inputs[ind, :] = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), visualize=False)
    targets[ind, :] = np.array([1, 0])

for ind in range(n_negatives):
    str1 = path_negative + files_negative[ind]
    img = plt.imread(str1)
    inputs[ind + n_positives, :] = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), visualize=False)
    targets[ind + n_positives, :] = np.array([0, 1])

model = load_model('model.h5')
outputs = model.predict(inputs)

print(outputs[10, :])
trues = 0
falses = 0

for ind in range(n_count):
    if (outputs[ind, 0] > outputs[ind, 1] and ind < n_positives) or (
            outputs[ind, 0] < outputs[ind, 1] and ind >= n_negatives):
        trues += 1
    else:
        falses += 1
print(trues)
print(falses)
print(trues / (trues + falses))
