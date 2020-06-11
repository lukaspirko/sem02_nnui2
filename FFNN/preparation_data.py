from matplotlib import pyplot as plt
from skimage import feature

img = plt.imread('..\\Training\\negative\\p (10).png')
plt.imshow(img)
plt.show()

(H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), visualize=True)
plt.imshow(hogImage)
plt.show()

# print count parameters of image for next processing
print(H.shape)
