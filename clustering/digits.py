from sklearn import datasets

from matplotlib import pyplot as plt

digits = datasets.load_digits()

# just displaying examples...
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4*9:4]):
    plt.subplot(3, 3, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')