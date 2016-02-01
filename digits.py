from sklearn import datasets, metrics, svm

from matplotlib import pyplot as plt

# helpful - http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

iris = datasets.load_iris()
digits = datasets.load_digits()

# images_and_labels = list(zip(digits.images, digits.target))
# for index, (image, label) in enumerate(images_and_labels[:9]):
#     plt.subplot(3, 3, index + 1)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Training: %i' % label)

# plt.show()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# classifier = svm.SVR(kernel='linear')
# classifier.fit([[1],[2],[3],[4],[4],[6],[7]], [2,3,4,5,6,7,8])
# print classifier.predict(22)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])

images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[-159:-150]):
    plt.subplot(3, 3, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()