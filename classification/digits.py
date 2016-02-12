from sklearn import datasets, metrics, svm, grid_search, neighbors, ensemble

from matplotlib import pyplot as plt

digits = datasets.load_digits()

# just displaying examples...
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4*9:4]):
    plt.subplot(3, 3, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

plt.show()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

params = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['rbf']},
]

algorithms = [
    ('SVC', grid_search.GridSearchCV(svm.SVC(), params)),
    ('KNearestNeighbors', neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance')),
    ('RandomForest', ensemble.RandomForestClassifier(n_estimators=15))
]


# We learn the digits on the first half of the digits
algorithms[0][1].fit(data[:n_samples / 2], digits.target[:n_samples / 2])

# smaller dataset --> worse predictions
# algorithms[0].fit(data[:20], digits.target[:20])

# Predict the value of the digit on the second half:
expected = digits.target[n_samples / 2:]
predicted = algorithms[0][1].predict(data[n_samples / 2:])

images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
start = 134
for index, (image, prediction) in enumerate(images_and_predictions[-start - 9:-start]):
    plt.subplot(3, 3, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()

# Now, let's print out the percent in the second half that are correct...
def differences(a, b):
    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i != j for i, j in zip(a, b))

for name, alg in algorithms:
    alg.fit(data[:n_samples / 2], digits.target[:n_samples / 2])
    predicted = alg.predict(data[n_samples / 2:])
    num_differences = differences(expected, predicted)
    all_training = len(expected)

    # cast to float for decimals
    # subtract from one so it gives % correct
    print '{0} got an accuracy of {1}%'.format(name, 100*(1 - (float(num_differences) / float(all_training))))