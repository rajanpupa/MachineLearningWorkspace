# Simple python example that is using sklearn library to classify new points based on learning data
import numpy as np

# input array of points (x,y)
X = np.array([
    [-1, -1],
    [-2, -1],
    [-3, -2],
    [1, 1],
    [2, 1],
    [3, 2]]
)

# labels for the inputs
Y = np.array( [ 1, 1, 1, 2, 2, 2 ] )

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

# supervised learn
# pass the points array and the corresponding labels for the library to learn
classifier.fit(X, Y)

# predict the label of the new point based on the learning
print(classifier.predict([[-0.8, -1]]))
