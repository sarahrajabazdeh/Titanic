import numpy as np
import matplotlib.pyplot as plt
data = np.load("improved.npz")
W = data["arr_0"]
b = data["arr_1"]
# Load_file is uses for loading the data file.
def load_file(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y
# logreg_inference uses for calculating probabilities.
def logreg_inference(X,W,b):
    z = ( X @ W ) + b
    p = 1/(1 + np.exp(-z))
    return p

X, Y = load_file('titanic-test.txt')
P = logreg_inference(X, W, b)
predictions = (P > 0.5)
accuracy = ( predictions == Y ).mean()
print("Accuracy =",accuracy * 100)