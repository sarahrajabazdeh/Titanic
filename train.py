import numpy as np 
import matplotlib.pyplot as plt
 
 #  logreg_inference for calculating probabilities
def logreg_inference(X,w,b):
    z = ( X @ w ) + b
    p = 1/(1 + np.exp(-z))
    return p

def cross_entropy(P,Y):
    return (-Y * np.log(P)- (1 - Y) * np.log(1-P)).mean()

def logreg_train(X, Y, lambda_, lr=1e-3, steps= 1000):
    m,n = X.shape
    w = np.zeros(n)
    b = 0
    acc = []
    losses=[]
    for step in range(steps):
        P = logreg_inference(X,w,b)
        loss = cross_entropy(P, Y)
        prediction = (P >0.5)
        accuracy = (prediction == Y).mean()
        losses.append(loss)
        acc.append(accuracy)
        grad_w = (( P - Y ) @ X ) / m + 2 * lambda_ * w
        grad_b = ( P - Y ) . mean ()
        w -= lr * grad_w
        b -= lr * grad_b
    return w , b, acc, losses

# load file load the file and return it's features.
def load_file(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]
    Y = data[:,-1]
    return X, Y 
X,Y = load_file("titanic-train.txt")

w,b,acc, looses = logreg_train(X,Y, 0 , 1e-3 ,100000 )
plt.plot(acc)
plt.figure()
plt.plot(looses)
plt.show()
X_ = np.array([[1, 0, 29, 1, 0, 100]])
P = logreg_inference(X_, w, b)
print(w)
Xrnd = X + np.random.randn(X.shape[0], X.shape[1])/12
plt.scatter(Xrnd[:, 0],Xrnd[:, 1],c=Y)
plt.xlabel("Age")
plt.ylabel("Fare") 
plt.show() 
np.savez("weights.npz",w,b)

#Droping Feature Fare
X_droped = np.delete(X, np.s_[5], axis=1) 
w_improved, b_improved, acc,losses = logreg_train(X_droped, Y, 0, 0.009, 1000000)
print("W =", w)
print("b = ", b)

P = logreg_inference(X_droped, w_improved, b_improved)
predictions = (P > 0.5)
accuracy = ( predictions == Y ).mean()
print("Accuracy improved =",accuracy * 100)
np.savez("improved.npz",w_improved,b_improved)