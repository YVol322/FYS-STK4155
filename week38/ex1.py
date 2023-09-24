import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


n = 40
n_boostraps = 100
maxdegree = 20


# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)

error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)

for degree in range(2, maxdegree, 1):
    X=np.ones((n, degree+1))

    for i in range(1,degree+1,1):
        X[:, i] = (x**i).reshape(1,-1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    y_pred = np.empty((y_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        X_, y_ = resample(X_train, y_train)
        beta = np.linalg.pinv(X_.T.dot(X_)).dot(X_.T).dot(y_)
        y_pred[:, i] = (X_test @ beta).reshape(1,-1)

    polydegree[degree] = degree
    error[degree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
    bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
    variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )
    print('Polynomial degree:', degree)
    print('Error:', error[degree])
    print('Bias^2:', bias[degree])
    print('Var:', variance[degree])
    print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
plt.show()