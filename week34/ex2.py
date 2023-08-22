import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)
x = np.random.rand(100,1)
y = 2.0+5*x*x+0.1*np.random.randn(100,1)

n = 2

y_pred = np.zeros((100,1))
X = np.zeros((100, n + 1))

for j in range(n+1):
    for i in range(np.shape(x)[0]):
        X[i][j] = pow(x[i], j)

beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

y_pred = X.dot(beta)

y_pred = np.sort(y_pred, axis = 0)
x_sort = np.sort(x, axis = 0)

plt.figure(1)
plt.plot(x, y, 'ro', label = 'x + noise')
plt.plot(x_sort,y_pred, color = 'k', label = '2nd order fit')
plt.legend()
plt.show()