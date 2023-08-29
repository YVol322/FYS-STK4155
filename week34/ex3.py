import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

n = 100
x = np.linspace(-3, 3, n).reshape(-1,1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)

y_train_arr = np.zeros(13)
y_test_arr = np.zeros(13)

poly_degree = np.zeros(13)
for i in range(2,15):
    m = i

    X=np.ones((n, m+1))
    for i in range(1,m+1,1):
        X[:, i] = (x**i).reshape(1,-1)

    y_train, y_test, X_train, X_test = train_test_split(y, X, test_size = 0.2)

    reg = LinearRegression(fit_intercept = 0).fit(X_train, y_train)

    beta = reg.coef_.reshape(-1,1)
    y_train_pred = X_train.dot(beta) 
    y_test_pred = X_test.dot(beta)

    y_train_arr[i-2] = mean_squared_error(y_train_pred, y_train)
    y_test_arr[i-2] = mean_squared_error(y_test_pred, y_test)
    poly_degree[i-2] = i

    print(i)
    print(mean_squared_error(y_train_pred, y_train))
    print(mean_squared_error(y_test_pred, y_test))

plt.figure(1)
plt.plot(poly_degree, y_test_arr, color = 'r')
plt.plot(poly_degree, y_train_arr, color = 'k')
plt.show()