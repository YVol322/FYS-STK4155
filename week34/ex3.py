import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(13)
n = 100
max = 17
x = np.linspace(-3, 3, n).reshape(-1,1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)

MSE_train_arr = np.zeros(max-2)
MSE_test_arr = np.zeros(max-2)

poly_degree = np.zeros(max-2)
for i in range(2,max):
    m = i

    X=np.ones((n, m+1))
    for i in range(1,m+1,1):
        X[:, i] = (x**i).reshape(1,-1)

    y_train, y_test, X_train, X_test = train_test_split(y, X, test_size = 0.2)

    reg = LinearRegression(fit_intercept = 0).fit(X_train, y_train)

    beta = reg.coef_.reshape(-1,1)
    y_train_pred = X_train.dot(beta) 
    y_test_pred = X_test.dot(beta)

    MSE_train_arr[i-2] = mean_squared_error(y_train_pred, y_train)
    MSE_test_arr[i-2] = mean_squared_error(y_test_pred, y_test)
    poly_degree[i-2] = i

    print(i)
    print(mean_squared_error(y_train_pred, y_train))
    print(mean_squared_error(y_test_pred, y_test))

plt.figure(1)
plt.title('$y(x)=e^{-x^2} + 1.5e^{-(x-2)^2} + N(0,0.1)$')
plt.plot(poly_degree, MSE_test_arr, color = 'r', label = 'test MSE')
plt.plot(poly_degree, MSE_train_arr, color = 'k', label = 'train MSE')
plt.legend()
plt.savefig('week34/train_test_MSE')
plt.show()

# From the data on the plot one can see that test data MSE is smaller than train data MSE,
# in avarage. Plos is similat to plot 2.11 in Hastie book. When polynomial fit degree is
# "too large", the data is overfitted and test MSE start to grow very fast. This means
# that one need to fit the data with lower order polynomials to get better result.
# The best polynomial order fit is that one that gives lowest test MSE. In this 
# particular case it is 11th order polynomial fit.