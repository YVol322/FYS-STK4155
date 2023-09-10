import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
np.random.seed(4)
# Number of data points.
n = 100

# Data.
x = np.linspace(-3, 3, n).reshape(-1,1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)
y_nn = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)

# Polymonial fit degree.
ms = np.array([10,15])

# Loop for program to compute MSEs for 10 and 15 polynomial degree fit.
for m in ms:

    # Design Matrix.
    X=np.ones((n, m+1))
    for i in range(1,m+1,1):
        X[:, i] = (x**i).reshape(1,-1)

    # Scale the data.
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    
    # Train-test split.
    y_train, y_test, x_train, x_test, X_train, X_test, X_scaled_train, X_scaled_test = train_test_split(y, x, X, X_scaled, test_size = 0.2)
    
    # OLS.
    beta_OLS = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
    y_train_OLS = X_train @ beta_OLS
    y_test_OLS = X_test @ beta_OLS

    # Scaled OLS
    beta_scaled_OLS = np.linalg.pinv(X_scaled_train.T.dot(X_scaled_train)).dot(X_scaled_train.T).dot(y_train)
    y_train_scaled_OLS = X_scaled_train @ beta_scaled_OLS
    y_test_scaled_OLS = X_scaled_test @ beta_scaled_OLS
    
    # Printing OLS MSEs
    print('OLS train MSE: ', mean_squared_error(y_train, y_train_OLS))
    print('OLS test MSE: ', mean_squared_error(y_test, y_test_OLS))
    print('OLS scaled test MSE: ', mean_squared_error(y_test, y_test_scaled_OLS))
    
    # Penalty parameters.
    lmbdas = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    
    # Identity matrix.
    I = np.eye(np.shape(X_test)[1], np.shape(X_test)[1])
    
    for lmb in lmbdas:
        beta_Ridge = np.linalg.pinv(X_train.T.dot(X_train) + I * lmb).dot(X_train.T).dot(y_train)
    
        y_train_Ridge = X_train @ beta_Ridge
        y_test_Ridge = X_test @ beta_Ridge
        
        print('Ridge train MSE: ', mean_squared_error(y_train, y_train_Ridge), 'lmb =', lmb)
        print('Ridge test MSE: ', mean_squared_error(y_test, y_test_Ridge), 'lmb =', lmb)

# One can see that The best MSE is using OLS method. Bigger penalty parameter => bigger MSE.
# Also if one will uncooment lines 28-30, one could see that scaled data gives bigger MSE.
# That's because the data does not need the scaling, it is created that way, that it does not need scaling.

beta_0 = np.mean(y) + np.mean(np.sum(X_scaled @ beta_scaled_OLS))

plt.figure(1)
plt.title('ploynomial degree fit: 10')
plt.plot(x, y_nn, label = 'y(x) without noise')
plt.plot(x_test, y_test_OLS, 'x', label = 'test prdiction OLS')
plt.plot(x_test, y_test_scaled_OLS + beta_0 , 'x', label = 'test prdiction OLS, scaled')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.show()