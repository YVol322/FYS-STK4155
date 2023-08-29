import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score # Importing metrics from sklearn liabrary.

n = 100 # Number of poins.
m = 2 # Polynomial fit degree.

x = np.random.rand(n,1) # x values according to normal distribution.
y = 2.0 + 5*x*x + 0.1*np.random.randn(n,1) # y = 2 + 5x^2 + 0.1 N(0,1).

y_pred = np.zeros((n,1)) # Empty array for the prediction of the model.
X = np.zeros((n, m + 1)) # Empty matrix for design matrix.

# Filling design matrix.
for j in range(m+1):
    for i in range(np.shape(x)[0]):
        X[i][j] = x[i]**j

# Making sklearn linear model (indluding intercept).
reg = LinearRegression(fit_intercept = 0).fit(X, y)

# Finding the prediction using analytical expession.
beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

print("Analytical fit coefs:", beta.T) # Analytical result for the fit coefs.
print("   Sklearn fit coefs:", reg.coef_) # Sklearn result for the fit coefs.

# One can see that fit coefficients are equal.

# Analytical/Sklearn prediction.
y_pred = X.dot(beta)

# MSE & R2 score of the prediction.
print('     MSE:', mean_squared_error(y, y_pred))
print('R2 score: ', r2_score(y, y_pred))

# Sorting massives for plot to display correctly.
y_pred = np.sort(y_pred, axis = 0)
x_sort = np.sort(x, axis = 0)

# Plot of the data & model prediction.
plt.figure(1)
plt.title('$y(x) = 2 + 5x^2 + 0.1 \cdot N(0,1)$')
plt.plot(x, y, 'ro', label = 'x + noise')
plt.plot(x_sort,y_pred, color = 'k', label = '2nd order fit')
plt.legend()
plt.savefig('week34/2_order_fit')
plt.show()

# One can see, that analytical prediction and sklearn prediction gives the same result. That's because the intercept
# of the sklearn linear model is set to 0 (false bool).