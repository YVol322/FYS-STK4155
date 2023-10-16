import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from Functions import Data, Create_X, Optimal_coefs_OLS, Optimal_coefs_Ridge, Prediction
from sklearn.metrics import mean_squared_error

np.random.seed(12)

degree = 5 # Fit degree.

N = 20 # Number of x and y points.
#N = 200
x,y,z = Data(N) # Generating the data.

X = Create_X(x,y, degree) # Filling design matrix.

z_train, z_test, X_train, X_test = train_test_split(z, X, test_size = 0.2) # Train-test split of the data.

k = 10 # number of k-folds.
kfold = KFold(n_splits = k) # using sklearn to perform k-fold cross validation to split the data.

MSE_OLS = np.zeros(k)
MSE_Risdge = np.zeros(k)
MSE_Lasso = np.zeros(k)

i = 0
lmb = 0.01 # Penalty parameter.
#lmb = 10
for train_inds, test_inds in kfold.split(z):
    # Reshufling the data
    X_train = X[train_inds] 
    z_train = z[train_inds] 

    X_test = X[test_inds]
    z_test = z[test_inds]

    beta_OLS = Optimal_coefs_OLS(X_train, z_train) # OLS optimal coefs uing matrix inv.
    z_test_OLS = Prediction(X_test, beta_OLS) # OLS test prediction.

    beta_Ridge = Optimal_coefs_Ridge(X_train, z_train, lmb) # Ridge optimal coefs uing matrix inv.
    z_test_Ridge = Prediction(X_test, beta_Ridge) # Ridge test prediction.

    # Using skrearn library to make a Lasso predictio.
    clf = Lasso(alpha=lmb, fit_intercept= True)
    clf.fit(X_train, z_train)
    z_test_Lasso = clf.predict(X_test) # Lasso test prediction.

    MSE_OLS[i] = mean_squared_error(z_test, z_test_OLS)
    MSE_Risdge[i] = mean_squared_error(z_test, z_test_Ridge)
    MSE_Lasso[i] = mean_squared_error(z_test, z_test_Lasso)


    i += 1

# Computing mean errors.
estimated_mse_KFold_OLS = np.mean(MSE_OLS)
estimated_mse_KFold_Ridge = np.mean(MSE_Risdge)
estimated_mse_KFold_Lasso = np.mean(MSE_Lasso)

# Printing errors to terminal.
print('KFold MSE OLS: ', estimated_mse_KFold_OLS)
print('KFold MSE Ridge: ', estimated_mse_KFold_Ridge)
print('KFold MSE:Lasso ', estimated_mse_KFold_Lasso)