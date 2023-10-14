import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import  StandardScaler
from pathlib import Path
from imageio import imread
from sklearn.model_selection import train_test_split
from Functions import Terrain_Data, Create_X, Optimal_coefs_OLS, Optimal_coefs_Ridge, Prediction
from sklearn.metrics import mean_squared_error

np.random.seed(1)

current_path = Path.cwd().resolve()
file_path = current_path.parent / 'Data' / 'SRTM_data_Norway_2.tif'

terrain1 = imread(file_path)

degree = 5

N = 20

x,y,z = Terrain_Data(terrain1, N)

X = Create_X(x,y, degree)

z_train, z_test, X_train, X_test = train_test_split(z, X, test_size = 0.2)

scaler = StandardScaler(with_std=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
z_train = scaler.fit_transform(z_train)
z_test = scaler.fit_transform(z_test)


k = 10
kfold = KFold(n_splits = k)

MSE_OLS = np.zeros(k)
MSE_Risdge = np.zeros(k)
MSE_Lasso = np.zeros(k)

i = 0
lmb = 0.01
#lmb = 10
for train_inds, test_inds in kfold.split(z):
    X_train = X[train_inds]
    z_train = z[train_inds]

    X_test = X[test_inds]
    z_test = z[test_inds]

    beta_OLS = Optimal_coefs_OLS(X_train, z_train)
    z_test_OLS = Prediction(X_test, beta_OLS)

    beta_Ridge = Optimal_coefs_Ridge(X_train, z_train, lmb)
    z_test_Ridge = Prediction(X_test, beta_Ridge)

    clf = Lasso(alpha=lmb, fit_intercept= True)
    clf.fit(X_train, z_train)
    z_test_Lasso = clf.predict(X_test)

    MSE_OLS[i] = mean_squared_error(z_test, z_test_OLS)
    MSE_Risdge[i] = mean_squared_error(z_test, z_test_Ridge)
    MSE_Lasso[i] = mean_squared_error(z_test, z_test_Lasso)


    i += 1

estimated_mse_KFold_OLS = np.mean(MSE_OLS)
estimated_mse_KFold_Ridge = np.mean(MSE_Risdge)
estimated_mse_KFold_Lasso = np.mean(MSE_Lasso)

print('KFold MSE OLS: ', estimated_mse_KFold_OLS)
print('KFold MSE Ridge: ', estimated_mse_KFold_Ridge)
print('KFold MSE:Lasso ', estimated_mse_KFold_Lasso)