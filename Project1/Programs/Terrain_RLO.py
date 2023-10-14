import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import  StandardScaler
from imageio import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from Functions import Create_directory, Terrain_Data, Create_X, Optimal_coefs_OLS, Optimal_coefs_Ridge, Prediction

np.random.seed(1)

current_path = Path.cwd().resolve()
file_path = current_path.parent / 'Data' / 'SRTM_data_Norway_2.tif'
figures_path_PNG, figures_path_PDF = Create_directory('Terrain_RLO')

terrain1 = imread(file_path)

N = 40
#N=1000

x,y,z = Terrain_Data(terrain1, N)

test_MSE_Ridge = []
train_MSE_Ridge = []
test_R2_Ridge = []
train_R2_Ridge = []

test_MSE_OLS = []
train_MSE_OLS = []
test_R2_OLS= []
train_R2_OLS = []

test_MSE_Lasso = []
train_MSE_Lasso = []
test_R2_Lasso = []
train_R2_Lasso = []

fit_degree = []

#l = 0.01
l = 1
maxdegree = 5

for degree in range(1, maxdegree + 1):
    X = Create_X(x,y, degree)

    z_train, z_test, X_train, X_test = train_test_split(z, X, test_size = 0.2)

    scaler = StandardScaler(with_std=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    z_train = scaler.fit_transform(z_train)
    z_test = scaler.fit_transform(z_test)

    beta_OLS = Optimal_coefs_OLS(X_train, z_train)

    z_train_OLS = Prediction(X_train, beta_OLS)
    z_test_OLS = Prediction(X_test, beta_OLS)

    beta_Ridge = Optimal_coefs_Ridge(X_train, z_train, l)

    z_train_Ridge = Prediction(X_train, beta_Ridge)
    z_test_Ridge = Prediction(X_test, beta_Ridge)

    clf = Lasso(alpha = l, fit_intercept = True)
    clf.fit(X_train, z_train)
    z_train_Lasso = clf.predict(X_train)
    z_test_Lasso = clf.predict(X_test)

    test_MSE_OLS.append(mean_squared_error(z_test, z_test_OLS))
    train_MSE_OLS.append(mean_squared_error(z_train, z_train_OLS))
    test_R2_OLS.append(r2_score(z_test, z_test_OLS))
    train_R2_OLS.append(r2_score(z_train, z_train_OLS))

    test_MSE_Ridge.append(mean_squared_error(z_test, z_test_Ridge))
    train_MSE_Ridge.append(mean_squared_error(z_train, z_train_Ridge))
    test_R2_Ridge.append(r2_score(z_test, z_test_Ridge))
    train_R2_Ridge.append(r2_score(z_train, z_train_Ridge))

    test_MSE_Lasso.append(mean_squared_error(z_test, z_test_Lasso))
    train_MSE_Lasso.append(mean_squared_error(z_train, z_train_Lasso))
    test_R2_Lasso.append(r2_score(z_test, z_test_Lasso))
    train_R2_Lasso.append(r2_score(z_train, z_train_Lasso))

    fit_degree.append(degree)



plt.figure(1)
plt.style.use('ggplot')
plt.subplot(4,1,1)
plt.plot(fit_degree, train_MSE_Ridge, label = 'Ridge')
plt.plot(fit_degree, train_MSE_Lasso, label = 'Lasso')
plt.plot(fit_degree, train_MSE_OLS, label = 'OLS')
plt.ylabel('Train MSE')
plt.legend()
plt.subplot(4,1,2)
plt.plot(fit_degree, test_MSE_Ridge, label = 'Ridge')
plt.plot(fit_degree, test_MSE_Lasso, label = 'Lasso')
plt.plot(fit_degree, test_MSE_OLS, label = 'OLS')
plt.ylabel('Test MSE')
plt.legend()
plt.subplot(4,1,3)
plt.plot(fit_degree, train_R2_Ridge, label = 'Ridge')
plt.plot(fit_degree, train_R2_Lasso, label = 'Lasso')
plt.plot(fit_degree, train_R2_OLS, label = 'OLS')
plt.ylabel('Train R2')
plt.legend()
plt.subplot(4,1,4)
plt.plot(fit_degree, test_R2_Ridge, label = 'Ridge')
plt.plot(fit_degree, test_R2_Lasso, label = 'Lasso')
plt.plot(fit_degree, test_R2_OLS, label = 'OLS')
plt.xlabel('Polynomial fit degree')
plt.ylabel('Test R2')
plt.legend()

for i in range(1, 4):
    plt.subplot(4, 1, i)
    plt.gca().set_xticks([])

#plt.savefig(figures_path_PNG / 'Terrain_RLO_points20_lmb1e-2')
#plt.savefig(figures_path_PDF / 'Terrain_RLO_points20_lmb1e-2', format = "pdf")
plt.savefig(figures_path_PNG / 'RLO_points20_lmb1e1')
plt.savefig(figures_path_PDF / 'RLO_points20_lmb1e1', format = "pdf")
plt.show()