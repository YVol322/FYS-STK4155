import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import  StandardScaler
from imageio import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from Functions import Terrain_Data, Create_directory, Create_X, Optimal_coefs_Ridge, Prediction


np.random.seed(2)

current_path = Path.cwd().resolve()
file_path = current_path.parent / 'Data' / 'SRTM_data_Norway_2.tif'
figures_path_PNG, figures_path_PDF = Create_directory('Terrain_Ridge')

terrain1 = imread(file_path)

#N = 20
N = 200

x,y,z = Terrain_Data(terrain1, N)

test_MSE_Ridge = []
train_MSE_Ridge = []
test_R2_Ridge = []
train_R2_Ridge = []

n_lambdas = 100
l = np.logspace(-3, 3, n_lambdas)
degree = 5

for lmbda in l:
    X = Create_X(x,y, degree)

    z_train, z_test, X_train, X_test = train_test_split(z, X, test_size = 0.2)

    scaler = StandardScaler(with_std=True)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    z_train = scaler.fit_transform(z_train)
    z_test = scaler.fit_transform(z_test)

    beta_Ridge = Optimal_coefs_Ridge(X_train, z_train, lmbda)

    z_train_Ridge = Prediction(X_train, beta_Ridge)
    z_test_Ridge = Prediction(X_test, beta_Ridge)

    test_MSE_Ridge.append(mean_squared_error(z_test, z_test_Ridge))
    train_MSE_Ridge.append(mean_squared_error(z_train, z_train_Ridge))
    test_R2_Ridge.append(r2_score(z_test, z_test_Ridge))
    train_R2_Ridge.append(r2_score(z_train, z_train_Ridge))


plt.figure(1)
plt.style.use('ggplot')
plt.subplot(2,1,1)
plt.plot(l, train_MSE_Ridge, label = 'Train MSE')
plt.plot(l, test_MSE_Ridge, label = 'Test MSE')
plt.xscale('log')
plt.ylabel('MSE')
plt.legend()
plt.subplot(2,1,2)
plt.plot(l, train_R2_Ridge, label = 'Train r2 score')
plt.plot(l, test_R2_Ridge, label = 'Test r2 score')
plt.xlabel('Penalty parameter')
plt.xscale('log')
plt.ylabel('R2 score')
plt.legend()
#plt.savefig(figures_path_PNG / 'Terrain_Ridge_points20')
#plt.savefig(figures_path_PDF / 'Terrain_Ridge_points20', format = 'pdf')
plt.savefig(figures_path_PNG / 'Terrain_Ridge_points200')
plt.savefig(figures_path_PDF / 'Terrain_Ridge_points200', format = 'pdf')
plt.show()