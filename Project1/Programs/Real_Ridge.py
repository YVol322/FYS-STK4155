from imageio import imread
import matplotlib.pyplot as plt
from pathlib import Path
from Functions import FrankeFunction, create_X, Optimal_coefs_Ridge, Prediction, Create_directory
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import  StandardScaler

np.random.seed(6)

Create_directory("Real_Ridge")

current_path = Path.cwd().resolve()
file_path = current_path.parent / 'Data' / 'SRTM_data_Norway_2.tif'
figures_path_PNG = current_path.parent / "Figures" / "Real_Ridge" / "PNG"
figures_path_PDF = current_path.parent / "Figures" / "Real_Ridge" / "PDF"

# Load the terrain
terrain1 = imread(file_path)

N = 1000

maxdegree = 5

terrain = terrain1[:N,:N]

x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
x, y = np.meshgrid(x,y)

z = terrain
z = z.reshape(-1,1)

test_MSE_Ridge = []
train_MSE_Ridge = []
test_R2_Ridge = []
train_R2_Ridge = []

n_lambdas = 100
l = np.logspace(-3, 3, n_lambdas)
degree = 5

for lmbda in l:
    X = create_X(x,y, degree)

    z_train, z_test, X_train, X_test = train_test_split(z, X, test_size = 0.2)

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
plt.xlabel('$\lambda$')
plt.xscale('log')
plt.ylabel('R2 score')
plt.legend()
plt.savefig(figures_path_PNG / "Real_Ridge_lmbdas")
plt.savefig(figures_path_PDF / "Real_Ridge_lmbdas", format = "pdf")
plt.show()