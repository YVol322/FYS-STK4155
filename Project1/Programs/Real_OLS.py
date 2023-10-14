from imageio import imread
import matplotlib.pyplot as plt
from pathlib import Path
from Functions import create_X, Optimal_coefs_OLS, Prediction, Create_directory
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import  StandardScaler

Create_directory("Real_OLS")

current_path = Path.cwd().resolve()
file_path = current_path.parent / 'Data' / 'SRTM_data_Norway_2.tif'
figures_path_PNG = current_path.parent / "Figures" / "Real_OLS" / "PNG"
figures_path_PDF = current_path.parent / "Figures" / "Real_OLS" / "PDF"

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

test_MSE = []
train_MSE = []
test_R2 = []
train_R2 = []
betas = []
fit_degree = []

for degree in range(1,6):
    X = create_X(x,y, degree)

    z_train, z_test, X_train, X_test = train_test_split(z, X, test_size = 0.2)

    scaler = StandardScaler(with_std=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    z_train = scaler.fit_transform(z_train)
    z_test = scaler.fit_transform(z_test)

    beta_OLS = Optimal_coefs_OLS(X_train, z_train)
    betas.append(beta_OLS)

    z_train_OLS = Prediction(X_train, beta_OLS)
    z_test_OLS = Prediction(X_test, beta_OLS)

    test_MSE.append(mean_squared_error(z_test, z_test_OLS))
    train_MSE.append(mean_squared_error(z_train, z_train_OLS))
    test_R2.append(r2_score(z_test, z_test_OLS))
    train_R2.append(r2_score(z_train, z_train_OLS))
    fit_degree.append(degree)

plt.figure(1)
plt.style.use('ggplot')
plt.subplot(2,1,1)
plt.plot(fit_degree, train_MSE, label = 'Train MSE')
plt.plot(fit_degree, test_MSE, label = 'Test MSE')
plt.ylabel('MSE')
plt.legend()
plt.subplot(2,1,2)
plt.plot(fit_degree, train_R2, label = 'Train r2 score')
plt.plot(fit_degree, test_R2, label = 'Test r2 score')
plt.xlabel('Polynomial fit degree')
plt.ylabel('R2 score')
plt.legend()
plt.savefig(figures_path_PNG / "Real_OLS")
plt.savefig(figures_path_PDF / "Real_OLS", format = "pdf")
plt.show()

plt.figure(2)
for x in betas:
    plt.scatter(x, [fit_degree[betas.index(x)]] * len(x))

plt.xlabel(r'$\beta$')
plt.ylabel('n')
plt.savefig(figures_path_PNG / "Real_OLS_betas")
plt.savefig(figures_path_PDF / "Real_OLS_betas", format = "pdf")
plt.show()