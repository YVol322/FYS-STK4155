import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from Functions import FrankeFunction, create_X
from pathlib import Path

np.random.seed(12)

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)

z = FrankeFunction(x, y)

test_MSE = []
train_MSE = []
test_R2 = []
train_R2 = []
fit_degree = []

current_path = Path.cwd().resolve()
figures_path = current_path.parent / "Results"

for degree in range(1,6):
    X = create_X(x,y, degree)

    z_train, z_test, X_train, X_test = train_test_split(z, X, test_size = 0.2)

    beta_OLS = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(z_train)

    z_train_OLS = X_train @ beta_OLS
    z_test_OLS = X_test @ beta_OLS

    test_MSE.append(mean_squared_error(z_test, z_test_OLS))
    train_MSE.append(mean_squared_error(z_train, z_train_OLS))
    test_R2.append(r2_score(z_test, z_test_OLS))
    train_R2.append(r2_score(z_train, z_train_OLS))
    fit_degree.append(degree)

plt.figure(1)
plt.plot(fit_degree, train_MSE, label = 'Train MSE')
plt.plot(fit_degree, test_MSE, label = 'Test MSE')
plt.xlabel('Polynomial fit degree')
plt.ylabel('MSE')
plt.legend()
plt.savefig(figures_path / "OLS_MSE_5_noise")
plt.savefig(figures_path / "OLS_MSE_5_noise", format = "pdf")
plt.show()

plt.figure(2)
plt.plot(fit_degree, train_R2, label = 'Train r2 score')
plt.plot(fit_degree, test_R2, label = 'Test r2 score')
plt.savefig(figures_path / "OLS_R2_5_noise")
plt.savefig(figures_path / "OLS_R2_5_noise", format = "pdf")
plt.xlabel('Polynomial fit degree')
plt.ylabel('R2 score')
plt.legend()
plt.show()