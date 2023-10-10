import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from Functions import FrankeFunction, create_X, Optimal_coefs, Prediction
from pathlib import Path


np.random.seed(12)
#np.random.seed(10)

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)

z = FrankeFunction(x, y)
#z = FrankeFunction(x, y) + np.random.normal(0, 0.1, np.shape(x))

test_MSE = []
train_MSE = []
test_R2 = []
train_R2 = []
fit_degree = []

current_path = Path.cwd().resolve()

figures_path = current_path.parent / "Results"
figures_path.mkdir(parents=True, exist_ok=True)

figures_path_PNG = current_path.parent / "Results" / "PNG"
figures_path_PNG.mkdir(parents=True, exist_ok=True)

figures_path_PDF = current_path.parent / "Results" / "PDF"
figures_path_PDF.mkdir(parents=True, exist_ok=True)

for degree in range(1,11):
    X = create_X(x,y, degree)

    z_train, z_test, X_train, X_test = train_test_split(z, X, test_size = 0.2)

    beta_OLS = Optimal_coefs(X_train, z_train)

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
#plt.savefig(figures_path_PNG / "OLS_5")
#plt.savefig(figures_path_PDF / "OLS_5", format = "pdf")
plt.show()