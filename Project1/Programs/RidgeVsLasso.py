import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from Functions import FrankeFunction, create_X, Optimal_coefs_Ridge, Optimal_coefs_OLS, Prediction
from sklearn.linear_model import Lasso
from pathlib import Path


np.random.seed(6)

#x = np.arange(0, 1, 0.05)
#y = np.arange(0, 1, 0.05)

x = np.arange(0, 1, 0.0001)
y = np.arange(0, 1, 0.0001)

z = FrankeFunction(x, y)

test_MSE_Lasso = []
train_MSE_Lasso = []
test_R2_Lasso = []
train_R2_Lasso = []

test_MSE_Ridge = []
train_MSE_Ridge = []
test_R2_Ridge = []
train_R2_Ridge = []

fit_degree = []

current_path = Path.cwd().resolve()

figures_path = current_path.parent / "Figures" / "RidgeVsLasso"
figures_path.mkdir(parents=True, exist_ok=True)

figures_path_PNG = current_path.parent / "Figures" / "RidgeVsLasso" / "PNG"
figures_path_PNG.mkdir(parents=True, exist_ok=True)

figures_path_PDF = current_path.parent / "Figures" / "RidgeVsLasso" / "PDF"
figures_path_PDF.mkdir(parents=True, exist_ok=True)

l = 10

for degree in range(1, 11, 1):
    X = create_X(x,y, degree)

    z_train, z_test, X_train, X_test = train_test_split(z, X, test_size = 0.2)

    clf = Lasso(alpha=l, fit_intercept= True)
    clf.fit(X_train, z_train)
    z_train_Lasso = clf.predict(X_train)
    z_test_Lasso = clf.predict(X_test)

    beta_Ridge = Optimal_coefs_Ridge(X_train, z_train, l)

    z_train_Ridge = Prediction(X_train, beta_Ridge)
    z_test_Ridge = Prediction(X_test, beta_Ridge)

    test_MSE_Lasso.append(mean_squared_error(z_test, z_test_Lasso))
    train_MSE_Lasso.append(mean_squared_error(z_train, z_train_Lasso))
    test_R2_Lasso.append(r2_score(z_test, z_test_Lasso))
    train_R2_Lasso.append(r2_score(z_train, z_train_Lasso))

    test_MSE_Ridge.append(mean_squared_error(z_test, z_test_Ridge))
    train_MSE_Ridge.append(mean_squared_error(z_train, z_train_Ridge))
    test_R2_Ridge.append(r2_score(z_test, z_test_Ridge))
    train_R2_Ridge.append(r2_score(z_train, z_train_Ridge))

    fit_degree.append(degree)



plt.figure(1)
plt.style.use('ggplot')
plt.subplot(4,1,1)
plt.plot(fit_degree, train_MSE_Lasso, label = 'Lasso')
plt.plot(fit_degree, train_MSE_Ridge, label = 'Ridge')
plt.ylabel('Train MSE')
plt.legend()
plt.subplot(4,1,2)
plt.plot(fit_degree, test_MSE_Lasso, label = 'Lasso')
plt.plot(fit_degree, test_MSE_Ridge, label = 'Ridge')
plt.ylabel('Test MSE')
plt.legend()
plt.subplot(4,1,3)
plt.plot(fit_degree, train_R2_Lasso, label = 'Lasso')
plt.plot(fit_degree, train_R2_Ridge, label = 'Ridge')
plt.ylabel('Train R2')
plt.legend()
plt.subplot(4,1,4)
plt.plot(fit_degree, test_R2_Lasso, label = 'Lasso')
plt.plot(fit_degree, test_R2_Ridge, label = 'Ridge')
plt.xlabel('Polynomial fit degree')
plt.ylabel('Test R2')
plt.legend()

for i in range(1, 4):
    plt.subplot(4, 1, i)
    plt.gca().set_xticks([])

plt.savefig(figures_path_PNG / "RidgeVsLasso_points1e3_lmb10")
plt.savefig(figures_path_PDF / "RidgeVsLasso_points1e3_lmb10", format = "pdf")
plt.show()