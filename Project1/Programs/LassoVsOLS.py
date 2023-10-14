import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from Functions import FrankeFunction, create_X, Optimal_coefs_OLS, Prediction, Create_directory
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

test_MSE_OLS = []
train_MSE_OLS = []
test_R2_OLS= []
train_R2_OLS = []

fit_degree = []

Create_directory("LassoVsOLS")

current_path = Path.cwd().resolve()
figures_path_PNG = current_path.parent / "Figures" / "LassoVsOLS" / "PNG"
figures_path_PDF = current_path.parent / "Figures" / "LassoVsOLS" / "PDF"

l = 10

for degree in range(1, 11, 1):
    X = create_X(x,y, degree)

    z_train, z_test, X_train, X_test = train_test_split(z, X, test_size = 0.2)

    clf = Lasso(alpha=l, fit_intercept= True)
    clf.fit(X_train, z_train)
    z_train_Lasso = clf.predict(X_train)
    z_test_Lasso = clf.predict(X_test)

    beta_OLS = Optimal_coefs_OLS(X_train, z_train)

    z_train_OLS = Prediction(X_train, beta_OLS)
    z_test_OLS = Prediction(X_test, beta_OLS)

    test_MSE_Lasso.append(mean_squared_error(z_test, z_test_Lasso))
    train_MSE_Lasso.append(mean_squared_error(z_train, z_train_Lasso))
    test_R2_Lasso.append(r2_score(z_test, z_test_Lasso))
    train_R2_Lasso.append(r2_score(z_train, z_train_Lasso))

    test_MSE_OLS.append(mean_squared_error(z_test, z_test_OLS))
    train_MSE_OLS.append(mean_squared_error(z_train, z_train_OLS))
    test_R2_OLS.append(r2_score(z_test, z_test_OLS))
    train_R2_OLS.append(r2_score(z_train, z_train_OLS))

    fit_degree.append(degree)



plt.figure(1)
plt.style.use('ggplot')
plt.subplot(4,1,1)
plt.plot(fit_degree, train_MSE_Lasso, label = 'Lasso')
plt.plot(fit_degree, train_MSE_OLS, label = 'OLS')
plt.ylabel('Train MSE')
plt.legend()
plt.subplot(4,1,2)
plt.plot(fit_degree, test_MSE_Lasso, label = 'Lasso')
plt.plot(fit_degree, test_MSE_OLS, label = 'OLS')
plt.ylabel('Test MSE')
plt.legend()
plt.subplot(4,1,3)
plt.plot(fit_degree, train_R2_Lasso, label = 'Lasso')
plt.plot(fit_degree, train_R2_OLS, label = 'OLS')
plt.ylabel('Train R2')
plt.legend()
plt.subplot(4,1,4)
plt.plot(fit_degree, test_R2_Lasso, label = 'Lasso')
plt.plot(fit_degree, test_R2_OLS, label = 'OLS')
plt.xlabel('Polynomial fit degree')
plt.ylabel('Test R2')
plt.legend()

for i in range(1, 4):
    plt.subplot(4, 1, i)
    plt.gca().set_xticks([])

#plt.savefig(figures_path_PNG / "LassoVsOLS_points1e3_lmb10")
#plt.savefig(figures_path_PDF / "LassoVsOLS_points1e3_lmb10", format = "pdf")
plt.show()