import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from Functions import FrankeFunction, create_X, Create_directory
from sklearn.linear_model import Lasso
from pathlib import Path


np.random.seed(6)

x = np.arange(0, 1, 0.0001)
y = np.arange(0, 1, 0.0001)

#x = np.arange(0, 1, 0.05)
#y = np.arange(0, 1, 0.05)

z = FrankeFunction(x, y)

test_MSE_Lasso = []
train_MSE_Lasso = []
test_R2_Lasso = []
train_R2_Lasso = []

Create_directory("Lasso")

current_path = Path.cwd().resolve()
figures_path_PNG = current_path.parent / "Figures" / "Lasso" / "PNG"
figures_path_PDF = current_path.parent / "Figures" / "Lasso" / "PDF"

n_lambdas = 100
l = np.logspace(-3, 3, n_lambdas)
degree = 5

for lmbda in l:
    X = create_X(x,y, degree)

    z_train, z_test, X_train, X_test = train_test_split(z, X, test_size = 0.2)

    clf = Lasso(alpha=lmbda, fit_intercept= True)
    clf.fit(X_train, z_train)
    z_train_Lasso = clf.predict(X_train)
    z_test_Lasso = clf.predict(X_test)

    test_MSE_Lasso.append(mean_squared_error(z_test, z_test_Lasso))
    train_MSE_Lasso.append(mean_squared_error(z_train, z_train_Lasso))
    test_R2_Lasso.append(r2_score(z_test, z_test_Lasso))
    train_R2_Lasso.append(r2_score(z_train, z_train_Lasso))


plt.figure(1)
plt.style.use('ggplot')
plt.subplot(2,1,1)
plt.plot(l, train_MSE_Lasso, label = 'Train MSE')
plt.plot(l, test_MSE_Lasso, label = 'Test MSE')
plt.xscale('log')
plt.ylabel('MSE')
plt.legend()
plt.subplot(2,1,2)
plt.plot(l, train_R2_Lasso, label = 'Train r2 score')
plt.plot(l, test_R2_Lasso, label = 'Test r2 score')
plt.xlabel('$\lambda$')
plt.xscale('log')
plt.ylabel('R2 score')
plt.legend()
plt.savefig(figures_path_PNG / "Lasso_points1e3_lmbdas")
plt.savefig(figures_path_PDF / "Lasso_points1e3_lmbdas", format = "pdf")
plt.show()