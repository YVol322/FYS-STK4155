import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from Functions import Data, Create_X, Create_directory


np.random.seed(1)

N = 20 # Number of x and y points.
#N = 200
x,y,z = Data(N) # Generating the data.

test_MSE_Lasso = []
train_MSE_Lasso = []
test_R2_Lasso = []
train_R2_Lasso = []

figures_path_PNG, figures_path_PDF = Create_directory('Lasso') # Creating directory, to save figures to.

n_lambdas = 100 # Number of penalty parameter lambda.
l = np.logspace(-3, 3, n_lambdas) # Log array of penalty parameters.
degree = 5 # Fit degree.

for lmbda in l:
    X = Create_X(x,y, degree) # Filling design matrix.

    z_train, z_test, X_train, X_test = train_test_split(z, X, test_size = 0.2) # Train-test split of the data.

    # Using skrearn library to make a Lasso predictio.
    clf = Lasso(alpha = lmbda, fit_intercept = True)
    clf.fit(X_train, z_train)
    z_train_Lasso = clf.predict(X_train) # Lasso train prediction.
    z_test_Lasso = clf.predict(X_test) # Lasso test prediction.

    # Saving MSEs and R2 scores to lists.
    test_MSE_Lasso.append(mean_squared_error(z_test, z_test_Lasso))
    train_MSE_Lasso.append(mean_squared_error(z_train, z_train_Lasso))
    test_R2_Lasso.append(r2_score(z_test, z_test_Lasso))
    train_R2_Lasso.append(r2_score(z_train, z_train_Lasso))

# Plot of MSEs and R2 scores vs lambdas.
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
plt.xlabel('Penalty parameter')
plt.xscale('log')
plt.ylabel('R2 score')
plt.legend()
#plt.savefig(figures_path_PNG / 'Lasso_points20')
#plt.savefig(figures_path_PDF / 'Lasso_points20', format = 'pdf')
#plt.savefig(figures_path_PNG / 'Lasso_points200')
#plt.savefig(figures_path_PDF / 'Lasso_points200', format = 'pdf')
plt.show()