import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Functions import FrankeFunction, create_X, Optimal_coefs_OLS, Prediction, Create_directory
from pathlib import Path
from sklearn.utils import resample

np.random.seed(9)

x = np.arange(0, 1, 0.01)
y = np.arange(0, 1, 0.01)

z = FrankeFunction(x, y)

maxdegree = 12
n_boostraps = 20

polydegree = np.zeros(maxdegree)

MSE = np.zeros(maxdegree)
BIAS = np.zeros(maxdegree)
VAR = np.zeros(maxdegree)

Create_directory('OLS_Bias_Variance')

current_path = Path.cwd().resolve()
figures_path_PNG = current_path.parent / "Figures" / "OLS_Bias_Variance" / "PNG"
figures_path_PDF = current_path.parent / "Figures" / "OLS_Bias_Variance" / "PDF"

for i in range(1, maxdegree+1, 1):

    X = create_X(x, y, i)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    
    z_test = z_test.reshape(z_test.shape[0],1)

    z_pred = np.empty((z_test.shape[0], n_boostraps))


    for j in range(n_boostraps):
        X_, z_ = resample(X_train, z_train)

        beta = Optimal_coefs_OLS(X_, z_)

        z_pred[:, j] = Prediction(X_test, beta)

    MSE[i-1] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    BIAS[i-1] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    VAR[i-1] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
    
    polydegree[i-1] = i

    print('Error:', MSE[i-1])
    print('Bias^2:', BIAS[i-1])
    print('Var:', VAR[i-1])
    print('{} >= {} + {} = {}'.format(MSE[i-1], BIAS[i-1], VAR[i-1], VAR[i-1]+BIAS[i-1]))

plt.figure()
plt.style.use('ggplot')
plt.plot(polydegree, MSE, label = 'MSE')
plt.plot(polydegree, BIAS, label = 'BIAS^2', linestyle = 'dashed')
plt.plot(polydegree, VAR, label = 'Varianse')
plt.xlabel("Polynomial fit degree")
plt.ylabel("Error value")
plt.legend()
#plt.savefig(figures_path_PNG / "OLS_Bias_Variance_points200")
#plt.savefig(figures_path_PDF / "OLS_Bias_Variance_points200", format = "pdf")
plt.show()