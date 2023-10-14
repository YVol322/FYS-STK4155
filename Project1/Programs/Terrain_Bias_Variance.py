import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.preprocessing import  StandardScaler
from imageio import imread
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from Functions import Create_directory, Terrain_Data, Create_X, Optimal_coefs_OLS, Prediction

np.random.seed(1)

current_path = Path.cwd().resolve()
file_path = current_path.parent / 'Data' / 'SRTM_data_Norway_2.tif'
figures_path_PNG, figures_path_PDF = Create_directory('Terrain_Bias_Variance')

terrain1 = imread(file_path)

N = 20

x,y,z = Terrain_Data(terrain1, N)

maxdegree = 16
n_boostraps = 20

polydegree = np.zeros(maxdegree)

MSE = np.zeros(maxdegree)
BIAS = np.zeros(maxdegree)
VAR = np.zeros(maxdegree)

figures_path_PNG, figures_path_PDF = Create_directory('Bias_Variance')

for i in range(1, maxdegree+1, 1):

    X = Create_X(x, y, i)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    
    scaler = StandardScaler(with_std=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    z_train = scaler.fit_transform(z_train)
    z_test = scaler.fit_transform(z_test)

    z_pred = np.empty((z_test.shape[0], n_boostraps))


    for j in range(n_boostraps):
        X_, z_ = resample(X_train, z_train)

        beta = Optimal_coefs_OLS(X_, z_)

        z_pred[:, j] = Prediction(X_test, beta).ravel()

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
plt.plot(polydegree, BIAS, label = 'BIAS^2')
plt.plot(polydegree, VAR, label = 'Varianse')
plt.xlabel('Polynomial fit degree')
plt.ylabel('Error value')
plt.legend()
plt.savefig(figures_path_PNG / 'Terrain_Bias_Variance')
plt.savefig(figures_path_PDF / 'Terrain_Bias_Variance', format = 'pdf')
plt.show()