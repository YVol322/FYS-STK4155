import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pathlib import Path

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

def Create_directory(path):
    current_path = Path.cwd().resolve()

    figures_path = current_path.parent / "Figures" / path
    figures_path.mkdir(parents=True, exist_ok=True)

    figures_path_PNG = current_path.parent / "Figures" / path / "PNG"
    figures_path_PNG.mkdir(parents=True, exist_ok=True)

    figures_path_PDF = current_path.parent / "Figures" / path / "PDF"
    figures_path_PDF.mkdir(parents=True, exist_ok=True)

def Plot_Franke(x, y, z, name):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    current_path = Path.cwd().resolve()
    figures_path = current_path.parent / "Figures" / "Franke_plot"

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(figures_path / 'PDF' / name, format = 'pdf')
    plt.savefig(figures_path / 'PNG' / name)
    plt.show()

def Optimal_coefs_OLS(X_train, z_train):
    beta_OLS = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(z_train)

    return beta_OLS

def Prediction(X, beta):
    z_OLS = X @ beta

    return z_OLS

def Optimal_coefs_Ridge(X_train, z_train, l):
    I = np.eye(np.shape(X_train.T.dot(X_train))[0])
    beta_Ridge = np.linalg.pinv(X_train.T.dot(X_train) + l * I).dot(X_train.T).dot(z_train)

    return beta_Ridge