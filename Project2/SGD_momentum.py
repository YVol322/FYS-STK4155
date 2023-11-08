import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

def learning_schedule(t):
    return t0/(t+t1)

np.random.seed(2)

n = 1000
degree = 3
x = np.arange(0, 1, 1/n).reshape(-1,1)
y = 3 - 5 * x + 4 * x ** 2

X = np.zeros((n, degree))

for i in range(degree):
    X[:, i] = (x**i).ravel()

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y

Ms = np.array((10, 20, 25, 50, 100))
t0, t1 = 650, 1000

eps = 1e-5
momentums = np.arange(0.1, 0.8, 0.15).round(3)
change = 0

map = np.zeros((Ms.shape[0], momentums.shape[0]))
j, l = 0, 0
for momentum in momentums:
    for M in Ms:
        beta = np.random.randn(degree,1)
        m = int(n/M)
        epoch = 0

        while(mean_squared_error(beta_linreg, beta)>eps):
            for i in range(m):
                k = M*np.random.randint(m)
                xi = X[k:k+M]
                yi = y[k:k+M]
                gradients = (2.0/M)* xi.T @ ((xi @ beta)-yi)
                eta = learning_schedule(epoch*m+i)
                new_change = eta*gradients + momentum * change
                beta -= new_change
                change = new_change

            epoch += 1
        map[j,l] = epoch
        j += 1
    j = 0
    l += 1
    
print(map)
plt.figure()
sns.heatmap(map, cmap="YlGnBu", annot=True, square=True, xticklabels = momentums, yticklabels = Ms,fmt= '.0f')
plt.xlabel("Minibatch size")
plt.ylabel("Momentum")
plt.show()

