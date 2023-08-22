import numpy as np

x = np.arange(0, 10, 1)
k = np.zeros((10,10))

for i in range(np.shape(k)[0]):
    for j in range(np.shape(k)[1]):
        k[i][j] = pow(x[i], j)

print(k)

