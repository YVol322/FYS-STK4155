import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from Functions import sigmoid, Create_dir

np.random.seed(0) # Setting the seed results can be repoduced.

PNG_path, PDF_path = Create_dir('LogReg') # Creating directories to save figures to.

cancer = load_breast_cancer() # Loading dataset.

# Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# Sclaing the dataset.
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Setting intial weights.
weights = np.random.rand(X_train.shape[1])


epochs = 10000 # Number of iterations.
gammas = np.array((0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005)) # Array of learning rates.
lmbs = np.array((0.001, 0.01, 0.1, 1, 3, 10)) # Array of penalty parameters.


# Initilizing empty matrix.
map = np.zeros((gammas.shape[0], gammas.shape[0]))




j, l = 0, 0 # Initial indices.

# Loop over learning rates.
for gamma in gammas:

    # Loop over penalty parameters.
    for lmb in lmbs:

        # Loop over number of iterantions.
        for epoch in range(epochs):
            
            # Make a prediction.
            predictions = sigmoid(np.dot(X_train, weights))

            # Calculate error.
            error = y_train - predictions

            # Update weights using GD.
            gradient = np.dot(X_train.T, error) - 2 * lmb * weights
            weights += gamma * gradient

        # Predict on the test set.
        test_predictions = sigmoid(np.dot(X_test, weights))

        # Convert probabilities to binary predictions (0 or 1).
        test_predictions = np.where(test_predictions >= 0.5, 1, 0)

        # Calculate accuracy on the test set.
        accuracy = np.mean(test_predictions == y_test)
        
        print(f"Accuracy: {accuracy}")
        map[j,l] = accuracy # Add acuracies to the list.
        j += 1
    j = 0
    l += 1



# Generate a plot.

plt.figure(1)
sns.heatmap(map, cmap="YlGnBu", annot=True, square=True, xticklabels = gammas, yticklabels = lmbs, fmt='.4f')
plt.xlabel(r"$\gamma$")
plt.ylabel(r'$\lambda$')
#plt.savefig(PNG_path / 'LogReg_heatmap')
#plt.savefig(PDF_path / 'LogReg_heatmap.pdf')
plt.show()