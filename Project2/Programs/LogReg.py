import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from Functions import sigmoid, Create_dir

np.random.seed(0)

PNG_path, PDF_path = Create_dir('LogReg')

# Load the breast cancer dataset
cancer = load_breast_cancer()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


weights = np.random.rand(X_train.shape[1])

# Hyperparameters
epochs = 10000
learning_rates = np.array((0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005))
penalties = np.array((0.001, 0.01, 0.1, 1, 3, 10))

map = np.zeros((learning_rates.shape[0], learning_rates.shape[0]))

j, l = 0, 0
for learning_rate in learning_rates:
    for alpha in penalties:
        # Gradient Descent for logistic regression
        for epoch in range(epochs):
            # Calculate hypothesis/prediction
            predictions = sigmoid(np.dot(X_train, weights))

            # Calculate error (difference between prediction and actual)
            error = y_train - predictions

            # Update weights using gradient descent
            gradient = np.dot(X_train.T, error) - 2 * alpha * weights  # Regularization term
            weights += learning_rate * gradient

        # Predict on the test set
        test_predictions = sigmoid(np.dot(X_test, weights))

        # Convert probabilities to binary predictions (0 or 1)
        test_predictions = np.where(test_predictions >= 0.5, 1, 0)

        # Calculate accuracy on the test set
        accuracy = np.mean(test_predictions == y_test)
        
        print(f"Accuracy: {accuracy}")
        map[j,l] = accuracy
        j += 1
    j = 0
    l += 1



plt.figure(1)
sns.heatmap(map, cmap="YlGnBu", annot=True, square=True, xticklabels = learning_rates, yticklabels = penalties, fmt='.4f')
plt.xlabel(r"$\gamma$")
plt.ylabel(r'$\lambda$')
plt.savefig(PNG_path / 'LogReg_heatmap')
plt.savefig(PDF_path / 'LogReg_heatmap.pdf')
plt.show()