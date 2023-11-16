import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from Functions import  Create_dir, FeedForward_class, initialize_W_and_b, StochasticBackPropagation_class

np.random.seed(0)

PNG_path, PDF_path = Create_dir('FFNN_class')

cancer = load_breast_cancer()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

n_inputs, n_features = X_train.shape


M = 10  # Batch size
n_epoch = 50
t0 = 5
t1 = 50
n = 10

n_hidden_nodes = 5 # Number of nods in hidden layers
n_hidden_layers = 3 # Number of hidden layers
n_layers = np.array((1, 2, 3, 4, 5, 6))
n_nodes = np.array((5, 10, 20, 30, 40, 50))
Ms = np.array((10, 20, 25, 50, 100))
n_epoch_array = np.array((25, 50, 100, 150, 200))
t0s = np.array((1, 20, 50, 70))
t1s = np.array((100, 1000, 10000, 100000))
n_output_nodes = 1 # Number of hidden layers


Accuracy_nodes = np.zeros((n_layers.shape[0], n_layers.shape[0]))
Accuracy_t = np.zeros((t1s.shape[0], t1s.shape[0]))
Accuracy_M = np.zeros((Ms.shape[0], Ms.shape[0]))

j, l = 0, 0
for n_hidden_nodes in n_nodes:
    for n_hidden_layers in n_layers:
        weights, biases = initialize_W_and_b(n_features, n_hidden_nodes, n_hidden_layers, n_output_nodes)

        weights_updated, biases_updated = StochasticBackPropagation_class(y_train, X_train, weights, biases, M, n_epoch, t0, t1)

        # Forward pass on test data
        _, a_list_test = FeedForward_class(X_test, weights_updated, biases_updated)
        predicted_output_test = a_list_test[-1]

        # Making predictions
        predictions = (predicted_output_test > 0.5).astype(int).flatten()

        # Calculating accuracy
        accuracy = np.mean(predictions == y_test)
        print(f"Test accuracy: {accuracy}")

        Accuracy_nodes[j,l] = accuracy
        j += 1
    j = 0
    l += 1

j, l = 0, 0
for t0 in t0s:
    for t1 in t1s:
        weights, biases = initialize_W_and_b(n_features, n_hidden_nodes, n_hidden_layers, n_output_nodes)

        weights_updated, biases_updated = StochasticBackPropagation_class(y_train, X_train, weights, biases, M, n_epoch, t0, t1)

        # Forward pass on test data
        _, a_list_test = FeedForward_class(X_test, weights_updated, biases_updated)
        predicted_output_test = a_list_test[-1]

        # Making predictions
        predictions = (predicted_output_test > 0.5).astype(int).flatten()

        # Calculating accuracy
        accuracy = np.mean(predictions == y_test)
        print(f"Test accuracy: {accuracy}")

        Accuracy_t[j,l] = accuracy
        j += 1
    j = 0
    l += 1

j, l = 0, 0
for M in Ms:
    for n_epoch in n_epoch_array:
        weights, biases = initialize_W_and_b(n_features, n_hidden_nodes, n_hidden_layers, n_output_nodes)

        weights_updated, biases_updated = StochasticBackPropagation_class(y_train, X_train, weights, biases, M, n_epoch, t0, t1)

        # Forward pass on test data
        _, a_list_test = FeedForward_class(X_test, weights_updated, biases_updated)
        predicted_output_test = a_list_test[-1]

        # Making predictions
        predictions = (predicted_output_test > 0.5).astype(int).flatten()

        # Calculating accuracy
        accuracy = np.mean(predictions == y_test)
        print(f"Test accuracy: {accuracy}")

        Accuracy_M[j,l] = accuracy
        j += 1
    j = 0
    l += 1


plt.figure(1)
sns.heatmap(Accuracy_nodes, cmap="YlGnBu", annot=True, square=True, xticklabels = n_layers, yticklabels = n_nodes, fmt = '.4f')
plt.xlabel(r"$n_{layers}$")
plt.ylabel(r'$n_{nodes}$')
plt.savefig(PNG_path / 'Class_nodes_layers')
plt.savefig(PDF_path / 'Class_nodes_layers.pdf')



plt.figure(2)
sns.heatmap(Accuracy_t, cmap="YlGnBu", annot=True, square=True, xticklabels = t0s, yticklabels = t1s)
plt.xlabel(r'$t_{0}$')
plt.ylabel(r'$t_{1}$')
plt.savefig(PNG_path / 'Class_t0_t1')
plt.savefig(PDF_path / 'Class_t0_t1.pdf')


plt.figure(3)
sns.heatmap(Accuracy_M, cmap="YlGnBu", annot=True, square=True, xticklabels = Ms, yticklabels = n_epoch_array)
plt.xlabel(r'$n_{epochs}$')
plt.ylabel(r'$M$')
plt.savefig(PNG_path / 'Class_minibach_epoch')
plt.savefig(PDF_path / 'Class_minibach_epoch.pdf')
plt.show()