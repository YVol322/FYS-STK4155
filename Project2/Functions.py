import autograd.numpy as np
from sklearn.metrics import mean_squared_error
from autograd import grad



def Data():
    n = 1000
    degree = 3
    x = np.arange(0, 1, 1/n).reshape(-1,1)
    y = 3 - 5 * x + 4 * x ** 2

    X = np.zeros((n, degree))

    for i in range(degree):
        X[:, i] = (x**i).ravel()

    return x, y, X, n, degree

def learning_schedule(t, t0, t1):
    return t0/(t+t1)


def CostOLS(beta, n, y, X):
    return (1.0/n) * np.sum((y - X @ beta)**2)

def CostRidge(beta, lmb, n, y, X):
    return (1.0/n) * np.sum((y - X @ beta)**2) + lmb * np.sum(beta**2)


def GD(X, y, degree, n, eps, eta):

    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)

    training_gradient = grad(CostOLS, 0)
    #training_gradient = grad(CostRidge, 0)

    i=0
    while(mean_squared_error(beta_linreg, beta)>eps):
        #gradient = (2.0/n)*X.T @ (X @ beta-y)
        gradient = training_gradient(beta, n, y, X)


        beta -= eta*gradient

        i+=1
    
    return beta, i


def Ada(X, y, degree, n, eps, eta, delta):

    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)
    G = np.diag(np.zeros(degree))

    training_gradient = grad(CostOLS, 0)

    i=0
    while(mean_squared_error(beta_linreg, beta)>eps):
        #gradient = (2.0/n)*X.T @ (X @ beta-y)
        gradient = training_gradient(beta, n, y, X)

        G += gradient*gradient
        G_diag = G.diagonal()
        sqrt_G = np.sqrt(G_diag)

        gamma = eta/(delta + sqrt_G).reshape(-1,1)

        beta -= gamma*gradient

        i+=1


    return beta, i

def RMS(X, y, degree, n, eps, eta, delta, rho):
    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)
    G = np.diag(np.zeros(degree))

    training_gradient = grad(CostOLS, 0)

    i=0
    while(mean_squared_error(beta_linreg, beta)>eps):
        #gradient = (2.0/n)*X.T @ (X @ beta-y)
        gradient = training_gradient(beta, n, y, X)

        G = (rho*G+(1-rho)*gradient*gradient)
        G_diag = G.diagonal()
        sqrt_G = np.sqrt(G_diag)
        gamma = eta/(delta + sqrt_G).reshape(-1,1)

        beta -= gamma*gradient

        i+=1


    return beta, i

def ADAM(X, y, degree, n, eps, eta, delta, beta1, beta2):
    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)

    first_moment = 0.0
    second_moment = 0.0

    training_gradient = grad(CostOLS, 0)
    
    i=0
    while(mean_squared_error(beta_linreg, beta)>eps):
        #gradient = (2.0/n)*X.T @ (X @ beta-y)
        gradient = training_gradient(beta, n, y, X)

        first_moment = beta1*first_moment + (1-beta1)*gradient
        second_moment = beta2*second_moment+(1-beta2)*gradient*gradient
        first_term = first_moment/(1.0-beta1**(i+1))
        second_term = second_moment/(1.0-beta2**(i+1))

        update = eta*first_term/(np.sqrt(second_term)+delta)
        beta -= update
        
        i+=1


    return beta, i

def GD_momentum(X, y, degree, n, eps, eta, moment):

    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)

    change = 0

    training_gradient = grad(CostOLS, 0)

    i=0
    while(mean_squared_error(beta_linreg, beta)>eps):
        #gradient = (2.0/n)*X.T @ (X @ beta-y)
        gradient = training_gradient(beta, n, y, X)

        new_change = eta*gradient + moment * change

        beta -= new_change
        change = new_change

        i+=1
    
    return beta, i

def Ada_momentum(X, y, degree, n, eps, eta, delta, moment):

    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)
    G = np.diag(np.zeros(degree))

    change = 0

    training_gradient = grad(CostOLS, 0)

    i=0
    while(mean_squared_error(beta_linreg, beta)>eps):
        #gradient = (2.0/n)*X.T @ (X @ beta-y)
        gradient = training_gradient(beta, n, y, X)

        G += gradient*gradient
        G_diag = G.diagonal()
        sqrt_G = np.sqrt(G_diag)

        gamma = eta/(delta + sqrt_G).reshape(-1,1)

        new_change = gamma*gradient + moment * change

        beta -= new_change
        change = new_change

        i+=1

    return beta, i

def RMS_momentum(X, y, degree, n, eps, eta, delta, rho, moment):
    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)
    G = np.diag(np.zeros(degree))

    change = 0

    training_gradient = grad(CostOLS, 0)

    i=0
    while(mean_squared_error(beta_linreg, beta)>eps):
        #gradient = (2.0/n)*X.T @ (X @ beta-y)
        gradient = training_gradient(beta, n, y, X)

        G = (rho*G+(1-rho)*gradient*gradient)
        G_diag = G.diagonal()
        sqrt_G = np.sqrt(G_diag)
        gamma = eta/(delta + sqrt_G).reshape(-1,1)

        new_change = gamma*gradient + moment * change

        beta -= new_change
        print(beta[0])
        change = new_change

        i+=1

    return beta, i


def ADAM_momentum(X, y, degree, n, eps, eta, delta, beta1, beta2, moment):
    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)

    first_moment = 0.0
    second_moment = 0.0

    change = 0

    training_gradient = grad(CostOLS, 0)
    
    i=0
    while(mean_squared_error(beta_linreg, beta)>eps):
        #gradient = (2.0/n)*X.T @ (X @ beta-y)
        gradient = training_gradient(beta, n, y, X)

        first_moment = beta1*first_moment + (1-beta1)*gradient
        second_moment = beta2*second_moment+(1-beta2)*gradient*gradient
        first_term = first_moment/(1.0-beta1**(i+1))
        second_term = second_moment/(1.0-beta2**(i+1))

        update = eta*first_term/(np.sqrt(second_term)+delta)

        new_change = update + moment * change

        beta -= new_change
        print(beta[0])
        change = new_change
        
        i+=1

    return beta, i

def SGD(X, y, degree, n, eps, t0, t1, M):

    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)

    m = int(n/M)

    training_gradient = grad(CostOLS, 0)

    epoch = 0
    while(mean_squared_error(beta_linreg, beta)>eps):
        for i in range(m):
            k = M*np.random.randint(m)
            xi = X[k:k+M]
            yi = y[k:k+M]
            #gradients = (2.0/M)* xi.T @ ((xi @ beta)-yi)
            gradients = training_gradient(beta, n, y, X)

            eta = learning_schedule(epoch*m+i, t0, t1)
            beta -= eta*gradients
        epoch += 1

    
    return beta, epoch


def SGD_Ada(X, y, degree, n, eps, delta, t0, t1, M):

    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)
    G = np.diag(np.zeros(degree))

    m = int(n/M)

    training_gradient = grad(CostOLS, 0)

    epoch = 0
    while(mean_squared_error(beta_linreg, beta)>eps):
        for i in range(m):
            k = M*np.random.randint(m)
            xi = X[k:k+M]
            yi = y[k:k+M]
            #gradients = (2.0/M)* xi.T @ ((xi @ beta)-yi)
            gradients = training_gradient(beta, n, y, X)

            G += gradients*gradients
            G_diag = G.diagonal()
            sqrt_G = np.sqrt(G_diag)

            eta = learning_schedule(epoch*m+i, t0, t1)
            gamma = eta/(delta + sqrt_G).reshape(-1,1)

            beta -= gamma*gradients
            print(beta[0])

        epoch+=1


    return beta, epoch


def SGD_RMS(X, y, degree, n, eps, delta, rho, t0, t1, M):
    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)
    G = np.diag(np.zeros(degree))

    m = int(n/M)

    training_gradient = grad(CostOLS, 0)

    epoch = 0
    while(mean_squared_error(beta_linreg, beta)>eps):
        for i in range(m):
            k = M*np.random.randint(m)
            xi = X[k:k+M]
            yi = y[k:k+M]
            #gradients = (2.0/M)* xi.T @ ((xi @ beta)-yi)
            gradients = training_gradient(beta, n, y, X)

            G = (rho*G+(1-rho)*gradients*gradients)
            G_diag = G.diagonal()
            sqrt_G = np.sqrt(G_diag)
            
            eta = learning_schedule(epoch*m+i, t0, t1)
            gamma = eta/(delta + sqrt_G).reshape(-1,1)

            beta -= gamma*gradients

        epoch += 1

    return beta, epoch


def SGD_ADAM(X, y, degree, n, eps, delta, beta1, beta2, t0, t1, M):
    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)

    first_moment = 0.0
    second_moment = 0.0

    m = int(n/M)

    training_gradient = grad(CostOLS, 0)
    
    epoch = 0
    while(mean_squared_error(beta_linreg, beta)>eps):
        for i in range(m):
            k = M*np.random.randint(m)
            xi = X[k:k+M]
            yi = y[k:k+M]
            #gradients = (2.0/M)* xi.T @ ((xi @ beta)-yi)
            gradients = training_gradient(beta, n, y, X)

            first_moment = beta1*first_moment + (1-beta1)*gradients
            second_moment = beta2*second_moment+(1-beta2)*gradients*gradients
            first_term = first_moment/(1.0-beta1**(i+1))
            second_term = second_moment/(1.0-beta2**(i+1))

            eta = learning_schedule(epoch*m+i, t0, t1)
            update = eta*first_term/(np.sqrt(second_term)+delta)

            beta -= update
            print(beta[0])
        
        epoch += 1


    return beta, epoch


def SGDM(X, y, degree, n, eps, moment, t0, t1, M):

    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)

    change = 0

    m = int(n/M)

    training_gradient = grad(CostOLS, 0)

    epoch = 0
    while(mean_squared_error(beta_linreg, beta)>eps):
        for i in range(m):
            k = M*np.random.randint(m)
            xi = X[k:k+M]
            yi = y[k:k+M]
            #gradients = (2.0/M)* xi.T @ ((xi @ beta)-yi)
            gradients = training_gradient(beta, n, y, X)

            eta = learning_schedule(epoch*m+i, t0, t1)
            new_change = eta*gradients + moment * change

            beta -= new_change
            change = new_change

        epoch += 1

    
    return beta, epoch


def SGDM_Ada(X, y, degree, n, eps, delta, moment, t0, t1, M):

    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)
    G = np.diag(np.zeros(degree))

    change = 0

    m = int(n/M)

    training_gradient = grad(CostOLS, 0)

    epoch = 0
    while(mean_squared_error(beta_linreg, beta)>eps):
        for i in range(m):
            k = M*np.random.randint(m)
            xi = X[k:k+M]
            yi = y[k:k+M]
            #gradients = (2.0/M)* xi.T @ ((xi @ beta)-yi)
            gradients = training_gradient(beta, n, y, X)

            G += gradients*gradients
            G_diag = G.diagonal()
            sqrt_G = np.sqrt(G_diag)

            eta = learning_schedule(epoch*m+i, t0, t1)
            gamma = eta/(delta + sqrt_G).reshape(-1,1)
            new_change = gamma*gradients + moment * change

            beta -= new_change
            change = new_change
            print(beta[0])

        epoch+=1


    return beta, epoch


def SGDM_RMS(X, y, degree, n, eps, delta, rho, moment, t0, t1, M):
    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)
    G = np.diag(np.zeros(degree))

    change = 0

    m = int(n/M)

    training_gradient = grad(CostOLS, 0)

    epoch = 0
    while(mean_squared_error(beta_Ridge, beta)>eps):
        for i in range(m):
            k = M*np.random.randint(m)
            xi = X[k:k+M]
            yi = y[k:k+M]
            #gradients = (2.0/M)* xi.T @ ((xi @ beta)-yi)
            gradients = training_gradient(beta, n, y, X)

            G = (rho*G+(1-rho)*gradients*gradients)
            G_diag = G.diagonal()
            sqrt_G = np.sqrt(G_diag)
            
            eta = learning_schedule(epoch*m+i, t0, t1)
            gamma = eta/(delta + sqrt_G).reshape(-1,1)
            new_change = gamma*gradients + moment * change

            beta -= new_change
            print(beta[0])
            change = new_change

        epoch += 1

    return beta, epoch


def SGDM_ADAM(X, y, degree, n, eps, delta, beta1, beta2, moment, t0, t1, M):
    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)

    first_moment = 0.0
    second_moment = 0.0
    
    change = 0

    m = int(n/M)

    training_gradient = grad(CostOLS, 0)
    
    epoch = 0
    while(mean_squared_error(beta_linreg, beta)>eps):
        for i in range(m):
            k = M*np.random.randint(m)
            xi = X[k:k+M]
            yi = y[k:k+M]
            #gradients = (2.0/M)* xi.T @ ((xi @ beta)-yi)
            #gradients = training_gradient(beta, n, y, X)

            first_moment = beta1*first_moment + (1-beta1)*gradients
            second_moment = beta2*second_moment+(1-beta2)*gradients*gradients
            first_term = first_moment/(1.0-beta1**(i+1))
            second_term = second_moment/(1.0-beta2**(i+1))

            eta = learning_schedule(epoch*m+i, t0, t1)
            update = eta*first_term/(np.sqrt(second_term)+delta)
            new_change = update + moment * change

            beta -= new_change
            change = new_change
        
        epoch += 1


    return beta, epoch


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def RELU(x):
    return np.maximum(x, 0)

def RELU_derivative(x):
    return np.where(x >= 0, 1, 0)

def Costfunction_grad(y_true, y_pred):
    return (y_pred - y_true)

def initialize_W_and_b(n_features, n_hidden_nodes, n_hidden_layers, n_output_nodes):
    weigths = []
    biases = []

    hidden_weights_1 = np.random.randn(n_features, n_hidden_nodes)
    hidden_bias_1 = np.zeros(n_hidden_nodes) + 0.01

    weigths.append(hidden_weights_1)
    biases.append(hidden_bias_1)

    for i in range(n_hidden_layers - 1):
        hidden_weights_ = np.random.randn(n_hidden_nodes, n_hidden_nodes)
        hidden_bias_ = np.zeros(n_hidden_nodes) + 0.01
        weigths.append(hidden_weights_)
        biases.append(hidden_bias_)

    output_weights = np.random.randn(n_hidden_nodes, n_output_nodes)
    output_bias = np.zeros(n_output_nodes) + 0.01
    weigths.append(output_weights)
    biases.append(output_bias)

    return weigths, biases

def FeedForward(X, W_list, b_list):
    z_list = []
    a_list = []

    z_1 = X @ W_list[0] + b_list[0]
    z_list.append(z_1)

    a_1 = sigmoid(z_1)
    a_list.append(a_1)

    for i in range(len(W_list) - 1):
        z_i = a_list[i] @ W_list[i+1] + b_list[i+1]
        z_list.append(z_i)

        if i == len(W_list) - 2: break

        a_i = sigmoid(z_i)
        a_list.append(a_i)
    
    return z_list, a_list

def BackPropagation(y_train, X_train, W_list, b_list, a_list, z_list, gamma):
    delta_list = []
    delta_out = Costfunction_grad(y_train, z_list[-1])
    delta_list.append(delta_out)

    for i in range(len(W_list) - 1):
        delta_i = (delta_list[-1] @ (W_list[-1 - i]).T) * sigmoid_derivative(a_list[-1 - i])
        delta_list.append(delta_i)
    
    delta_list.reverse()


    W_list[0] -= gamma * (X_train.T @ delta_list[0])
    b_list[0] -= gamma * np.sum(delta_list[0])

    for i in range(len(W_list) - 1):
        W_list[i + 1] -= gamma * (a_list[i].T @ delta_list[i + 1])
        b_list[i + 1] -= gamma * np.sum(delta_list[i + 1])


    return W_list, b_list