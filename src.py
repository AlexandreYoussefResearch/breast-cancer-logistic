import matplotlib.pyplot as plt
import numpy as np
from scipy import special


def import_data():
    X = []
    Y = []
    f = open("breast-cancer-wisconsin.data", 'r')
    for i in f.readlines():
        lst = i.split(",")
        x_i = []
        test = True
        for j in range(1, len(lst)):
            if j != 10:
                if lst[j] == '?':
                    test = False
                    break
                x_i.append(int(lst[j]))
            else:
                if int(lst[j]) == 2:
                    Y.append(0)
                else:
                    Y.append(1)
        if test == True:
            X.append(x_i)

    return np.array(X), np.array(Y)


def expit_b(x, b):
    """Compute sigmoid(x) - b component-wise."""
    idx = x < 0
    out = np.zeros_like(x)
    exp_x = np.exp(x[idx])
    b_idx = b[idx]
    out[idx] = ((1 - b_idx) * exp_x - b_idx) / (1 + exp_x)
    exp_nx = np.exp(-x[~idx])
    b_nidx = b[~idx]
    out[~idx] = ((1 - b_nidx) - b_nidx * exp_nx) / (1 + exp_nx)
    return out


def f_grad(x, A, b):
    z = A.dot(x)
    s = expit_b(z, b)
    return A.T.dot(s)

def f_gradbis(theta, x, Y):
    #print("theta_grad : ", theta)
    #print(x)
    te = +np.dot(x, theta[1:]) + theta[0]
    #print(te)
    grad = [0]
    for k in range(len(x[0])):
        sol = 0
        for t in range(len(te)):
            if te[t] < 0:
                exp_t = np.exp(te[t])
                sol += (((1 - Y[t]) * exp_t - Y[t]) / (1 + exp_t)) * x[t, k]
            else:
                exp_nt = np.exp(-te[t])
                #print("exp_nt : ", exp_nt)
                sol += (((1 - Y[t]) - Y[t] * exp_nt) / (1 + exp_nt)) * x[t, k]
        grad.append(sol)
    return np.array(grad)

def f_grad2(x, lamb):
    return lamb*x

def gradient():
    (X, Y) = import_data()
    X_test = X[0:136]
    Y_test = Y[0:136]
    X_train = X[136:-1]
    Y_train = Y[136:-1]

    L = 0
    for i in range(len(X_train)):
        L += (np.linalg.norm(X_train[i])) ** 2 + 1
    L *= 1/4

    theta_k = [np.zeros(len(X_train[0])+1)]

    max_iteration = 50000
    precision = 1e-4
    previous_step_size = 1
    iteration = 0
    previous_2 = 1
    while (iteration < max_iteration) and (previous_2 >= precision):
        theta_k.append(theta_k[iteration] - ((1 / L) * f_grad(theta_k[iteration], np.append(np.array([np.ones(len(X_train))]).T, X_train, axis=1), Y_train)))
        previous_step_size = np.sum(np.abs(theta_k[iteration+1] - theta_k[iteration]))
        
        previous_2 = np.abs(f(theta_k[iteration+1],np.append(np.array([np.ones(len(X_train))]).T, X_train, axis=1),Y_train) - f(theta_k[iteration],np.append(np.array([np.ones(len(X_train))]).T, X_train, axis=1),Y_train))
        
        
        
        iteration += 1
        #print("Iteration", iteration, "\n Theta =", theta_k[iteration])

    
    print(iteration)
    return theta_k




def accelerated_gradient():
    (X, Y) = import_data()
    X_test = X[0:136]
    Y_test = Y[0:136]
    X_train = X[136:-1]
    Y_train = Y[136:-1]

    L = 0
    for i in range(len(X)):
        L += (np.linalg.norm(X[i])) ** 2 + 1
    L *= 1/4

    theta_k = [np.zeros(len(X_train[0])+1)]

    y = theta_k[0]
    t = 1

    max_iteration = 50000
    precision = 1e-4
    previous_step_size = 1
    iteration = 0
    previous_2 = 1
    while (iteration <= max_iteration) and (previous_2 > precision):
        prev_t = t
        theta_k.append(y - (1/L) * f_grad(y, np.append(np.array([np.ones(len(X_train))]).T, X_train, axis=1), Y_train))
        t = (1 + np.sqrt(1+4*t**2))/2
        y = theta_k[iteration+1] + (theta_k[iteration+1]-theta_k[iteration])*(prev_t-1)/t

        previous_step_size = np.sum(np.abs(theta_k[iteration+1] - theta_k[iteration]))
        previous_2 = np.abs(f(theta_k[iteration+1],np.append(np.array([np.ones(len(X_train))]).T, X_train, axis=1),Y_train) - f(theta_k[iteration],np.append(np.array([np.ones(len(X_train))]).T, X_train, axis=1),Y_train))
        iteration += 1
        #print("Iteration", iteration, "\n Theta =", previous_step_size)

    print(iteration)
    return theta_k


def logsig(x):
    """Compute the log-sigmoid function component-wise."""
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    return out


def f(x, A, b):
    """Logistic loss, numerically stable implementation.

    Parameters
    ----------
    x: array-like, shape (n_features,)
        Coefficients

    A: array-like, shape (n_samples, n_features)
        Data matrix

    b: array-like, shape (n_samples,)
        Labels

    Returns
    -------
    loss: float
    """
    z = np.dot(A, x)
    b = np.asarray(b)
    return np.sum((1 - b) * z - logsig(z))


def plot():
    (X, Y) = import_data()
    X_test = X[0:136]
    Y_test = Y[0:136]
    X_train = X[136:-1]
    Y_train = Y[136:-1]
    theta_lst = accelerated_gradient()
    theta_lst2 = gradient()
    f_theta = []
    f_theta2 = []
    for i in range(len(theta_lst)):
        f_theta.append(f(theta_lst[i], np.append(np.array([np.ones(len(X_train))]).T, X_train, axis=1), Y_train))
        
    for i in range(len(theta_lst2)):
        f_theta2.append(f(theta_lst2[i], np.append(np.array([np.ones(len(X_train))]).T, X_train, axis=1), Y_train))
    
    print ("Theta accelerated  =", theta_lst[-1])
    print ("Theta gradient =", theta_lst2[-1])
    print("f accelerated = ",f_theta[-1])
    print("f theta gradient = ",f_theta2[-1])
    
    k = np.arange(0, len(theta_lst), 1)
    k2 = np.arange(0, len(theta_lst2),1)
    plt.scatter(k, f_theta, label = "Accelerated gradient method", color = "black")
    plt.scatter(k2,f_theta2, label = "Gradient method", color = "red")
    plt.xlabel("Iteration k of the algorithm   [-]")
    plt.ylabel("Value of f(theta_k)")
    plt.title("Convergence of gradient vs accelerated gradient methods")
    plt.legend()
    plt.savefig("Convergence.pdf")
    plt.show()


def m_theta(x,theta):
    term = 0
    for i in range(1, len(theta)):
        term += x[i-1]*theta[i]
    return 1/(1 + np.exp(-(theta[0] + term)))


def clasification_err(X_test, Y_test, theta):
    Err = 0
    for i in range(len(X_test)):
        m = m_theta(X_test[i], theta)
        if(m < 0.5):
            if(Y_test[i] == 1):
                Err += 1
        if(m >= 0.5):
            if(Y_test[i] == 0):
                Err += 1
    return Err

"""
(X, Y) = import_data()
X_test = X[0:136]
Y_test = Y[0:136]
X_train = X[136:-1]
Y_train = Y[136:-1]
the = gradient()
theta = the[-1]
thet2 = accelerated_gradient()
theta2 = thet2[-1]

print(clasification_err(X_test, Y_test, theta))
print (len(X_test))
print(clasification_err(X_test, Y_test, theta2))
"""

gradient()
#plot()
accelerated_gradient()
