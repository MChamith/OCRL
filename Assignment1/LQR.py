import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# a) write a python program for LQR

A = np.array([[1, 0.05], [0, 1.2]])
B = np.array([0.05, 1.2]).reshape(-1, 1)

Q = np.matrix(([0.25, 0], [0, 0.25]))
R = 0.1
N = 30

S = np.zeros((N + 1, 2, 2))
K = np.zeros((N, 1, 2))
# S(N+1) = 0 since no H

for i in range(N):
    K_k = -inv(B.T @ S[i] @ B + R) @ B.T @ S[i] @ A
    S[i + 1] = (A + B @ K_k).T @ S[i] @ (A + B @ K_k) + K_k.T @ (R * K_k) + Q
    K[i,:] = K_k

# b) plot results using given x initial values

X = np.zeros((N + 1, 2))
U = np.zeros((N,1))

X[0] = np.array([1, 2])

for i in range(N):
    U[i] = K[N-i-1,:] @ X[i]

    X[i + 1] = A @ X[i] + B @ U[i]

x_axis = [i for i in range(N+1)]
plt.figure(1)
plt.plot(x_axis, X[:,0])
plt.figure(2)
plt.plot(x_axis, X[:,1])
plt.show()
