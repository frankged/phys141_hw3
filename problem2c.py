import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
L = 30
x0 = -5
sigma = 1
k0 = 5
# Function to compute the second derivative using finite differences
def P(x,t, psi, a):
    return (psi(x + a,t) + psi(x - a,t) - 2 * psi(x,t)) / (a * a)

#this is for c_k(0) = [0, 0, ..., 1,..., 0]
#with the 1 at index related to k0
def gaussian_wavepacket(x,t, x0, sigma, k0):
    return np.exp(-(x - x0)**2 / (2 * sigma**2) - 1j*k0*x) 

N = 1000
x = np.linspace(-L, L, N)
a = x[1] - x[0]
V_0s = [0,k0**2,2*k0**2]
T = np.linspace(0, 5, N)
h = T[1] - T[0]
for V_0 in V_0s:
    #step function
    V_vec = np.where(x < 0, 0, V_0)
    V_mat = diags(V_vec, 0)
    main_diag = -2.0 * np.ones(N)
    off_diag = 1.0 * np.ones(N - 1)
    D = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1])
    I = diags([np.ones(N)], [0], format='csc')
    H = 1/2/a**2 * D - V_mat
    A = I + 1j*h/2*H
    B = I - 1j*h/2*H      
    psi_t = []
    for t in range(len(T)):
        if t == 0:
            psi = gaussian_wavepacket(x,t, x0, sigma, k0)
        else:
            psi_new = spsolve(A, B @ psi)
            psi = psi_new
            psi_t.append(psi.copy())
    #kymograph
    plt.figure(figsize=(10, 6))
    psi_squared = np.abs(psi_t)**2
    # Plot kymograph
    plt.figure(figsize=(10, 6))
    extent = [x[0], x[-1], T[-1], T[0]]  # time on y-axis (top to bottom)
    plt.imshow(psi_squared, extent=extent, aspect='auto', cmap='inferno')
    plt.colorbar(label=r'$|\psi(x,t)|^2$')
    plt.xlabel('Position x')
    plt.ylabel('Time t')
    plt.title(f'Kymograph of $|\psi(x,t)|^2$: V_0 = {V_0}')
    plt.tight_layout()
    plt.show()