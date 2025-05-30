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
mean_positions = [[0 for _ in range(len(T)-1)] for _ in range(len(V_0s))]
count = 0
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
    psi_t = np.array(psi_t)  # Convert list to array
    #kymograph
    plt.figure(figsize=(10, 6))
    # this has ordering (t, x)
    psi_squared = np.abs(psi_t)**2
    P_right = lambda t_index: a * np.sum(psi_squared[t_index, int(N/2):])
    P_left = lambda t_index: a * np.sum(psi_squared[t_index, :int(N/2)])
    P_total =lambda t_index: P_left(t_index) + P_right(t_index)
    P_left_normalized = lambda i: P_left(i)/total_probability[i]
    P_right_normalized = lambda i: P_right(i)/total_probability[i]
    #mean position
    mean_position = lambda t_index: a * np.sum(x*psi_squared[t_index, :]) / P_total(t_index)
    # Plot kymograph
    plt.figure(figsize=(10, 6))
    extent = [x[0], x[-1], T[-1], T[0]]  # time on y-axis (top to bottom)
    total_probability = np.array([P_total(t_index) for t_index in range(len(T)-1)])
    indices = list(range(0, len(T)-1))
    for i in indices:
        # if total_probability[i] == 0:
        #     total_probability[i] = 1e-10 
        plt.plot(i/200,P_left_normalized(i),'o', color='blue', linewidth=1)
        plt.plot(i/200,P_right_normalized(i),'o', color='red', linewidth=1)
        plt.plot(i/200,total_probability[i]/total_probability[0],'o', color='black', linewidth=1)
        mean_positions[count][i] = mean_position(i)
    plt.title(f'Probability of finding the particle in left vs right half: V_0 = {V_0}, k0 = {k0}, x0 = {x0}, red = right, blue = left')
    plt.grid()
    plt.legend()
    plt.xlabel('Time index')
    plt.ylabel('Probability')
    plt.show()
        # plt.plot(i,P_right(i)/total_probability[i],'o', label='Right Probability', color='red')


    # plt.plot(indices, P_left(indices)/total_probability[indices], 'o', label='Left Probability', color='blue', linewidth=1)
    # plt.plot(indices, P_right(indices)/total_probability[indices], 'o', label='Right Probability', color='red')
    plt.show()
    plt.imshow(psi_squared/total_probability[0], extent=extent, aspect='auto', cmap='inferno')
    plt.colorbar(label=r'$|\psi(x,t)|^2$')
    plt.xlabel('Position x')
    plt.ylabel('Time t')
    plt.title(f'Kymograph of $|\psi(x,t)|^2$: V_0 = {V_0}')
    plt.tight_layout()
    plt.show()

    count += 1
# Plot mean positions
indices = list(range(0, len(T)-1))
T_plot = np.linspace(0, T[-2], len(T)-1)
plt.figure(figsize=(10, 6))
for i, V_0 in enumerate(V_0s):
    plt.plot(T_plot, mean_positions[i], label=f'V_0 = {V_0}')
plt.title('Mean Position of the Particle Over Time')
plt.xlabel('Time index')
plt.ylabel('Mean Position')
plt.grid()
plt.legend()
plt.show()