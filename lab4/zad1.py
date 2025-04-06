import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parametry
L = 1.0  # Długość domeny
D = 0.25  # Współczynnik dyfuzji
dx = 0.01  # Krok przestrzenny
x = np.linspace(0, L, int(L/dx))  # Siatka przestrzenna
t_vals = np.array([0.000, 0.002, 0.004, 0.007, 0.012, 0.018, 0.027, 
                   0.040, 0.059, 0.085, 0.122, 0.174, 0.247, 0.351, 
                   0.498, 0.706, 1.000])  # Konkretne wartości czasu

# Warunki początkowe
def f1(x):  # Pierwsza funkcja początkowa
    return np.abs(np.sin(3 * np.pi * x / L))

def f2(x):  # Druga funkcja początkowa
    return 2 * np.abs(np.abs(x - L/2) - L/2)

# Obliczanie współczynników Fouriera
def compute_bn(n, f, L):
    return (2 / L) * np.trapz(f(x) * np.sin(n * np.pi * x / L), x)

# Rozwiązanie równania
def u_xt(x, t, f, terms=50):
    u = np.zeros_like(x)
    for n in range(1, terms + 1):
        bn = compute_bn(n, f, L)
        u += bn * np.sin(n * np.pi * x / L) * np.exp(-n**2 * np.pi**2 * D * t / L**2)
    return u

# Obliczanie wartości u(x,t) dla każdej funkcji początkowej
solutions_f1 = np.array([u_xt(x, t, f1) for t in t_vals])
solutions_f2 = np.array([u_xt(x, t, f2) for t in t_vals])

# --- WYKRES 1: f1(x) ---
plt.figure(figsize=(10, 5))
for i, sol in enumerate(solutions_f1):
    plt.plot(x, sol, label=f"t={t_vals[i]:.3f}")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Ewolucja rozwiązania w czasie dla f1(x)")
plt.legend(ncol=2, fontsize=8, loc='upper right')
plt.grid()
plt.show()

# --- WYKRES 2: f2(x) ---
plt.figure(figsize=(10, 5))
for i, sol in enumerate(solutions_f2):
    plt.plot(x, sol, label=f"t={t_vals[i]:.3f}")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Ewolucja rozwiązania w czasie dla f2(x)")
plt.legend(ncol=2, fontsize=8, loc='upper right')
plt.grid()
plt.show()

# --- WYKRES 3: 3D dla f1(x) ---
X, T = np.meshgrid(x, t_vals)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, solutions_f1, cmap="viridis", edgecolor='none')
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x,t)")
ax.set_title("Rozwiązanie równania dyfuzji w 3D dla f1(x)")
plt.show()

# --- WYKRES 4: 3D dla f2(x) ---
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, solutions_f2, cmap="viridis", edgecolor='none')
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x,t)")
ax.set_title("Rozwiązanie równania dyfuzji w 3D dla f2(x)")
plt.show()

# --- WYKRES 5: Współczynniki Fouriera dla f1(x) ---
n_vals = np.arange(1, 51)
b_n_vals_f1 = np.array([compute_bn(n, f1, L) for n in n_vals])
plt.figure(figsize=(8, 5))
plt.stem(n_vals, b_n_vals_f1, basefmt=" ", use_line_collection=True)
plt.xlabel("n")
plt.ylabel("$b_n$")
plt.title("Współczynniki Fouriera $b_n$ dla f1(x)")
plt.grid()
plt.show()

# --- WYKRES 6: Współczynniki Fouriera dla f2(x) ---
b_n_vals_f2 = np.array([compute_bn(n, f2, L) for n in n_vals])
plt.figure(figsize=(8, 5))
plt.stem(n_vals, b_n_vals_f2, basefmt=" ", use_line_collection=True)
plt.xlabel("n")
plt.ylabel("$b_n$")
plt.title("Współczynniki Fouriera $b_n$ dla f2(x)")
plt.grid()
plt.show()