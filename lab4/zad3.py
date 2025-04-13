import numpy as np
import matplotlib.pyplot as plt

# --- Parametry ---
L = 1.0       # Długość rury
D = 0.25      # Współczynnik dyfuzji
Nx = 100
dx = L / Nx
dt = 0.0001
T_max = 1.0
Nt = int(T_max / dt)

# Warunki brzegowe
C1 = 0.6
C2 = 0.1
x = np.linspace(0, L, Nx)

# --- Funkcje początkowe ---
def f1(x): return np.abs(np.sin(3 * np.pi * x / L))
def f2(x): return 2 * np.abs(np.abs(x - L / 2) - L / 2)
def f3(x): return np.where((x > 0.3) & (x < 0.7), 1, 0)

initial_conditions = [(f1, "f1(x) = |sin(3πx)|"),
                      (f2, "f2(x) = 2 * ||x - 0.5| - 0.5|"),
                      (f3, "f3(x) = prostokąt")]

# --- Rozwiązanie stacjonarne ---
def v(x, C1, C2, L):
    A = C1
    B = (C2 - C1) / L
    return A + B * x

# --- Warunki brzegowe ---
def apply_inhomogeneous_bc(u):
    u[0] = 1.0
    u[-1] = 0.0
    return u

# --- Solver dyfuzji ---
def solve_diffusion(w_initial, D, dx, dt, t):
    Nt = int(t / dt)
    w = np.copy(w_initial)
    r = D * dt / dx**2

    if r > 0.5:
        raise ValueError(f"Unstable: Courant number r = {r} exceeds 0.5.")

    for _ in range(Nt):
        w_new = np.copy(w)
        w_new[1:-1] = w[1:-1] + r * (w[2:] - 2 * w[1:-1] + w[:-2])
        w = apply_inhomogeneous_bc(w_new)
    return w

# --- Wartości czasu ---
t_vals = [0.000, 0.002, 0.004, 0.007, 0.012, 0.018, 0.027, 0.040, 0.059, 0.085, 0.122, 0.174, 0.247, 0.351, 0.498, 0.706, 1.000]

# --- Pętla po funkcjach początkowych ---
for fx, label in initial_conditions:
    w_initial = fx(x) - v(x, C1, C2, L)
    solutions = []
    
    for t in t_vals:
        w_t = solve_diffusion(w_initial, D, dx, dt, t)
        u_t = v(x, C1, C2, L) + w_t
        solutions.append(u_t)
    
    # --- Wykres 2D ---
    plt.figure(figsize=(10, 5))
    for i, sol in enumerate(solutions):
        plt.plot(x, sol, label=f"t={t_vals[i]:.3f}")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title(f"Ewolucja u(x,t) dla warunku początkowego: {label}")
    plt.legend(loc='upper right', fontsize=8)
    plt.grid()
    plt.tight_layout()
    plt.show()

    # --- Wykres 3D ---
    X, T = np.meshgrid(x, t_vals)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, np.array(solutions), cmap="viridis", edgecolor='none')
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u(x,t)")
    ax.set_title(f"Rozwiązanie u(x,t) (3D) dla: {label}")
    plt.tight_layout()
    plt.show()
