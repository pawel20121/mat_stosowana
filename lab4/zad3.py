# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 12:15:02 2025

@author: mozgo
"""

import numpy as np
import matplotlib.pyplot as plt

# Parametry
L = 1.0       # Długość rury
D = 0.25      # Współczynnik dyfuzji
Nx = 100      # Liczba punktów przestrzennych
dx = L / Nx   # Rozdzielczość przestrzenna
dt = 0.0001   # Rozdzielczość czasowa
T_max = 1.0   # Czas maksymalny
Nt = int(T_max / dt)  # Liczba kroków czasowych

# Parametry warunków brzegowych
C1 = 0.6  # Warunek brzegowy w x=0
C2 = 0.1  # Warunek brzegowy w x=L
# Siatka przestrzenna
x = np.linspace(0, L, Nx)

# Warunki początkowe
def f1(x):  # Pierwsza funkcja początkowa
    return np.abs(np.sin(3 * np.pi * x / L))

# Funkcja v(x) = A + Bx
def v(x, C1, C2, L):
    A = C1
    B = (C2 - C1) / L
    return A + (B * x)

# Warunki brzegowe - inhomogeniczne
def apply_inhomogeneous_bc(u):
    # Na końcu x=0 i x=L różne warunki brzegowe
    u[0] = 1.0   # Warunek brzegowy na początku rury
    u[-1] = 0.0  # Warunek brzegowy na końcu rury
    return u

# Funkcja rozwiązująca równanie dyfuzji dla w(x, t)
def solve_diffusion(w_initial, D, dx, dt, t):
    # Parametry
    Nt = int(t / dt)  # Liczba kroków czasowych
    Nx = len(w_initial)   # Liczba punktów przestrzennych
    w = np.copy(w_initial)  # Rozwiązanie
    
    r = D * dt / dx**2  # Liczba Couranta
    
    if r > 0.5:
        raise ValueError(f"Unstable: Courant number r = {r} exceeds 0.5.")
    
    # Główna pętla czasowa
    for _ in range(Nt):
        w_new = np.copy(w)
        w_new[1:-1] = w[1:-1] + r * (w[2:] - 2 * w[1:-1] + w[:-2])
        w = apply_inhomogeneous_bc(w_new)  # Aplikowanie warunków brzegowych
    return w


# Obliczenia
t_vals = [0.000, 0.002, 0.004, 0.007, 0.012, 0.018, 0.027, 0.040, 0.059, 0.085, 0.122, 0.174, 0.247, 0.351, 0.498, 0.706, 1.000]
solutions = []

# Obliczanie rozwiązania w(x, t)
w_initial = f1(x) - v(x, C1, C2, L)  # w(x,0) = f(x) - v(x)
for t in t_vals:
    w_t = solve_diffusion(w_initial, D, dx, dt, t)
    solutions.append(v(x, C1, C2, L) + w_t)

# --- Wykres 1: Ewolucja rozwiązania w czasie ---
plt.figure(figsize=(10, 5))
for i, sol in enumerate(solutions):  # Rysujemy co 10 kroków czasowych
    plt.plot(x, sol, label=f"t={t_vals[i]:.3f}")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Ewolucja rozwiązania w czasie")
plt.legend(loc='upper right')
plt.grid()
plt.show()

# --- Wykres 2: 3D wykres rozwiązania u(x,t) ---
X, T = np.meshgrid(x, t_vals)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, np.array(solutions), cmap="viridis", edgecolor='none')
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x,t)")
ax.set_title("Rozwiązanie równania dyfuzji w 3D")
plt.show()