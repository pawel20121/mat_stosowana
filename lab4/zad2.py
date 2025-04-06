# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 11:20:02 2025

@author: mozgo
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Parametry
L = 1.0       # Długość przedziału
D = 0.25      # Współczynnik dyfuzji
Nx = 100      # Liczba punktów przestrzennych
dx = L / Nx   # Rozdzielczość przestrzenna
dt = 0.00005   # Zmniejszamy dt, aby zapewnić stabilność

# Współczynniki Fouriera dla początkowego rozkładu f(x)
def f(x):
    return np.sin(2 * np.pi * x)  # Przykładowa funkcja początkowa

x = np.linspace(0, L, Nx)
u = f(x)  # Warunek początkowy

# Warunki początkowe
def f1(x):  # Pierwsza funkcja początkowa
    return np.abs(np.sin(3 * np.pi * x / L))

def f2(x):  # Druga funkcja początkowa
    return 2 * np.abs(np.abs(x - L / 2) - L / 2)
def f3(x):  # Trzecia funkcja - prostokątna
    return np.where((x > 0.3) & (x < 0.7), 1, 0)

def compute_bn(n, f, L, x):
    return (2 / L) * np.trapz(f(x) * np.sin(n * np.pi * x / L), x)

# Warunki brzegowe Neumanna (pochodne zerowe na brzegach)
def apply_neumann_bc(u):
    u[0] = u[1]
    u[-1] = u[-2]
    return u

# Metoda różnic skończonych (schemat explicite)
def solve_diffusion(u, D, dx, dt,t,terms = 50):
    Nt = int(t / dt)
    r = D * dt / dx**2  # Liczba Couranta
    if r > 0.5:
        raise ValueError(f"Unstable: Courant number r = {r} exceeds 0.5.")
    for _ in range(Nt):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        u = apply_neumann_bc(u_new)
    return u

t_vals = [0.000, 0.002, 0.004, 0.007, 0.012, 0.018, 0.027, 0.040, 0.059, 0.085, 0.122, 0.174, 0.247, 0.351, 0.498, 0.706, 1.000]  # Wybrane momenty czasowe
solutions_f1 = [f1(x)]  # Przechowywanie wyników
solutions_f1 = np.array([solve_diffusion(solutions_f1[-1], D,dx,dt,t) for t in t_vals])
solutions_f2 = [f2(x)]  # Przechowywanie wyników
solutions_f2 = np.array([solve_diffusion(solutions_f2[-1], D,dx,dt,t) for t in t_vals])
solutions_f3 = [f3(x)]  # Przechowywanie wyników
solutions_f3 = np.array([solve_diffusion(solutions_f3[-1], D,dx,dt,t) for t in t_vals])
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

# --- WYKRES 3: f3(x) ---
plt.figure(figsize=(10, 5))
for i, sol in enumerate(solutions_f3):
    plt.plot(x, sol, label=f"t={t_vals[i]:.3f}")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Ewolucja rozwiązania w czasie dla f3(x) - funkcja prostokątna")
plt.legend(ncol=2, fontsize=8, loc='upper right')
plt.grid()
plt.show()

# --- WYKRES 4: 3D dla f1(x) ---
X, T = np.meshgrid(x, t_vals)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, solutions_f1, cmap="viridis", edgecolor='none')
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x,t)")
ax.set_title("Rozwiązanie równania dyfuzji w 3D dla f1(x)")
plt.show()

# --- WYKRES 5: 3D dla f2(x) ---
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, solutions_f2, cmap="viridis", edgecolor='none')
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x,t)")
ax.set_title("Rozwiązanie równania dyfuzji w 3D dla f2(x)")
plt.show()

# --- WYKRES 6: 3D dla f3(x) ---
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, solutions_f3, cmap="viridis", edgecolor='none')
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x,t)")
ax.set_title("Rozwiązanie równania dyfuzji w 3D dla f3(x) - funkcja prostokątna")
plt.show()

# --- WYKRES 7: Współczynniki Fouriera dla f1(x) ---
n_vals = np.arange(1, 51)
b_n_vals_f1 = np.array([compute_bn(n, f1, L, x) for n in n_vals])
plt.figure(figsize=(8, 5))
plt.stem(n_vals, b_n_vals_f1, basefmt=" ", use_line_collection=True)
plt.xlabel("n")
plt.ylabel("$b_n$")
plt.title("Współczynniki Fouriera $b_n$ dla f1(x)")
plt.grid()
plt.show()

# --- WYKRES 8: Współczynniki Fouriera dla f2(x) ---
b_n_vals_f2 = np.array([compute_bn(n, f2, L,x) for n in n_vals])
plt.figure(figsize=(8, 5))
plt.stem(n_vals, b_n_vals_f2, basefmt=" ", use_line_collection=True)
plt.xlabel("n")
plt.ylabel("$b_n$")
plt.title("Współczynniki Fouriera $b_n$ dla f2(x)")
plt.grid()
plt.show()
# --- WYKRES 9: Współczynniki Fouriera dla f3(x) ---
b_n_vals_f3 = np.array([compute_bn(n, f3, L, x) for n in n_vals])
plt.figure(figsize=(8, 5))
plt.stem(n_vals, b_n_vals_f3, basefmt=" ", use_line_collection=True)
plt.xlabel("n")
plt.ylabel("$b_n$")
plt.title("Współczynniki Fouriera $b_n$ dla f3(x) - funkcja prostokątna")
plt.grid()
plt.show()