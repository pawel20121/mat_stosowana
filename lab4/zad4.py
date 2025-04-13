# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 21:45:56 2025

@author: mozgo
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parametry
L = 1.0
D = 0.25
Nx = 100
dx = L / Nx
x = np.linspace(0, L, Nx + 1)

C1 = 0.6
C2 = 0.1
A = C1
B = (C2 - C1) / L

# Warunek początkowy f(x)
def f(x):
    return np.sin(5 * np.pi * x) + 1  # przykład jak na wykresie

def f1(x):  # Pierwsza funkcja początkowa
    return np.abs(np.sin(3 * np.pi * x / L))

def f2(x):  # Druga funkcja początkowa
    return 2 * np.abs(np.abs(x - L / 2) - L / 2)
def f3(x):  # Trzecia funkcja - prostokątna
    return np.where((x > 0.3) & (x < 0.7), 1, 0)
# V(x) – rozwiązanie stacjonarne
def v(x):
    return A + B * x

# W(x,0)
def w0_1(x):
    return f1(x) - v(x)
def w0_2(x):
    return f2(x) - v(x)
def w0_3(x):
    return f3(x) - v(x)
# Rozwiązanie równania dyfuzji dla jednorodnych brzegów
def solve_diffusion(u0, D, dx, dt, t_max):
    Nx = len(u0) - 1
    Nt = int(t_max / dt)
    alpha = D * dt / dx**2

    u = u0.copy()
    for _ in range(Nt):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + alpha * (u[2:] - 2*u[1:-1] + u[:-2])
        u_new[0] = 0
        u_new[-1] = 0
        u = u_new
    return u

# Ewolucja czasowa
dt = 0.00005
t_vals = [0.000, 0.002, 0.004, 0.007, 0.012, 0.018, 0.027, 0.040, 0.059,
          0.085, 0.122, 0.174, 0.247, 0.351, 0.498, 0.706, 1.000]

w_solutions = np.array([solve_diffusion(w0_1(x), D, dx, dt, t) for t in t_vals])
u_solutions = w_solutions + v(x)  # dodajemy funkcję v(x)
w_solutions1 = np.array([solve_diffusion(w0_2(x), D, dx, dt, t) for t in t_vals])
u_solutions1 = w_solutions1 + v(x)  # dodajemy funkcję v(x)
w_solutions2 = np.array([solve_diffusion(w0_3(x), D, dx, dt, t) for t in t_vals])
u_solutions2 = w_solutions2 + v(x)  # dodajemy funkcję v(x)
# --- WYKRES 2D ---
plt.figure(figsize=(7, 5))
for i, u in enumerate(u_solutions):
    plt.plot(x, u, marker='.', label=f"t={t_vals[i]:.3f}")
plt.plot(x, f1(x), 'g', linewidth=2, label="f(x)")  # początkowy profil
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title(f"D={D}, L={L}, C₁={C1}, C₂={C2}, dx={dx}")
plt.legend(ncol=2, fontsize=8, loc='upper right')
plt.grid()
plt.show()

# --- WYKRES 3D ---
X, T = np.meshgrid(x, t_vals)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, u_solutions, cmap='viridis', edgecolor='none')
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x,t)")
ax.set_title(f"D={D}, L={L}, C₁={C1}, C₂={C2}, dx={dx}")
plt.show()


# --- WYKRES 2D ---
plt.figure(figsize=(7, 5))
for i, u in enumerate(u_solutions1):
    plt.plot(x, u, marker='.', label=f"t={t_vals[i]:.3f}")
plt.plot(x, f2(x), 'g', linewidth=2, label="f(x)")  # początkowy profil
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title(f"D={D}, L={L}, C₁={C1}, C₂={C2}, dx={dx}")
plt.legend(ncol=2, fontsize=8, loc='upper right')
plt.grid()
plt.show()

# --- WYKRES 3D ---
X, T = np.meshgrid(x, t_vals)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, u_solutions1, cmap='viridis', edgecolor='none')
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x,t)")
ax.set_title(f"D={D}, L={L}, C₁={C1}, C₂={C2}, dx={dx}")
plt.show()# --- WYKRES 2D ---
plt.figure(figsize=(7, 5))
for i, u in enumerate(u_solutions2):
    plt.plot(x, u, marker='.', label=f"t={t_vals[i]:.3f}")
plt.plot(x, f3(x), 'g', linewidth=2, label="f(x)")  # początkowy profil
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title(f"D={D}, L={L}, C₁={C1}, C₂={C2}, dx={dx}")
plt.legend(ncol=2, fontsize=8, loc='upper right')
plt.grid()
plt.show()

# --- WYKRES 3D ---
X, T = np.meshgrid(x, t_vals)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, u_solutions2, cmap='viridis', edgecolor='none')
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x,t)")
ax.set_title(f"D={D}, L={L}, C₁={C1}, C₂={C2}, dx={dx}")
plt.show()