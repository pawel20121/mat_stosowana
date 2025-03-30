# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 22:31:13 2025

@author: mozgo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import jn

# Bessel equation rewritten as a system of first-order ODEs
def bessel_eqn(x, y, n):
    dy1 = y[1]
    dy2 = -(x * y[1] + (x**2 - n**2) * y[0]) / x**2
    return [dy1, dy2]

# Initial conditions for J0 and J1
def initial_conditions(n):
    if n == 0:
        return [1.0, 0.0]  # J0(0) = 1, J0'(0) = 0
    elif n == 1:
        return [0.0, 0.5]  # J1(0) = 0, J1'(0) = 0.5

# Solve for n=0 and n=1
x_span = (0.001, 10)  # Avoid division by zero at x=0
x_eval = np.linspace(0.001, 10, 1000)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

for i, n in enumerate([0, 1]):
    y0 = initial_conditions(n)
    sol = solve_ivp(bessel_eqn, x_span, y0, args=(n,), t_eval=x_eval, method='RK45')
    
    # Analytical solution using SciPy's jn function
    j_analytic = jn(n, x_eval)
    
    # Plot numerical and analytical solutions
    axs[i].plot(x_eval, j_analytic, 'o', label='analytic', markersize=3, markerfacecolor='none', markeredgecolor='blue')
    axs[i].plot(sol.t, sol.y[0], '-', label='numeric', color='orange')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('y(x)')
    axs[i].set_title(fr'$J_{n}(x)$, dx=0.001')
    axs[i].legend()
    axs[i].grid()

plt.tight_layout()
plt.show()
