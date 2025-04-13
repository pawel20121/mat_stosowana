# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 10:39:51 2025

@author: mozgo
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev as Cheb

# Ustawienia
x = np.linspace(-np.pi, np.pi, 1000)
functions = {
    'exp(x)': np.exp(x),
    '|sin(x)|': np.abs(np.sin(x))
}

# 1. Aproksymacja szeregami Taylora
def taylor_series_exp(x, n):
    return sum([x**i / np.math.factorial(i) for i in range(n)])

def taylor_series_sin_abs(x, n):
    # rozwinięcie Taylora |sin(x)| nie istnieje globalnie, przybliżymy przez |suma_n sin(x)|
    sin_approx = sum([(-1)**i * x**(2*i+1) / np.math.factorial(2*i+1) for i in range(n)])
    return np.abs(sin_approx)

# 2. Szereg Fouriera
def fourier_series_exp(x, mu, N):
    a0 = 2 * np.sinh(mu * np.pi) / np.pi
    result = a0 * 1/2*mu
    for n in range(1, N + 1):
        an = ((2 * np.sinh(mu * np.pi)) / (np.pi)) * (((-1)**n * (mu * np.cos(n * x) - n * np.sin(n * x)))) / (mu**2 + n**2)
        result += an
    return result

def fourier_series_abs_sin(x, N):
    result = 2 / np.pi
    for n in range(1, N + 1):
        result -= 4 / (np.pi) * ((4 * n**2 - 1)) * np.cos(2 * n * x)
    return result

# 3. WLS z wielomianami standardowymi
def wls_poly_fit(x, y, degree, weights):
    X = np.vander(x, degree + 1, increasing=True)
    W = np.diag(weights)
    coeffs = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
    return X @ coeffs

# 4. WLS z wielomianami Czebyszewa
def wls_chebyshev_fit(x, y, degree, weights):
    T = Cheb.fit(x, y, degree, w=weights)
    return T(x)

# Rysowanie
def plot_combined_approximations():
    for name, y_true in functions.items():
        plt.figure(figsize=(10, 6))
        plt.title(f'Aproksymacja funkcji {name}')
        plt.plot(x, y_true, label='Oryginalna', linewidth=2)

        # Taylor
        taylor = taylor_series_exp(x, 5) if 'exp' in name else taylor_series_sin_abs(x, 5)
        plt.plot(x, taylor, label='Taylor (n=5)', linestyle='--')

        # # Fourier
        # if 'exp' in name:
        #     fourier = fourier_series_exp(x, 1, 10)
        # else:
        #     fourier = fourier_series_abs_sin(x, 10)
        # plt.plot(x, fourier, label='Fourier (n=10)', linestyle=':')

        # WLS Poly
        weights_const = np.ones_like(x)
        wls_poly = wls_poly_fit(x, y_true, 5, weights_const)
        plt.plot(x, wls_poly, label='WLS Poly (5)', linestyle='-.')

        # WLS Chebyshev
        cheb = wls_chebyshev_fit(x, y_true, 5, weights_const)
        plt.plot(x, cheb, label='WLS Chebyshev (5)', linestyle=(0, (3, 5, 1, 5)))

        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

plot_combined_approximations()

def plot_combined_approximations_with_error():
    for name, y_true in functions.items():
        plt.figure(figsize=(14, 10))
        plt.subplot(2, 1, 1)
        plt.title(f'Aproksymacja funkcji {name}')
        plt.plot(x, y_true, label='Oryginalna', linewidth=2)

        # Aproksymacje
        taylor = taylor_series_exp(x, 10) if 'exp' in name else taylor_series_sin_abs(x, 10)
        fourier = fourier_series_exp(x, 1, 20) if 'exp' in name else fourier_series_abs_sin(x, 10)
        weights_const = np.ones_like(x)
        wls_poly = wls_poly_fit(x, y_true, 10, weights_const)
        cheb = wls_chebyshev_fit(x, y_true, 10, weights_const)
        x_pow = x**10

        # Wykresy aproksymacji
        plt.plot(x, taylor, '--', label='Taylor (n=10)')
        plt.plot(x, fourier, ':', label='Fourier (n=10)')
        plt.plot(x, wls_poly, '-.', label='WLS Poly (n=10)')
        plt.plot(x, cheb, linestyle=(0, (3, 5, 1, 5)), label='Chebyshev (n=10)')
        plt.plot(x, x_pow, linestyle='dotted', label=r'$x^{10}$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        plt.legend()

        # Błąd aproksymacji
        plt.subplot(2, 1, 2)
        plt.title('Błąd aproksymacji (|oryginalna - przybliżenie|)')
        plt.plot(x, np.abs(y_true - taylor), '--', label='Taylor error')
        plt.plot(x, np.abs(y_true - fourier), ':', label='Fourier error')
        plt.plot(x, np.abs(y_true - wls_poly), '-.', label='WLS Poly error')
        plt.plot(x, np.abs(y_true - cheb), linestyle=(0, (3, 5, 1, 5)), label='Chebyshev error')
        plt.plot(x, np.abs(y_true - x_pow), linestyle='dotted', label=r'$x^{10}$ error')
        plt.xlabel('x')
        plt.ylabel('Błąd')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

# plot_combined_approximations_with_error()