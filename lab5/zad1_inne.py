# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 11:45:22 2025

@author: mozgo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_chebyt  # For Chebyshev polynomials

# Define the original functions
def exp_func(x):
    return np.exp(x)

def abs_sin_func(x):
    return np.abs(np.sin(x))

# Taylor series approximation for e^x
def taylor_exp(x, n_terms):
    result = 0
    for i in range(n_terms):
        result += x**i / np.math.factorial(i)
    return result

# Taylor series approximation for sin(x)
def taylor_sin(x, n_terms):
    result = 0
    for i in range(n_terms):
        term = 2*i + 1  # Only odd terms
        if i % 2 == 0:  # Alternating signs
            result += x**term / np.math.factorial(term)
        else:
            result -= x**term / np.math.factorial(term)
    return result

# Fourier series approximation for e^x
def fourier_exp(x, n_terms, mu=1):
    # The formula provided in the problem
    result = 2 * np.sinh(mu*np.pi) / np.pi * (1/(2*mu))
    for n in range(1, n_terms+1):
        numerator = (-1)**n * (mu * np.cos(n*x) - n * np.sin(n*x))
        denominator = mu**2 + n**2
        result += 2 * np.sinh(mu*np.pi) / np.pi * numerator / denominator
    return result

# Fourier series approximation for |sin(x)|
def fourier_abs_sin(x, n_terms):
    # The formula provided in the problem, fixing the n=1 case
    result = 2/np.pi
    
    # Use the exact coefficient formulation from the problem
    # First term (n=1) is handled separately
    result -= 4/(np.pi) * (1/3) * np.cos(1*x)
    
    # Remaining terms
    for n in range(2, n_terms+1):
        if n % 2 == 0:  # Even terms
            continue  # These terms are zero
        
        # For odd n > 1
        term = 4/(np.pi * n * (1 - n**2)) * np.cos(n*x)
        result -= term
    
    return result

# WLS approximation with polynomials
def wls_poly(x, coeffs):
    result = 0
    for i, coeff in enumerate(coeffs):
        result += coeff * x**i
    return result

# Calculate the WLS coefficients for polynomials
def calculate_wls_poly_coeffs(func, degree, x_range, weights):
    # Create the matrix X of powers of x
    X = np.vander(x_range, degree+1, increasing=True)
    
    # Calculate the weighted coefficients
    W = np.diag(weights)
    XtW = X.T @ W
    A = XtW @ X
    b = XtW @ func(x_range)
    
    # Solve the linear system for coefficients
    coeffs = np.linalg.solve(A, b)
    return coeffs

# WLS approximation with Chebyshev polynomials
def wls_chebyshev(x, coeffs):
    result = 0
    for i, coeff in enumerate(coeffs):
        result += coeff * eval_chebyt(i, x)
    return result

# Calculate the WLS coefficients for Chebyshev polynomials
def calculate_wls_chebyshev_coeffs(func, degree, x_range, weights):
    # Create the matrix of Chebyshev polynomials
    X = np.array([[eval_chebyt(j, x_i) for j in range(degree+1)] for x_i in x_range])
    
    # Calculate the weighted coefficients
    W = np.diag(weights)
    XtW = X.T @ W
    A = XtW @ X
    b = XtW @ func(x_range)
    
    # Solve the linear system for coefficients
    coeffs = np.linalg.solve(A, b)
    return coeffs

# Generate x values for plotting
x_interval = np.linspace(-np.pi, np.pi, 1000)

# Plot approximation errors for e^x
def plot_exp_approximations():
    # Original function
    original_exp = exp_func(x_interval)
    
    # Taylor series approximation
    taylor_n = 5
    taylor_approx = taylor_exp(x_interval, taylor_n)
    taylor_error = taylor_approx - original_exp
    
    # Fourier series approximation
    fourier_n = 10
    fourier_approx = np.array([fourier_exp(x, fourier_n) for x in x_interval])
    fourier_error = fourier_approx - original_exp
    
    # WLS with polynomials
    x_wls = np.linspace(-1, 1, 100)  # Rescaled interval for WLS
    x_scaled = 2 * (x_interval - (-np.pi)) / (np.pi - (-np.pi)) - 1  # Scale x to [-1, 1]
    
    # Constant weighting
    const_weights = np.ones_like(x_wls)
    poly_degree = 5
    poly_coeffs_const = calculate_wls_poly_coeffs(exp_func, poly_degree, x_wls, const_weights)
    poly_approx_const = wls_poly(x_scaled, poly_coeffs_const)
    poly_error_const = poly_approx_const - original_exp
    
    # Linear weighting
    linear_weights = 1 - np.abs(x_wls)
    poly_coeffs_linear = calculate_wls_poly_coeffs(exp_func, poly_degree, x_wls, linear_weights)
    poly_approx_linear = wls_poly(x_scaled, poly_coeffs_linear)
    poly_error_linear = poly_approx_linear - original_exp
    
    # WLS with Chebyshev polynomials
    cheb_degree = 5
    cheb_coeffs_const = calculate_wls_chebyshev_coeffs(exp_func, cheb_degree, x_wls, const_weights)
    cheb_approx_const = wls_chebyshev(x_scaled, cheb_coeffs_const)
    cheb_error_const = cheb_approx_const - original_exp
    
    # Linear weighting for Chebyshev
    cheb_coeffs_linear = calculate_wls_chebyshev_coeffs(exp_func, cheb_degree, x_wls, linear_weights)
    cheb_approx_linear = wls_chebyshev(x_scaled, cheb_coeffs_linear)
    cheb_error_linear = cheb_approx_linear - original_exp
    
    # Create figure for approximation and errors
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot original function and approximations
    axs[0, 0].plot(x_interval, original_exp, 'k-', label='Original e^x')
    axs[0, 0].plot(x_interval, taylor_approx, 'r-', label=f'Taylor N={taylor_n}')
    axs[0, 0].plot(x_interval, fourier_approx, 'b-', label=f'Fourier N={fourier_n}')
    axs[0, 0].plot(x_interval, poly_approx_const, 'g-', label=f'WLS Poly N={poly_degree}')
    axs[0, 0].plot(x_interval, cheb_approx_const, 'y-', label=f'Chebyshev N={cheb_degree}')
    axs[0, 0].set_title('Function e^x')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot approximation errors
    axs[0, 1].plot(x_interval, taylor_error, 'r-', label=f'Taylor N={taylor_n}')
    axs[0, 1].plot(x_interval, fourier_error, 'b-', label=f'Fourier N={fourier_n}')
    axs[0, 1].plot(x_interval, poly_error_const, 'g-', label=f'WLS Poly N={poly_degree}')
    axs[0, 1].plot(x_interval, cheb_error_const, 'y-', label=f'Chebyshev N={cheb_degree}')
    axs[0, 1].set_title('Approximation Error for e^x')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot polynomial basis functions
    x_basis = np.linspace(-1, 1, 100)
    axs[1, 0].set_title('Polynomial Basis Functions')
    for i in range(poly_degree + 1):
        axs[1, 0].plot(x_basis, x_basis**i, label=f'x^{i}')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # Plot Chebyshev basis functions
    axs[1, 1].set_title('Chebyshev Polynomials')
    for i in range(cheb_degree + 1):
        axs[1, 1].plot(x_basis, eval_chebyt(i, x_basis), label=f'T_{i}')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('exp_approximations.png', dpi=300)
    plt.show()

# Plot approximation errors for |sin(x)|
def plot_abs_sin_approximations():
    # Original function
    original_abs_sin = abs_sin_func(x_interval)
    
    # Taylor series approximation - for |sin(x)| we need to take absolute value of sin approximation
    taylor_n = 5
    taylor_approx = np.abs(taylor_sin(x_interval, taylor_n))
    taylor_error = taylor_approx - original_abs_sin
    
    # Fourier series approximation
    fourier_n = 10
    fourier_approx = fourier_abs_sin(x_interval, fourier_n)
    fourier_error = fourier_approx - original_abs_sin
    
    # WLS with polynomials
    x_wls = np.linspace(-1, 1, 100)  # Rescaled interval for WLS
    x_scaled = 2 * (x_interval - (-np.pi)) / (np.pi - (-np.pi)) - 1  # Scale x to [-1, 1]
    
    # Constant weighting
    const_weights = np.ones_like(x_wls)
    poly_degree = 5
    poly_coeffs_const = calculate_wls_poly_coeffs(abs_sin_func, poly_degree, x_wls, const_weights)
    poly_approx_const = wls_poly(x_scaled, poly_coeffs_const)
    poly_error_const = poly_approx_const - original_abs_sin
    
    # Linear weighting
    linear_weights = 1 - np.abs(x_wls)
    poly_coeffs_linear = calculate_wls_poly_coeffs(abs_sin_func, poly_degree, x_wls, linear_weights)
    poly_approx_linear = wls_poly(x_scaled, poly_coeffs_linear)
    poly_error_linear = poly_approx_linear - original_abs_sin
    
    # WLS with Chebyshev polynomials
    cheb_degree = 5
    cheb_coeffs_const = calculate_wls_chebyshev_coeffs(abs_sin_func, cheb_degree, x_wls, const_weights)
    cheb_approx_const = wls_chebyshev(x_scaled, cheb_coeffs_const)
    cheb_error_const = cheb_approx_const - original_abs_sin
    
    # Linear weighting for Chebyshev
    cheb_coeffs_linear = calculate_wls_chebyshev_coeffs(abs_sin_func, cheb_degree, x_wls, linear_weights)
    cheb_approx_linear = wls_chebyshev(x_scaled, cheb_coeffs_linear)
    cheb_error_linear = cheb_approx_linear - original_abs_sin
    
    # Create figure for approximation and errors
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot original function and approximations
    axs[0, 0].plot(x_interval, original_abs_sin, 'k-', label='Original |sin(x)|')
    axs[0, 0].plot(x_interval, taylor_approx, 'r-', label=f'Taylor N={taylor_n}')
    axs[0, 0].plot(x_interval, fourier_approx, 'b-', label=f'Fourier N={fourier_n}')
    axs[0, 0].plot(x_interval, poly_approx_const, 'g-', label=f'WLS Poly N={poly_degree}')
    axs[0, 0].plot(x_interval, cheb_approx_const, 'y-', label=f'Chebyshev N={cheb_degree}')
    axs[0, 0].set_title('Function |sin(x)|')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot approximation errors
    axs[0, 1].plot(x_interval, taylor_error, 'r-', label=f'Taylor N={taylor_n}')
    axs[0, 1].plot(x_interval, fourier_error, 'b-', label=f'Fourier N={fourier_n}')
    axs[0, 1].plot(x_interval, poly_error_const, 'g-', label=f'WLS Poly N={poly_degree}')
    axs[0, 1].plot(x_interval, cheb_error_const, 'y-', label=f'Chebyshev N={cheb_degree}')
    axs[0, 1].set_title('Approximation Error for |sin(x)|')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot polynomial basis functions
    x_basis = np.linspace(-1, 1, 100)
    axs[1, 0].set_title('Polynomial Basis Functions')
    for i in range(poly_degree + 1):
        axs[1, 0].plot(x_basis, x_basis**i, label=f'x^{i}')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # Plot Chebyshev basis functions
    axs[1, 1].set_title('Chebyshev Polynomials')
    for i in range(cheb_degree + 1):
        axs[1, 1].plot(x_basis, eval_chebyt(i, x_basis), label=f'T_{i}')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('abs_sin_approximations.png', dpi=300)
    plt.show()

# Compare different weighting functions for WLS
def compare_weightings():
    # Original functions
    original_exp = exp_func(x_interval)
    original_sin = abs_sin_func(x_interval)
    
    # Scaled x values for WLS
    x_wls = np.linspace(-1, 1, 100)
    x_scaled = 2 * (x_interval - (-np.pi)) / (np.pi - (-np.pi)) - 1
    
    # Different weighting functions
    const_weights = np.ones_like(x_wls)
    linear_weights = 1 - np.abs(x_wls)
    
    # Polynomial approximations
    poly_degree = 5
    
    # e^x approximations
    poly_coeffs_exp_const = calculate_wls_poly_coeffs(exp_func, poly_degree, x_wls, const_weights)
    poly_approx_exp_const = wls_poly(x_scaled, poly_coeffs_exp_const)
    
    poly_coeffs_exp_linear = calculate_wls_poly_coeffs(exp_func, poly_degree, x_wls, linear_weights)
    poly_approx_exp_linear = wls_poly(x_scaled, poly_coeffs_exp_linear)
    
    # |sin(x)| approximations
    poly_coeffs_sin_const = calculate_wls_poly_coeffs(abs_sin_func, poly_degree, x_wls, const_weights)
    poly_approx_sin_const = wls_poly(x_scaled, poly_coeffs_sin_const)
    
    poly_coeffs_sin_linear = calculate_wls_poly_coeffs(abs_sin_func, poly_degree, x_wls, linear_weights)
    poly_approx_sin_linear = wls_poly(x_scaled, poly_coeffs_sin_linear)
    
    # Create figure for comparison
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot e^x approximations
    axs[0, 0].plot(x_interval, original_exp, 'k-', label='Original e^x')
    axs[0, 0].plot(x_interval, poly_approx_exp_const, 'g-', label='Constant Weighting')
    axs[0, 0].plot(x_interval, poly_approx_exp_linear, 'r-', label='Linear Weighting')
    axs[0, 0].set_title('e^x Approximation with Different Weightings')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot e^x errors
    axs[0, 1].plot(x_interval, poly_approx_exp_const - original_exp, 'g-', label='Constant Weighting Error')
    axs[0, 1].plot(x_interval, poly_approx_exp_linear - original_exp, 'r-', label='Linear Weighting Error')
    axs[0, 1].set_title('e^x Approximation Errors')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot |sin(x)| approximations
    axs[1, 0].plot(x_interval, original_sin, 'k-', label='Original |sin(x)|')
    axs[1, 0].plot(x_interval, poly_approx_sin_const, 'g-', label='Constant Weighting')
    axs[1, 0].plot(x_interval, poly_approx_sin_linear, 'r-', label='Linear Weighting')
    axs[1, 0].set_title('|sin(x)| Approximation with Different Weightings')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot |sin(x)| errors
    axs[1, 1].plot(x_interval, poly_approx_sin_const - original_sin, 'g-', label='Constant Weighting Error')
    axs[1, 1].plot(x_interval, poly_approx_sin_linear - original_sin, 'r-', label='Linear Weighting Error')
    axs[1, 1].set_title('|sin(x)| Approximation Errors')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('weighting_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_exp_approximations()
    plot_abs_sin_approximations()
    compare_weightings()