# -*- coding: utf-8 -*-
# """
# Created on Mon Mar 24 11:09:34 2025

# @author: mozgo
# """

# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# gamma = 0.5  # Damping coefficient
# omega_0 = 1.0  # Natural frequency
# omega = 1.0  # Forcing frequency
# f = 1.0  # Forcing amplitude
# dt = 0.01  # Time step
# T = 50  # Total time
# N = int(T/dt)  # Number of steps

# # Initial conditions
# x = np.zeros(N)
# u = np.zeros(N)
# x[0] = 1.0  # Initial position
# u[0] = 0.0  # Initial velocity

# # Time integration using Euler's method
# t_values = np.linspace(0, T, N)
# for n in range(1, N):
#     t_n = t_values[n-1]
#     x[n] = x[n-1] + dt * u[n-1]
#     u[n] = u[n-1] + dt * (f * np.cos(omega * t_n) - (gamma / omega) * u[n-1] - omega_0**2 * x[n])

# # Plot results
# plt.figure(figsize=(10, 5))
# plt.plot(t_values, x, label='x(t)', color='b')
# plt.xlabel('Time t')
# plt.ylabel('Displacement x')
# plt.title('Numerical Solution of Forced Damped Harmonic Oscillator')
# plt.legend()
# plt.grid()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint

# # Parameters
# gamma = 0  # Damping coefficient
# omega_0 = 5.0  # Natural frequency
# omega = 7.0  # Forcing frequency
# f = 1.0  # Forcing amplitude
# dt = 0.001  # Time step
# T = 50  # Total time
# N = int(T/dt)  # Number of steps

# # Initial conditions
# x0 = 0.0
# u0 = 0.0

# # Define the system of ODEs
# def forced_damped_oscillator(y, t, gamma, omega_0, omega, f):
#     x, u = y
#     dxdt = u
#     dudt = f * np.cos(omega * t) - (gamma / omega) * u - omega_0**2 * x
#     return [dxdt, dudt]

# # Time array
# t_values = np.linspace(0, T, N)

# # Solve numerically using odeint
# sol = odeint(forced_damped_oscillator, [x0, u0], t_values, args=(gamma, omega_0, omega, f))
# x_numeric = sol[:, 0]

# # Compute the analytical solution
# if np.isclose(omega, omega_0):
#     delta = np.pi / 2  # Set to 90 degrees at resonance
# else:
#     delta = np.arctan((gamma * omega) / (omega_0**2 - omega**2))
# A = f / np.sqrt((omega_0**2 - omega**2)**2 + (gamma * omega)**2)

# x_particular = A * np.cos(omega * t_values - delta)
# x_homogeneous = np.exp(-gamma * t_values / (2 * omega)) * (
#     x0 * np.cos(np.sqrt(omega_0**2 - (gamma**2 / (4 * omega**2))) * t_values) +
#     (u0 + (gamma / (2 * omega)) * x0) * np.sin(np.sqrt(omega_0**2 - (gamma**2 / (4 * omega**2))) * t_values)
# )

# x_analytic = x_particular + x_homogeneous
# xSol(t) = (exp(-t*w0*1j)*(u0 - w0*x0*1j)*1j)/(2*w0) - (exp(t*w0*1j)*(u0 + w0*x0*1j)*1j)/(2*w0)
# # Compute error
# error = np.sum(np.abs(x_numeric - x_analytic))
# print(f"x₀={x0}, u₀={u0}, f={f}, Γ={gamma}, ω={omega}, ω₀={omega_0}, dt={dt}")
# # Plot results
# plt.figure(figsize=(10, 5))
# plt.plot(t_values, x_numeric, label="Numerical Solution", linestyle="--", color="r")
# plt.plot(t_values, x_analytic, label="Analytical Solution", linestyle="-", color="b")
# plt.xlabel("Time t")
# plt.xlim([0, 20])
# plt.ylabel("Displacement x")
# plt.title(f"Comparison of Numerical and Analytical Solutions (Error = {error:.4f})")
# plt.legend()
# plt.grid()
# plt.show()

# omega_ratios = np.linspace(0.5, 1.5, 100)
# amplitudes = []

# for omega in omega_ratios * omega_0:
#     A = f / np.sqrt((omega_0**2 - omega**2)**2 + (gamma * omega)**2)
#     amplitudes.append(A)

# plt.figure(figsize=(8, 5))
# plt.plot(omega_ratios, amplitudes, color="g")
# plt.xlabel(r"Relative Frequency $\omega/\omega_0$")
# plt.ylabel("Steady-State Amplitude")
# plt.title("Amplitude vs. Relative Frequency")
# plt.grid()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sympy import symbols, Function, Eq, dsolve, cos, sin, exp, sqrt, atan, lambdify

# Parametry
gamma = 0  # Współczynnik tłumienia
omega_0 = 5.0  # Częstotliwość własna
omega = 7.0  # Częstotliwość wymuszenia
f = 1.0  # Amplituda wymuszenia
dt = 0.001  # Krok czasowy
T = 50  # Całkowity czas
N = int(T/dt)  # Liczba kroków

# Warunki początkowe
x0 = 0.0
u0 = 0.0

# Definicja układu równań różniczkowych
def forced_damped_oscillator(y, t, gamma, omega_0, omega, f):
    x, u = y
    dxdt = u
    dudt = f * np.cos(omega * t) - gamma * u - omega_0**2 * x
    return [dxdt, dudt]

# Tablica czasu
t_values = np.linspace(0, T, N)

# Rozwiązanie numeryczne za pomocą odeint
sol = odeint(forced_damped_oscillator, [x0, u0], t_values, args=(gamma, omega_0, omega, f))
x_numeric = sol[:, 0]

# Obliczenie rozwiązania analitycznego
t = symbols('t', real=True)
X = Function('X')(t)
X0, U0, F, Gamma, Omega, Omega0 = symbols('X0 U0 F Gamma Omega Omega0', real=True)

# Równanie różniczkowe
ode = Eq(X.diff(t, t) + Gamma * X.diff(t) + Omega0**2 * X, F * cos(Omega * t))

# Rozwiązanie ogólne
x_homogeneous = exp(-Gamma * t / 2) * (X0 * cos(sqrt(Omega0**2 - (Gamma**2 / 4)) * t) +
    (U0 + (Gamma / 2) * X0) * sin(sqrt(Omega0**2 - (Gamma**2 / 4)) * t))

# Rozwiązanie szczególne
if Omega == Omega0:
    delta = np.pi / 2  # Ustawienie na 90 stopni w rezonansie
else:
    delta = atan((Gamma * Omega) / (Omega0**2 - Omega**2))
A = F / sqrt((Omega0**2 - Omega**2)**2 + (Gamma * Omega)**2)
x_particular = A * cos(Omega * t - delta)

# Pełne rozwiązanie analityczne
x_analytic_expr = x_homogeneous + x_particular

# Funkcja lambdify do obliczeń numerycznych
x_analytic_func = lambdify((t, X0, U0, F, Gamma, Omega, Omega0), x_analytic_expr, 'numpy')

# Obliczenie rozwiązania analitycznego dla zadanych parametrów
x_analytic = x_analytic_func(t_values, x0, u0, f, gamma, omega, omega_0)

# Obliczenie błędu
error = np.sum(np.abs(x_numeric - x_analytic))
print(f"x₀={x0}, u₀={u0}, f={f}, γ={gamma}, ω={omega}, ω₀={omega_0}, dt={dt}")

# Wykres porównawczy
plt.figure(figsize=(10, 5))
plt.plot(t_values, x_numeric, label="Rozwiązanie numeryczne", linestyle="--", color="r")
plt.plot(t_values, x_analytic, label="Rozwiązanie analityczne", linestyle="-", color="b")
plt.xlabel("Czas t")
plt.xlim([0, 20])
plt.ylabel("Przemieszczenie x")
plt.title(f"Porównanie rozwiązań numerycznego i analitycznego (Błąd = {error:.4f})")
plt.legend()
plt.grid()
plt.show()

# # Wykres amplitudy w zależności od częstotliwości
# omega_ratios = np.linspace(0.5, 1.5, 100)
# amplitudes = []

# for omega_ratio in omega_ratios:
#     omega = omega_ratio * omega_0
#     A = f / np.sqrt((omega_0**2 - omega**2)**2 + (gamma * omega)**2)
#     amplitudes.append(A)

# plt.figure(figsize=(8, 5))
# plt.plot(omega_ratios, amplitudes, color="g")
# plt.xlabel(r"Stosunek częstotliwości $\omega/\omega_0$")
# plt.ylabel("Amplituda stanu ustalonego")
# plt.title("Amplituda w zależności od stosunku częstotliwości")
# plt.grid()
# plt.show()
