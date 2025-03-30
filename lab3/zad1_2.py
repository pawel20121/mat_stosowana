import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parametry
gamma = 0.0  # Współczynnik tłumienia (możesz zmieniać ten parametr)
omega_0 = 5.0  # Częstotliwość własna
omega = 7.0  # Częstotliwość wymuszenia
f = 1.0  # Amplituda wymuszenia
dt = 0.01  # Krok czasowy
T = 10  # Całkowity czas
N = int(T/dt)  # Liczba kroków
m = 1  # Masa (założenie m = 1)

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

# Rozwiązanie analityczne dla tłumienia gamma (zakładając ogólne rozwiązanie)
x_analytic = (f / m) *eg np.cos(omega * t_values) / (omega_0**2 - omega**2 + gamma**2)

# Wykres porównawczy
plt.figure(figsize=(10, 5))
plt.plot(t_values, x_numeric, label="Rozwiązanie numeryczne", linestyle="--", color="r")
plt.plot(t_values, x_analytic, label="Rozwiązanie analityczne", linestyle="-", color="b")
plt.xlabel("Czas t")
plt.xlim([0, 20])
plt.ylabel("Przemieszczenie x")
plt.title(f"Porównanie rozwiązań numerycznego i analitycznego (gamma = {gamma})")
plt.legend()
plt.grid()
plt.show()
