# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:21:36 2025

@author: mozgo
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

value = np.mean([10, 8, 3, 1])
print(value)

# Dane z 24 czujników temperatury
temperatures = np.array([-12.237, -9.712, -9.218, -7.235, -6.455, -4.869, -4.842, -4.407,
                         -3.460, -2.527, -1.764, -1.711, -0.613, 0.252, 0.363, 1.193, 1.720, 2.185,
                         3.379, 5.496, 6.511, 8.722, 10.292, 19.126])
# a) Wyznaczenie średniej i odchylenia standardowego za pomocą NumPy
mean_numpy = np.mean(temperatures)
std_numpy = np.std(temperatures, ddof=0)  # ddof=0 dla odchylenia standardowego próby
n = len(temperatures)
# b) Wyznaczenie średniej i odchylenia standardowego ręcznie
mean_manual = sum(temperatures) / len(temperatures)
variance_manual = sum((x - mean_manual) ** 2 for x in temperatures) / (len(temperatures) - 1)
std_manual = variance_manual ** 0.5

# Wyświetlenie wyników
print(f"Średnia temperatura (NumPy): {mean_numpy:.2f}°C")
print(f"Odchylenie standardowe (NumPy): {std_numpy:.2f}°C")
print(f"Średnia temperatura (ręcznie): {mean_manual:.2f}°C")
print(f"Odchylenie standardowe (ręcznie): {std_manual:.2f}°C")

# Tworzenie histogramu
temp_bins = 10  # Liczba przedziałów histogramu można urzyć 1000
plt.hist(temperatures, bins=temp_bins, alpha=0.7, color='blue', edgecolor='black', label='Dane')

# Dodanie linii dla mx i mx ± 3sx
plt.axvline(mean_numpy, color='red', linestyle='dashed', linewidth=2, label='Średnia (mx)')
plt.axvline(mean_numpy - 3 * std_numpy, color='green', linestyle='dashed', linewidth=2, label='mx - 3sx')
plt.axvline(mean_numpy + 3 * std_numpy, color='green', linestyle='dashed', linewidth=2, label='mx + 3sx')

# Opisy osi i tytuł
plt.xlabel("Temperatura [mK]")
plt.ylabel("Liczność")
plt.title("Histogram temperatury czujników")
plt.legend()
plt.grid(True)

# Wyświetlenie wykresu
plt.show()


#ZADDD2
# Generowanie 10k próbek z rozkładu normalnego N(mx, sx)
norm_samples = np.random.normal(mean_numpy, std_numpy, 10000)

# Tworzenie histogramu dla próbek
plt.hist(norm_samples, bins=temp_bins, alpha=0.7, color='purple', edgecolor='black', label='Próbki N(mx, sx)')

# Dodanie linii dla mx i mx ± 3sx
plt.axvline(mean_numpy, color='red', linestyle='dashed', linewidth=2, label='Średnia (mx)')
plt.axvline(mean_numpy - 3 * std_numpy, color='green', linestyle='dashed', linewidth=2, label='mx - 3sx')
plt.axvline(mean_numpy + 3 * std_numpy, color='green', linestyle='dashed', linewidth=2, label='mx + 3sx')

# Opisy osi i tytuł
plt.xlabel("Temperatura [mK]")
plt.ylabel("Liczność")
plt.title("Histogram 10k próbek z rozkładu N(mx, sx)")
plt.legend()
plt.grid(True)

# Wyświetlenie wykresu
plt.show()

#//////////////////////////////////////////////////////////////////////////////////////////////////////////

#zad3
# Losowy wybór 24 próbek z wygenerowanych 10k wartości
random_samples = np.random.choice(norm_samples, 24, replace=False)
mean_random = np.mean(random_samples)
std_random = np.std(random_samples, ddof=1)

# Tworzenie histogramu dla losowo wybranych 24 próbek
plt.hist(random_samples, bins=temp_bins, alpha=0.7, color='orange', edgecolor='black', label='Losowe próbki (n=24)')

# Dodanie linii dla mx i mx ± 3sx
plt.axvline(mean_random, color='red', linestyle='dashed', linewidth=2, label='Średnia (mx)')
plt.axvline(mean_random - 3 * std_random, color='green', linestyle='dashed', linewidth=2, label='mx - 3sx')
plt.axvline(mean_random + 3 * std_random, color='green', linestyle='dashed', linewidth=2, label='mx + 3sx')

# Opisy osi i tytuł
plt.xlabel("Temperatura [mK]")
plt.ylabel("Liczność")
plt.title("Histogram losowych 24 próbek z N(mx, sx)")
plt.legend()
plt.grid(True)

# Wyświetlenie wykresu
plt.show()

# Wnioski:
# 1. Średnia i odchylenie standardowe rzeczywistych danych różnią się od wygenerowanych wartości,
#    co wynika z ograniczonego zbioru próbek.
# 2. Histogram 10k próbek z N(mx, sx) pokazuje bardziej gładki rozkład normalny, w przeciwieństwie
#    do rzeczywistych danych.
# 3. Losowo wybrane 24 próbki mogą mieć znaczące odchylenia od pełnej populacji,
#    co pokazuje wpływ wielkości próbki na estymację parametrów rozkładu.
#//////////////////////////////////////////////////////////////////////////////////////////////////////

# Wyznaczenie 95% przedziału ufności dla średniej
alpha = 0.05
t_critical = stats.t.ppf(1 - alpha/2, n - 1)  # Wartość krytyczna dla rozkładu t-Studenta
margin_error_mean = t_critical * (std_numpy / np.sqrt(n))
confidence_interval_mean = (mean_numpy - margin_error_mean, mean_numpy + margin_error_mean)

# Wyznaczenie 95% przedziału ufności dla odchylenia standardowego
chi2_lower = stats.chi2.ppf(alpha/2, n - 1)
chi2_upper = stats.chi2.ppf(1 - alpha/2, n - 1)
confidence_interval_std = (
    np.sqrt((n - 1) * std_numpy**2 / chi2_upper),
    np.sqrt((n - 1) * std_numpy**2 / chi2_lower)
)

print(f"Średnia temperatura (NumPy): {mean_numpy:.2f} mK")
print(f"Odchylenie standardowe (NumPy): {std_numpy:.2f} mK")
print(f"95% Przedział ufności dla średniej: {confidence_interval_mean}")
print(f"95% Przedział ufności dla odchylenia standardowego: {confidence_interval_std}")

# Tworzenie histogramu
temp_bins = 10  # Liczba przedziałów histogramu można urzyć 1000
plt.hist(temperatures, bins=temp_bins, alpha=0.7, color='blue', edgecolor='black', label='Dane')

# Dodanie linii dla mx i mx ± 3sx
plt.axvline(confidence_interval_mean[0], color='red', linestyle='dashed', linewidth=2, label='Średnia (mx)')
plt.axvline(confidence_interval_mean[1], color='red', linestyle='dashed', linewidth=2, label='Średnia (mx)')
plt.axvline(confidence_interval_std[1], color='green', linestyle='dashed', linewidth=2, label='mx - 3sx')
plt.axvline(confidence_interval_std[0], color='green', linestyle='dashed', linewidth=2, label='mx - 3sx')
#plt.axvline(confidence_interval_std[(]2), color='green', linestyle='dashed', linewidth=2, label='mx + 3sx')

# Opisy osi i tytuł
plt.xlabel("Temperatura [mK]")
plt.ylabel("Liczność")
plt.title("Histogram temperatury czujników")
plt.legend()
plt.grid(True)

# Wyświetlenie wykresu
plt.show()


####################################################################################
#zad5
# Parametry do bootstrapu
n_iterations = 10000  # Liczba prób bootstrapowych
bootstrap_medians = []  # Lista na wyniki median z prób bootstrapowych

# Bootstrap - losowanie z powtórzeniem i obliczanie mediany
for _ in range(n_iterations):
    sample = np.random.choice(temperatures, size=len(temperatures), replace=True)
    bootstrap_medians.append(np.median(sample))

# Obliczenie 95% przedziału ufności dla mediany
lower_bound = np.percentile(bootstrap_medians, 2.5)
upper_bound = np.percentile(bootstrap_medians, 97.5)

# Wyświetlenie wyników
print(f"95% przedział ufności dla mediany: [{lower_bound:.2f}, {upper_bound:.2f}] °C")