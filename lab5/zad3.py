# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 11:17:36 2025

@author: mozgo
"""

import numpy as np
import matplotlib.pyplot as plt

# Parametry
M = 4  # Długość filtru (rząd)
omega_cutoff = np.pi / 2  # Częstotliwość odcięcia
N = 100  # Liczba punktów w zakresie częstotliwości
omega = np.linspace(0, np.pi, N)  # Zakres częstotliwości (od 0 do pi)

# Funkcja wagowa (domyślnie równa 1 w całym zakresie)
S = np.ones(N)

# Pożądana odpowiedź częstotliwościowa (filtr dolnoprzepustowy)
H = np.zeros(N)
for k in range(N):
    if omega[k] < omega_cutoff:
        H[k] = 1  # Pasmo przepustowe
    else:
        H[k] = 0  # Pasmo zaporowe

# Konstrukcja macierzy A (cosinusowe człony)
A = np.zeros((N, M+1))
for n in range(M+1):
    A[:, n] = np.cos(omega * n)

# Rozwiązanie dla c przy użyciu wzoru WLS
S_diag = np.diag(S)  # Diagonalna macierz funkcji wagowej
c = np.linalg.inv(A.T @ S_diag.T @ S_diag @ A) @ (A.T @ S_diag.T @ S_diag @ H)

# Wyświetlenie współczynników filtru
print("Współczynniki filtru:")
print(c)

# Odpowiedź częstotliwościowa zaprojektowanego filtru
H_design = A @ c

# Wykres odpowiedzi częstotliwościowej
plt.figure()
plt.plot(omega, np.abs(H), 'k--', label='Pożądana', linewidth=2)
plt.plot(omega, np.abs(H_design), 'b-', label='Zaprojektowana', linewidth=2)
plt.title('Odpowiedź częstotliwościowa filtru FIR')
plt.xlabel('Częstotliwość (radiany)')
plt.ylabel('Magnituda')
plt.legend()
plt.grid(True)
plt.show()
