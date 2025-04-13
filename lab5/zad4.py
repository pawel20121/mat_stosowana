# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 11:23:04 2025

@author: mozgo
"""

import numpy as np
import matplotlib.pyplot as plt

# Parametry
M = 30  # Rząd filtru (wybrano dużą wartość p)
omega_cutoff = np.pi / 2  # Częstotliwość odcięcia (pi/2)
N = 100  # Liczba punktów w analizie częstotliwości
omega = np.linspace(0, np.pi, N)  # Zakres częstotliwości (od 0 do pi)
p = 30  # P-norma

# Funkcja wagowa dla IRLS
def irls(A, H, S, max_iter=100, tol=1e-6):
    c = np.linalg.lstsq(A, H, rcond=None)[0]  # Początkowe rozwiązanie LS
    for i in range(max_iter):
        # Oblicz resztę
        residual = H - A @ c
        # Oblicz wagę na podstawie reszty (P-norm)
        weights = np.abs(residual) ** (p - 2)  # Waga zależna od reszty i p-norm
        # Nowa funkcja wagowa
        W = np.diag(weights)
        # Zaktualizuj rozwiązanie
        c_new = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ H, rcond=None)[0]
        
        # Sprawdź zbieżność
        if np.linalg.norm(c_new - c) < tol:
            break
        c = c_new
    return c

# Pożądana odpowiedź częstotliwościowa (filtr dolnoprzepustowy)
S = np.ones(N)
H = np.zeros(N)
for k in range(N):
    if omega[k] < omega_cutoff:
        H[k] = 1  # Pasmo przepustowe
    else:
        H[k] = 0  # Pasmo zaporowe

# Konstrukcja macierzy A (cosinusowe człony)
M = 30  # Ustawiamy dużą długość filtru dla p = 30
A = np.zeros((N, M + 1))
for n in range(M + 1):
    A[:, n] = np.cos(omega * n)

# Zastosowanie algorytmu IRLS do obliczenia współczynników filtru
c = irls(A, H, S)

# Odpowiedź częstotliwościowa zaprojektowanego filtru
H_design = A @ c

# Wykres odpowiedzi częstotliwościowej
plt.figure()
plt.plot(omega, np.abs(H), 'k--', label='Pożądana', linewidth=2)
plt.plot(omega, np.abs(H_design), 'b-', label='Zaprojektowana', linewidth=2)
plt.title(f'Odpowiedź częstotliwościowa filtru FIR (p={p})')
plt.xlabel('Częstotliwość (radiany)')
plt.ylabel('Magnituda')
plt.legend()
plt.grid(True)
plt.show()