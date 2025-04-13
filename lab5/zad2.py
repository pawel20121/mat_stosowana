import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Parametry
M = 9                   # Połowa długości filtru
wc = np.pi / 2            # Częstotliwość odcięcia
N = 20                 # Długość sygnału wejściowego
filter_len = 2 * M + 1    # Pełna długość filtru

# Indeksy czasowe dla filtru
n = np.arange(-M, M + 1)

# Odpowiedź impulsowa filtru dolnoprzepustowego h_LP[n]
h_lp = np.sinc(wc * n / np.pi / np.pi)
h_lp[M] = wc / np.pi  # Poprawka dla n = 0 (sinc(0) = 1)

# Sygnał wejściowy - szum Gaussowski o zerowej średniej
f = np.random.normal(0, 1, N)

# Odpowiedź systemu na sygnał wejściowy (konwolucja)
d = np.convolve(f, h_lp, mode='same')

# Dodanie szumu Gaussowskiego do sygnału wyjściowego
noise = np.random.normal(0, 0.1, N)
d_noisy = d + noise

# Budowa macierzy Toeplitza z przesunięć sygnału wejściowego
valid_len = N - filter_len + 1
X = np.array([f[i:i+filter_len] for i in range(valid_len)])

# Dopasowanie długości sygnału wyjściowego do X
d_noisy_valid = d_noisy[M:-M]  # Odcinamy brzegi

# Estymacja odpowiedzi impulsowej: h_hat = (X^T X)^-1 X^T d
h_hat = np.linalg.pinv(X) @ d_noisy_valid

# Wyjście systemu: y_est (na podstawie estymowanego filtru) i y_true (na podstawie prawdziwego)
y_est = np.convolve(f, h_hat, mode='same')
y_true = d  # już obliczone wcześniej jako filtracja f przez h_lp

# Błąd pomiędzy estymowanym i prawdziwym wyjściem
error = y_true - y_est

# Wykresy
plt.figure(figsize=(14, 6))

# Wykres 1: Porównanie wyjść
plt.subplot(1, 2, 1)
plt.plot(y_true, label='True Output d[n]')
plt.plot(y_est, label='Estimated Output $\hat{d}[n]$', linestyle='--')
plt.title('Disturbed Output Comparison')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

# Wykres 2: Błąd estymacji
plt.subplot(1, 2, 2)
plt.plot(error, color='red')
plt.title('Estimation Error: $e[n] = d[n] - \hat{d}[n]$')
plt.xlabel('n')
plt.ylabel('Error Amplitude')
plt.grid()

plt.tight_layout()
plt.show()