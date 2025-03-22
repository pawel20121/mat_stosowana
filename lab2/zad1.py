import numpy as np
import matplotlib.pyplot as plt
'''
#wykres1
# Parametry modelu
r = 10  # Wskaźnik wzrostu populacji
K = 1.0e+07  # Pojemność środowiska
x0 = 5.0e+06  # Początkowa populacja
t_max = 1  # Czas symulacji
dt = 0.001  # Zwiększamy krok czasowy na 1
'''
#wykres2
# Parametry modelu
r = 10  # Wskaźnik wzrostu populacji
K = 1.0e+07  # Pojemność środowiska
x0 = 1.5e+07  # Początkowa populacja
t_max = 1  # Czas symulacji
dt = 0.001  # Zwiększamy krok czasowy na 1
# Zakres wartości E (wysiłek zbiorów)
E_values = [0, 1, 2, 3,4,5,6,7,8]  # Wartości E do przetestowania

# Funkcja do rozwiązania numerycznego metodą Eulera
def euler_method(r, K, E, x0, t_max, dt):
    t_values = np.arange(0, t_max, dt)
    x_values = np.zeros(len(t_values))
    x_values[0] = x0
    
    for i in range(1, len(t_values)):
        dx_dt = r * x_values[i-1] * (1 - x_values[i-1] / K) - E * x_values[i-1]
        x_values[i] = x_values[i-1] + dx_dt * dt
        
        # Warunki ograniczające:
        # Unikamy ujemnych wartości populacji
        if x_values[i] < 0:
            x_values[i] = 0
        # Jeśli populacja przekroczyła pojemność środowiska, ustawiamy ją na K
        elif x_values[i] > K:
            x_values[i] = K
    
    return t_values, x_values

# Funkcja do rozwiązania analitycznego
def analytic_solution(r, K, E, x0, t_values):
    return (K * x0 * (r - E)) / (r * x0 + (r * (K - x0) - E * K) * np.exp(t_values * (E - r)))

# Rysowanie wykresów dla różnych wartości E
plt.figure(figsize=(12, 8))

for E in E_values:
    t_values, x_values_num = euler_method(r, K, E, x0, t_max, dt)
    x_values_analytic = analytic_solution(r, K, E, x0, t_values)
    
    # Porównanie wykresów: numeryczne i analityczne
    plt.plot(t_values, x_values_num, label=f'Numeryczne (E = {E})', linestyle='--')
    plt.plot(t_values, x_values_analytic, label=f'Analityczne (E = {E})', linestyle='-')

plt.title('Porównanie rozwiązania numerycznego i analitycznego dla różnych wartości E')
plt.xlabel('Czas (t)')
plt.ylabel('Populacja (x)')
plt.legend()
plt.grid(True)
plt.xlim([0, 1])  # Oś X (czas) ma zakres od 0 do 10^6
plt.show()
