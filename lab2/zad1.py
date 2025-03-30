import numpy as np
import matplotlib.pyplot as plt
'''
# #wykres1
# # Parametry modelu
# r = 10  # Wskaźnik wzrostu populacji
# K = 1.0e+07  # Pojemność środowiska
# x0 = 5.0e+06  # Początkowa populacja
# t_max = 1  # Czas symulacji
# dt = 0.001  # Zwiększamy krok czasowy na 1
# '''
# #wykres2

# # Parametry modelu
# r = 10  # Wskaźnik wzrostu populacji
# K = 1.0e+07  # Pojemność środowiska
# x0 = 1.5e+07  # Początkowa populacja
# t_max = 1  # Czas symulacji
# dt = 0.001  # Zwiększamy krok czasowy na 1
# # Zakres wartości E (wysiłek zbiorów)
# E_values = [0, 1, 2, 3,4,5,6,7,8]  # Wartości E do przetestowania

# # Funkcja do rozwiązania numerycznego metodą Eulera
# def euler_method(r, K, E, x0, t_max, dt):
#     t_values = np.arange(0, t_max, dt)
#     x_values = np.zeros(len(t_values))
#     x_values[0] = x0
    
#     for i in range(1, len(t_values)):
#         dx_dt = r * x_values[i-1] * (1 - x_values[i-1] / K) - E * x_values[i-1]
#         x_values[i] = x_values[i-1] + dx_dt * dt
        
#         # # Warunki ograniczające:
#         # # Unikamy ujemnych wartości populacji
#         # if x_values[i] < 0:
#         #     x_values[i] = 0
#         # # Jeśli populacja przekroczyła pojemność środowiska, ustawiamy ją na K
#         # elif x_values[i] > K:
#         #     x_values[i] = K
    
#     return t_values, x_values

# # Funkcja do rozwiązania analitycznego
# def analytic_solution(r, K, E, x0, t_values):
#     return (K * x0 * (r - E)) / (r * x0 + (r * (K - x0) - E * K) * np.exp(t_values * (E - r)))

# # Rysowanie wykresów dla różnych wartości E
# plt.figure(figsize=(12, 8))

# for E in E_values:
#     t_values, x_values_num = euler_method(r, K, E, x0, t_max, dt)
#     x_values_analytic = analytic_solution(r, K, E, x0, t_values)
    
#     # Porównanie wykresów: numeryczne i analityczne
#     plt.plot(t_values, x_values_num, label=f'Numeryczne (E = {E})', linestyle='--')
#     plt.plot(t_values, x_values_analytic + 30000, label=f'Analityczne (E = {E})', linestyle='-')

# plt.title('Porównanie rozwiązania numerycznego i analitycznego dla różnych wartości E')
# plt.xlabel('Czas (t)')
# plt.ylabel('Populacja (x)')
# plt.legend()
# plt.grid(True)
# plt.xlim([0, 1])
# plt.show()


# # Parametry do diagramu bifurkacyjnego
# x0 = 1.0e+06  # Początkowa populacja
# K = 1.0e+08  # Pojemność środowiska
# E = 0  # Stała wartość E dla tego przypadku
# t_max = 1  # Czas symulacji
# dt = 0.001  # Krok czasowy
# iterations = int(t_max / dt)  # Liczba iteracji
# # last_iterations = 200  # Ilość ostatnich punktów do rysowania

# # Zakres wartości współczynnika r
# r_values = np.linspace(0, 20, 1000)

# # Listy do przechowywania wyników
# bifurcation_x = []
# bifurcation_r = []

# # # Generowanie diagramu bifurkacyjnego
# # for r in r_values:
# #     t_val = np.arange(0, t_max, dt)
# #     x = np.zeros(len(t_val))
# #     x[0] = x0
# #     for i in range(1, len(t_val)):
# #         # dx_dt = r * x * (1 - x / K) - E * x
# #         dx_dt = r * x[i-1] * (1 - x[i-1] / K) - E * x[i-1]
# #         x[i] = x[i-1] + dx_dt * dt  # Metoda Eulera
        
# #         if i == len(t_val)-1:  # Zbieramy tylko końcowe wartości
# #             bifurcation_x.append(x[i])
# #             bifurcation_r.append(r)


# # for r in r_values:
# #     t_val = np.arange(0, t_max, dt)
# #     x = np.zeros(len(t_val))
# #     x[0] = x0
# #     r_dt = r * dt  # Poprawiona oś X

# #     for i in range(1, len(t_val)):
# #         dx_dt = r * x[i-1] * (1 - x[i-1] / K) - E * x[i-1]
# #         x[i] = x[i-1] + dx_dt * dt  # Metoda Eulera

# #         if i >= (len(t_val) - last_iterations):  # Zbieramy końcowe wartości
# #             bifurcation_x.append(x[i])
# #             bifurcation_r.append(r_dt)
# # # Rysowanie wykresu
# # plt.figure(figsize=(10, 6))
# # # plt.scatter(bifurcation_r, bifurcation_x, s=0.1, color="black", alpha=0.5)
# # plt.scatter(bifurcation_r, bifurcation_x, label=f'', linestyle='-')
# # plt.xlabel("r")
# # plt.ylabel("Populacja (x)")
# # plt.title("Diagram bifurkacyjny dla modelu logistycznego z E = 0")
# # plt.show()
#wykres1
# Parametry modelu
# r = 10  # Wskaźnik wzrostu populacji
# K = 1.0e+07  # Pojemność środowiska
# x0 = 5.0e+06  # Początkowa populacja
# t_max = 1  # Czas symulacji
# dt = 0.001  # Zwiększamy krok czasowy na 1
    
# def bifurcation_diagram(seed, n_skip, n_iter, step=0.001, r_min=0):
#     print("Starting with x0 seed {0}, skip plotting first {1} iterations, then plot next {2} iterations.".format(seed, n_skip, n_iter));
#     # Array of r values, the x axis of the bifurcation plot
#     R = []
#     # Array of x_t values, the y axis of the bifurcation plot
#     X = []
#     K = 1.0e+07
#     E = 0
#     # Create the r values to loop. For each r value we will plot n_iter points
#     r_range = np.linspace(r_min, 4, int(1/step))

#     for r in r_range:
#         x = np.zeros(n_iter + n_skip + 1)
#         x[0] = seed;
#         # For each r, iterate the logistic function and collect datapoint if n_skip iterations have occurred
#         for i in range(1, n_iter + n_skip + 1):
#             dx_dt = r * x[i-1] * (1 - x[i-1] / K) - E * x[i-1]
#             x[i] = x[i-1] + dx_dt * dt
#             if i >= n_skip:
#                 R.append(r)
#                 X.append(x[i])
                

#             # x = logistic_eq(r,x);
#     # Plot the data    
#     plt.plot(R, X, ls='', marker=',')
#     # plt.ylim(0, 125)
#     plt.xlim(r_min, 4)
#     plt.xlabel('r')
#     plt.ylabel('X')
#     plt.show()    
    
# bifurcation_diagram(1.0e+06, 0, int(t_max / dt))    

import numpy as np
import matplotlib.pyplot as plt

# Parametry modelu
K = 1.0e+07  # Pojemność środowiska
E = 0  # Brak zbiorów
dt = 0.001  # Krok czasowy
t_max = 1  # Czas symulacji

def bifurcation_diagram(seed, n_skip, n_iter, step=0.001, r_min=0, r_max=20):
    print(f"Starting with x0 = {seed}, skipping first {n_skip} iterations, then plotting {n_iter} iterations.")

    R = []  # Oś X - wartości r
    X = []  # Oś Y - wartości populacji

    r_range = np.linspace(r_min, r_max, int((r_max - r_min) / step))

    for r in r_range:
        x = np.zeros(n_iter + n_skip + 1)
        x[0] = seed

        for i in range(1, n_iter + n_skip + 1):
            dx_dt = r * x[i-1] * (1 - x[i-1] / K) - E * x[i-1]
            x[i] = x[i-1] + dx_dt * dt
            
            if i >= n_skip:
                R.append(r)
                X.append(x[i])
    
    # Rysowanie wykresu
    plt.figure(figsize=(10, 6))
    plt.scatter(R, X, s=0.1, color="black", alpha=0.5)
    plt.xlabel("r")
    plt.ylabel("Populacja (x)")
    plt.title("Diagram bifurkacyjny dla modelu logistycznego")
    plt.show()

# Wywołanie funkcji
# bifurcation_diagram(1.0e+06, 800, int(t_max / dt), step=0.001, r_min=0, r_max=4)

def logistic_equation_orbit(seed, r, n_iter, n_skip=0):
    print('Orbit for seed {0}, growth rate of {1}, plotting {2} iterations after skipping {3}'.format(seed, r, n_iter, n_skip))
    X_t=[]
    T=[]
    t=0
    x = np.zeros(n_iter + n_skip + 1)
    x[0] = seed
    # Iterate the logistic equation, printing only if n_skip steps have been skipped
    for i in range(n_iter + n_skip):
        if i >= n_skip:
            X_t.append(x)
            T.append(t)
            t+=1
        dx_dt = r * x[i-1] * (1 - x[i-1] / K) - E * x[i-1]
        x[i] = x[i-1] + dx_dt * dt
    # Configure and decorate the plot
    plt.plot(T, X_t)
    plt.ylim(0, 1)
    plt.xlabel('Time t')
    plt.ylabel('X_t')
    plt.show()
# Parametry modelu
K = 1.0e+07  # Pojemność środowiska
E = 500  # Brak zbiorów
dt = 0.001  # Krok czasowy
t_max = 1  # Czas symulacji    
logistic_equation_orbit(1.0e+06, 3.05, 100)