# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:57:43 2025

@author: mozgo
"""

import sympy as sp

# Definicja zmiennych symbolicznych
t, m, G, w0, f, w = sp.symbols('t m G w0 f w')
x = sp.Function('x')(t)

# Definicja pochodnych
Dx = sp.diff(x, t)
D2x = sp.diff(x, t, 2)

# Równanie różniczkowe
ode = sp.Eq(D2x + (G/m)*Dx + w0**2*x, f*sp.cos(w*t))

# Warunki początkowe
cond1 = sp.Eq(x.subs(t, 0), 0)  # Warunek początkowy dla przemieszczenia
cond2 = sp.Eq(Dx.subs(t, 0), 0)  # Warunek początkowy dla prędkości

# Rozwiązanie ODE
xSol = sp.dsolve(ode, ics={x.subs(t, 0): 0, Dx.subs(t, 0): 0})

# Uproszczenie rozwiązania
simplified_xSol = sp.simplify(xSol)

# Wyświetlenie wyniku
print(simplified_xSol)