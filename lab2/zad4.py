# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 10:58:50 2025

@author: mozgo
"""

import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    fractal = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            fractal[i, j] = mandelbrot(complex(x[j], y[i]), max_iter)
    
    return fractal

def plot_mandelbrot(fractal):
    plt.figure(figsize=(10, 10))
    plt.imshow(fractal, extent=(-2, 1, -1.5, 1.5), cmap='hot', interpolation='bilinear')
    plt.colorbar()
    plt.title("Mandelbrot Set")
    plt.show()

# Ustawienia zakresu i rozdzielczości
xmin, xmax, ymin, ymax = -2, 1, -1.5, 1.5
width, height = 3840, 2160
max_iter = 10000

# Obliczanie i rysowanie fraktala Mandelbrota
fractal = compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
plot_mandelbrot(fractal)