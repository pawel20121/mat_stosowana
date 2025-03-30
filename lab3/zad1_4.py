# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:40:26 2025

@author: mozgo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:39:19 2025

@author: mozgo
"""

import numpy as np
import matplotlib.pyplot as plt

def euler_method(x0, u0, f, Gamma, omega, omega0, dt, tmax):
    # Initialize arrays
    t = np.arange(0, tmax, dt)
    n = len(t)
    x = np.zeros(n)
    u = np.zeros(n)
    
    # Set initial conditions
    x[0] = x0
    u[0] = u0
    
    # Euler method
    for i in range(1, n):
        x[i] = x[i-1] + dt * u[i-1]
        u[i] = u[i-1] + dt * (f * np.cos(omega * t[i]) - (Gamma/omega) * u[i-1] - omega0**2 * x[i])
    
    return t, x, u

def analytic_solution(x0, u0, f, Gamma, omega, omega0, t):
    # For the undamped case (Gamma = 0)
    if Gamma == 0:
        # Homogeneous solution
        xh = x0 * np.cos(omega0 * t) + (u0/omega0) * np.sin(omega0 * t)
        
        # Particular solution (steady-state)
        if abs(omega - omega0) < 1e-10:  # For resonance case
            xp = (f/(2*omega0)) * t * np.sin(omega0 * t)
        else:
            xp = f * (np.cos(omega * t) - np.cos(omega0 * t)) / (omega0**2 - omega**2)
    else:
        # For the damped case, formula becomes more complex
        # This is a simplified approximation
        gamma = Gamma/omega
        denom = (omega0**2 - omega**2)**2 + (gamma * omega)**2
        A = f / np.sqrt(denom)
        phi = np.arctan2(gamma * omega, omega0**2 - omega**2)
        
        # Homogeneous solution (decaying oscillation)
        xh = np.exp(-gamma*t/2) * (x0 * np.cos(np.sqrt(omega0**2 - (gamma/2)**2) * t) + 
                                  (u0 + gamma*x0/2)/np.sqrt(omega0**2 - (gamma/2)**2) * 
                                  np.sin(np.sqrt(omega0**2 - (gamma/2)**2) * t))
        
        # Particular solution
        xp = A * np.cos(omega * t - phi)
    
    return xh, xp, xh + xp

# Parameters
x0 = 0.0      # Initial position
u0 = 0.0      # Initial velocity
f = 1.0       # Force amplitude
Gamma = 0.0   # Damping coefficient
omega = 7.0   # Driving frequency
omega0 = 5.0  # Natural frequency
dt = 1.0e-3   # Time step
tmax = 20.0   # End time

# Numerical solution
t, x_numeric, u_numeric = euler_method(x0, u0, f, Gamma, omega, omega0, dt, tmax)

# Analytic solution
x_homogeneous, x_particular, x_analytic = analytic_solution(x0, u0, f, Gamma, omega, omega0, t)

# Error calculation
error = np.sum(np.abs(x_numeric - x_analytic)) / len(t)

# Plotting
plt.figure(figsize=(10, 8))

# Plot 1: Numerical vs Analytic Solution
plt.subplot(2, 1, 1)
plt.plot(t, x_analytic, 'b-', label='analytic')
plt.plot(t, x_numeric, 'r--', label=f'numeric, ε = {error:.4f}')
plt.title(f'x₀={x0}, u₀={u0}, f={f}, Γ={Gamma}, ω={omega}, ω₀={omega0}, dt={dt}')
plt.legend()
plt.grid(True)

# Plot 2: Homogeneous vs Particular Solution
plt.subplot(2, 1, 2)
plt.plot(t, x_homogeneous, 'g-', label='homogeneous')
plt.plot(t, x_particular, 'c-', label='particular')
plt.legend()
plt.grid(True)
plt.xlabel('t')

plt.tight_layout()
plt.show()

# Now let's study the amplitude response vs frequency ratio
def amplitude_response(f, Gamma_values, omega0, omega_ratio):
    """Calculate amplitude for various damping and frequency ratios"""
    amplitudes = {}
    
    for Gamma in Gamma_values:
        amp = []
        for ratio in omega_ratio:
            omega = ratio * omega0
            denom = (omega0**2 - omega**2)**2 + (Gamma * omega/omega0)**2
            A = f / np.sqrt(denom)
            amp.append(A)
        amplitudes[Gamma] = amp
    
    return amplitudes

# Calculate amplitude response for different Gamma values
omega_ratio = np.linspace(0.1, 2.0, 100)
Gamma_values = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
amplitudes = amplitude_response(f=2.0, Gamma_values=Gamma_values, 
                               omega0=5.0, omega_ratio=omega_ratio)

# Plot amplitude response
plt.figure(figsize=(10, 6))
for Gamma, amp in amplitudes.items():
    plt.plot(omega_ratio, amp, label=f'Γ={Gamma}')

plt.xlabel('ω/ω₀')
plt.ylabel('Amplitude')
plt.title('x₀=0, u₀=0, f=2, ω₀=5')
plt.legend()
plt.grid(True)
plt.show()
