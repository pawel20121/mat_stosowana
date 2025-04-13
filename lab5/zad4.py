# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 22:04:39 2025

@author: mozgo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def irwls(A, x, p=2, gamma=1.5, max_iter=200, stopeps=1e-7):
    pk = 2  # starting value of p
    
    # Find an initial LS solution
    c = np.linalg.lstsq(A, x, rcond=None)[0]
    xhat = A @ c
    
    for k in range(max_iter):
        pk = min(p, gamma * pk)  # p for this iteration
        e = x - xhat  # estimation error
        s = np.abs(e) ** ((pk - 2) / 2)  # new weights
        
        # Handle zero weights to avoid division by zero
        mask = s == 0
        if np.any(mask):
            s[mask] = 1e-10
            
        WA = np.diag(s) @ A  # weighted matrix
        weighted_x = s * x   # weighted vector
        
        # Weighted least-squares solution
        chat = np.linalg.lstsq(WA, weighted_x, rcond=None)[0]
        
        lambda_val = 1 / (pk - 1)
        cnew = lambda_val * chat + (1 - lambda_val) * c
        
        if np.linalg.norm(c - cnew) < stopeps:
            c = cnew
            print(f"Converged at iteration {k}, p={pk}")
            break
            
        c = cnew
        xhat = A @ c
    
    return c, s

def design_p_norm_fir(N, cutoff, p=2, trans_width=0.1):
    if N % 2 == 1:
        N += 1  # Ensure N is even for Type I filter
    
    # Number of frequency points
    num_freq = 512
    
    # Frequency grid from 0 to π
    omega = np.linspace(0, np.pi, num_freq)
    
    # Define desired frequency response (ideal low-pass)
    D = np.zeros(num_freq)
    
    # Passband
    D[omega <= cutoff - trans_width/2] = 1
    
    # Transition band - linearly decreasing
    trans_indices = np.logical_and(omega > cutoff - trans_width/2, omega < cutoff + trans_width/2)
    trans_omega = omega[trans_indices]
    D[trans_indices] = 0.5 + 0.5 * np.cos(np.pi * (trans_omega - cutoff) / trans_width)
    
    # Stopband
    D[omega >= cutoff + trans_width/2] = 0
    
    # Create the A matrix for the frequency response
    L = (N // 2) + 1  # Number of unique coefficients for linear-phase
    A = np.zeros((num_freq, L))
    
    for i in range(num_freq):
        for n in range(L):
            if n == 0:
                A[i, n] = 1
            else:
                A[i, n] = 2 * np.cos(omega[i] * n)
    
    # Solve using IRLS
    c, weights = irwls(A, D, p)
    
    # Construct the filter coefficients
    h = np.zeros(N+1)
    h[N//2] = c[0]  # Center coefficient
    
    for n in range(1, L):
        h[N//2 + n] = c[n]
        h[N//2 - n] = c[n]  # Symmetry for linear phase
    
    return h, D, omega, weights

def compute_frequency_response(h, num_points=512):
    """
    Compute frequency response of the filter
    """
    w, H = signal.freqz(h, worN=num_points)
    return w, H

def plot_filter_characteristics(h, title="Filter Characteristics"):
    """
    Plot filter characteristics including impulse response and frequency response
    """
    # Compute frequency response
    w, H = signal.freqz(h, worN=1000)
    w_normalized = w / np.pi
    
    # Compute magnitude in dB and phase
    mag = 20 * np.log10(np.abs(H))
    phase = np.unwrap(np.angle(H))
    
    # Create figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot impulse response
    axs[0].stem(np.arange(len(h)), h)
    axs[0].set_title(f"Impulse Response - {title}")
    axs[0].set_xlabel("Sample (n)")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)
    
    # Plot magnitude response
    axs[1].plot(w_normalized, mag)
    axs[1].set_title("Magnitude Response")
    axs[1].set_xlabel("Normalized Frequency (×π rad/sample)")
    axs[1].set_ylabel("Magnitude (dB)")
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(-80, 5)
    axs[1].grid(True)
    
    # Add vertical line at cutoff frequency
    axs[1].axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
    axs[1].text(0.51, 0, "Cutoff", color='r', alpha=0.5)
    
    # Plot phase response
    axs[2].plot(w_normalized, phase)
    axs[2].set_title("Phase Response")
    axs[2].set_xlabel("Normalized Frequency (×π rad/sample)")
    axs[2].set_ylabel("Phase (radians)")
    axs[2].set_xlim(0, 1)
    axs[2].grid(True)
    
    plt.tight_layout()
    return fig

def compare_transition_widths(N, cutoff, p=30):
    """
    Compare different transition widths for fixed p-norm
    """
    transition_widths = [0.05, 0.1, 0.2]
    colors = ['blue', 'green', 'red']
    
    plt.figure(figsize=(10, 6))
    
    for i, tw in enumerate(transition_widths):
        h, _, _, _ = design_p_norm_fir(N, cutoff, p, tw)
        w, H = compute_frequency_response(h)
        mag = 20 * np.log10(np.abs(H))
        
        plt.plot(w/np.pi, mag, color=colors[i], label=f"TW = {tw}")
    
    plt.title(f"Magnitude Response for Different Transition Widths (p={p})")
    plt.xlabel("Normalized Frequency (×π rad/sample)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(-80, 5)
    
    # Add vertical line at cutoff frequency
    plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.5)
    plt.text(0.51, 0, "Cutoff", color='k', alpha=0.5)
    
    plt.tight_layout()
    return plt.gcf()

def compare_p_norms(N, cutoff, trans_width=0.1):
    """
    Compare different p-norm values for fixed transition width
    """
    p_values = [2, 8, 30]
    colors = ['blue', 'green', 'red']
    
    plt.figure(figsize=(10, 6))
    
    for i, p_val in enumerate(p_values):
        h, _, _, _ = design_p_norm_fir(N, cutoff, p_val, trans_width)
        w, H = compute_frequency_response(h)
        mag = 20 * np.log10(np.abs(H))
        
        p_label = "p = 2 (LS)" if p_val == 2 else f"p = {p_val}"
        plt.plot(w/np.pi, mag, color=colors[i], label=p_label)
    
    plt.title(f"Magnitude Response for Different P-norm Values (TW={trans_width})")
    plt.xlabel("Normalized Frequency (×π rad/sample)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(-80, 5)
    
    # Add vertical line at cutoff frequency
    plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.5)
    plt.text(0.51, 0, "Cutoff", color='k', alpha=0.5)
    
    plt.tight_layout()
    return plt.gcf()

def main():
    # Filter parameters
    N = 50  # Filter order
    cutoff = np.pi/2  # Cutoff frequency at π/2
    p = 30  # Target p-norm for optimization
    trans_width = 0.1  # Default transition width
    
    print(f"Designing low-pass FIR filter with cutoff frequency = {cutoff/np.pi}π")
    print(f"Filter order = {N}, p-norm = {p}")
    
    # Design filter with default parameters
    h, desired, omega, weights = design_p_norm_fir(N, cutoff, p, trans_width)
    
    # Plot filter characteristics
    fig1 = plot_filter_characteristics(h, f"p={p}, TW={trans_width}")
    fig1.savefig(f"filter_characteristics_p{p}_tw{trans_width}.png")
    
    # Compare different transition widths
    fig2 = compare_transition_widths(N, cutoff, p)
    fig2.savefig(f"comparison_transition_widths_p{p}.png")
    
    # Compare different p-norm values
    fig3 = compare_p_norms(N, cutoff, trans_width)
    fig3.savefig(f"comparison_p_norms_tw{trans_width}.png")
    
    # Show the filter coefficients
    print("\nFilter coefficients:")
    print(h)
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()