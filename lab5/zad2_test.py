# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 21:49:19 2025

@author: mozgo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.gridspec as gridspec

def generate_gaussian_noise(length, mean=0, std=1.0):
    """Generate Gaussian white noise"""
    return np.random.normal(mean, std, length)

def apply_filter(input_signal, impulse_response):
    """Apply a filter to an input signal using convolution"""
    return np.convolve(input_signal, impulse_response, mode='valid')

def estimate_impulse_response(input_signal, output_signal, filter_length):
    """
    Estimate impulse response using cross-correlation method
    (Wiener-Hopf equation solution using numpy's lstsq)
    """
    # Make sure we have enough input samples for each output sample
    usable_output_length = len(output_signal)
    
    # Construct the Toeplitz matrix for the input
    X = np.zeros((usable_output_length, filter_length))
    
    # Make sure we don't go out of bounds
    for i in range(usable_output_length):
        # Check if we have enough input samples left
        if i + filter_length <= len(input_signal):
            X[i, :] = input_signal[i:i+filter_length]
        else:
            # If we're near the end, adjust the usable output length
            usable_output_length = i
            X = X[:usable_output_length, :]
            break
    
    # Use only the portion of output that corresponds to complete input segments
    output_used = output_signal[:usable_output_length]
    
    # Solve the least squares problem
    h_est, residuals, rank, s = np.linalg.lstsq(X, output_used, rcond=None)
    
    return h_est

def run_experiment(true_impulse_response, filter_lengths, input_length=1000, noise_snr=None):
    """
    Run an experiment with different filter lengths
    noise_snr: Signal-to-Noise ratio in dB (None means no added noise)
    """
    # Generate input signal (Gaussian noise)
    # Add extra samples to ensure enough data for filtering with various lengths
    input_signal = generate_gaussian_noise(input_length + len(true_impulse_response) + max(filter_lengths))
    
    # Apply the true filter to get the output
    clean_output = apply_filter(input_signal, true_impulse_response)
    
    # Add noise to the output if specified
    if noise_snr is not None:
        # Calculate the power of the signal
        signal_power = np.mean(clean_output**2)
        
        # Calculate the noise power based on SNR
        noise_power = signal_power / (10**(noise_snr/10))
        
        # Generate and add noise
        output_noise = generate_gaussian_noise(len(clean_output), std=np.sqrt(noise_power))
        output_signal = clean_output + output_noise
    else:
        output_signal = clean_output
    
    # Estimate impulse response for each filter length
    results = []
    for length in filter_lengths:
        h_est = estimate_impulse_response(input_signal, output_signal, length)
        
        # Calculate MSE
        if length <= len(true_impulse_response):
            mse = np.mean((true_impulse_response[:length] - h_est)**2)
        else:
            # Pad true impulse response with zeros for comparison
            h_true_padded = np.pad(true_impulse_response, (0, length - len(true_impulse_response)))
            mse = np.mean((h_true_padded - h_est)**2)
        
        results.append((length, h_est, mse))
    
    return results, input_signal[:100], output_signal[:100]  # Return just a portion for visualization

def plot_results(true_impulse, results, input_signal, output_signal, noise_condition):
    """Plot the results of the experiment"""
    plt.figure(figsize=(15, 14))
    gs = gridspec.GridSpec(3, 2)
    
    # Plot the true impulse response
    ax1 = plt.subplot(gs[0, 0])
    ax1.stem(true_impulse, linefmt='b-', markerfmt='bo', basefmt='r-')
    ax1.set_title('True Impulse Response')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    # Plot input and output signals
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(input_signal, label='Input (first 100 samples)')
    ax2.plot(output_signal, label='Output (first 100 samples)')
    ax2.set_title(f'Input and Output Signals - {noise_condition}')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True)
    
    # Plot estimated impulse responses
    ax3 = plt.subplot(gs[1, :])
    
    # Select a subset of results to display
    plot_indices = np.linspace(0, len(results)-1, min(5, len(results)), dtype=int)
    
    true_length = len(true_impulse)
    max_length = max(len(r[1]) for r in results)
    x_range = np.arange(max_length)
    
    # Plot the true impulse response (padded if necessary)
    if true_length < max_length:
        padded_true = np.pad(true_impulse, (0, max_length - true_length))
        ax3.stem(x_range, padded_true, linefmt='k-', markerfmt='ko', basefmt='r-', 
                label='True IR (padded)')
    else:
        ax3.stem(x_range[:true_length], true_impulse, linefmt='k-', markerfmt='ko', 
                basefmt='r-', label='True IR')
        
    # Plot estimated impulse responses
    for idx in plot_indices:
        length, h_est, mse = results[idx]
        # Pad if necessary for visualization
        if len(h_est) < max_length:
            h_est = np.pad(h_est, (0, max_length - len(h_est)))
        ax3.plot(x_range, h_est, '-', label=f'Est. IR (length={length}, MSE={mse:.6f})')
    
    ax3.set_title('Estimated Impulse Responses for Different Filter Lengths')
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(True)
    
    # Plot MSE vs filter length
    ax4 = plt.subplot(gs[2, :])
    filter_lengths = [r[0] for r in results]
    mses = [r[2] for r in results]
    ax4.plot(filter_lengths, mses, 'o-')
    ax4.set_title('MSE vs. Filter Length')
    ax4.set_xlabel('Assumed Filter Length')
    ax4.set_ylabel('Mean Squared Error')
    ax4.axvline(x=len(true_impulse), color='r', linestyle='--', 
                label=f'True Length ({len(true_impulse)})')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.suptitle(f'LTI System Identification - {noise_condition}', fontsize=16)
    plt.subplots_adjust(top=0.94)
    plt.show()

# Define a system with a known impulse response (e.g., a simple low-pass filter)
def create_lowpass_impulse_response(cutoff=0.2, length=20):
    """Create a low-pass filter impulse response"""
    n = np.arange(length)
    h = np.sinc(2 * cutoff * (n - (length - 1) / 2))
    # Apply Hamming window for better frequency response
    h *= np.hamming(length)
    # Normalize to unit energy
    h /= np.sum(h)
    return h

# Main experiment
if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    
    # Create a true impulse response (low-pass filter)
    true_impulse_response = create_lowpass_impulse_response(cutoff=0.2, length=30)
    
    # Define filter lengths to test
    filter_lengths = np.arange(10, 60, 5)
    
    # Run experiment with clean output
    results_clean, input_signal_clean, output_clean = run_experiment(
        true_impulse_response, filter_lengths)
    
    # Plot clean results
    plot_results(true_impulse_response, results_clean, input_signal_clean, 
                output_clean, "Clean Output")
    
    # Run experiment with noisy output (SNR = 20dB)
    results_noisy, input_signal_noisy, output_noisy = run_experiment(
        true_impulse_response, filter_lengths, noise_snr=20)
    
    # Plot noisy results
    plot_results(true_impulse_response, results_noisy, input_signal_noisy, 
                output_noisy, "Noisy Output (SNR=20dB)")
    
    # Run experiment with very noisy output (SNR = 10dB)
    results_very_noisy, input_signal_very_noisy, output_very_noisy = run_experiment(
        true_impulse_response, filter_lengths, noise_snr=10)
    
    # Plot very noisy results
    plot_results(true_impulse_response, results_very_noisy, input_signal_very_noisy, 
                output_very_noisy, "Noisy Output (SNR=10dB)")
    
    # Analyze how filter length estimation varies with noise
    print("==== Effect of Filter Length on Estimation Error ====")
    print("Filter Length | Clean MSE    | SNR=20dB MSE | SNR=10dB MSE")
    print("----------------------------------------------------------------")
    for i, length in enumerate(filter_lengths):
        clean_mse = results_clean[i][2]
        noisy_mse = results_noisy[i][2]
        very_noisy_mse = results_very_noisy[i][2]
        print(f"{length:12d} | {clean_mse:.8f} | {noisy_mse:.8f} | {very_noisy_mse:.8f}")
    
    # Find optimal filter lengths
    optimal_clean = filter_lengths[np.argmin([r[2] for r in results_clean])]
    optimal_noisy = filter_lengths[np.argmin([r[2] for r in results_noisy])]
    optimal_very_noisy = filter_lengths[np.argmin([r[2] for r in results_very_noisy])]
    
    print("\n==== Optimal Filter Lengths ====")
    print(f"True impulse response length: {len(true_impulse_response)}")
    print(f"Optimal filter length (Clean): {optimal_clean}")
    print(f"Optimal filter length (SNR=20dB): {optimal_noisy}")
    print(f"Optimal filter length (SNR=10dB): {optimal_very_noisy}")