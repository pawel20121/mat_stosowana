import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.gridspec import GridSpec

def ideal_lp(cutoff, N):
    w = np.linspace(0, np.pi, N)
    H_ideal = np.zeros(N)
    H_ideal[w <= cutoff] = 1.0
    return H_ideal, w

def wls_fir_design(N, cutoff, weight_func, transition_width=0.1):
    # Ensure filter length is odd for firls (numtaps = N+1)
    if N % 2 == 0:
        N = N + 1  # Make filter order even so length (N+1) is odd
    
    # Number of frequency points for design
    num_freq_points = 1024
    w = np.linspace(0, np.pi, num_freq_points)
    
    # Ideal response
    H_ideal = np.zeros(num_freq_points)
    H_ideal[w <= cutoff] = 1.0
    
    # Create weight function
    weights = np.ones(num_freq_points)
    
    # Define transition band
    pass_edge = cutoff - transition_width/2
    stop_edge = cutoff + transition_width/2
    
    pass_band = w <= pass_edge
    stop_band = w >= stop_edge
    trans_band = ~(pass_band | stop_band)
    
    # Apply weighting function
    if callable(weight_func):
        weights = weight_func(w, pass_edge, stop_edge)
    elif weight_func == 1:
        # Equal weighting in passband and stopband
        weights[pass_band] = 1.0
        weights[stop_band] = 1.0
        weights[trans_band] = 0.01  # Small weight in transition band
    elif weight_func == 2:
        # Higher weight in stopband
        weights[pass_band] = 1.0
        weights[stop_band] = 10.0
        weights[trans_band] = 0.01
    elif weight_func == 3:
        # Higher weight in passband
        weights[pass_band] = 10.0
        weights[stop_band] = 1.0
        weights[trans_band] = 0.01
    elif weight_func == 4:
        # Custom weights to balance passband and stopband errors
        weights[pass_band] = 5.0
        weights[stop_band] = 5.0
        weights[trans_band] = 0.01
    
    # Design the filter using firls (Least-Squares FIR filter design)
    # Convert to band edges format required by firls
    bands = np.zeros(4)
    bands[0] = 0          # Start of passband
    bands[1] = pass_edge  # End of passband
    bands[2] = stop_edge  # Start of stopband
    bands[3] = np.pi      # End of stopband
    
    desired = np.array([1, 1, 0, 0])  # Desired response at band edges
    
    # Weight values at band edges
    weight_values = np.array([np.sqrt(np.mean(weights[pass_band])), 
                             np.sqrt(np.mean(weights[stop_band]))])
    
    # Design the filter
    h = signal.firls(N, bands/np.pi, desired, weight=weight_values)
    
    # Calculate the frequency response
    w_plot, H = signal.freqz(h, worN=num_freq_points)
    w_plot = w_plot.real
    H = np.abs(H)
    
    return h, w_plot, H

def custom_weight(w, pass_edge, stop_edge):
    """Custom exponential weighting function"""
    weights = np.ones_like(w)
    pass_band = w <= pass_edge
    stop_band = w >= stop_edge
    trans_band = ~(pass_band | stop_band)
    
    weights[pass_band] = 5.0
    weights[stop_band] = 5.0 * np.exp((w[stop_band] - stop_edge))
    weights[trans_band] = 0.01
    
    return weights

def plot_filter_response(h, w, H, H_ideal, weight_name, transition_width):
    """Plot the filter frequency response and impulse response"""
    plt.figure(figsize=(12, 8))
    
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    # Plot frequency response
    ax1 = plt.subplot(gs[0])
    ax1.plot(w/np.pi, H, 'b-', linewidth=2, label=f'WLS Filter (N={len(h)-1})')
    ax1.plot(w/np.pi, H_ideal, 'r--', linewidth=1.5, label='Ideal')
    ax1.set_title(f'WLS FIR LP Filter - Weight: {weight_name}, Transition Width: {transition_width:.2f}π')
    ax1.set_xlabel('Normalized Frequency (×π rad/sample)')
    ax1.set_ylabel('Magnitude')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xlim(0, 1)
    
    # Plot passband detail
    ax2 = plt.subplot(gs[1])
    ax2.plot(w/np.pi, H, 'b-', linewidth=2)
    ax2.plot(w/np.pi, H_ideal, 'r--', linewidth=1.5)
    ax2.set_title('Passband Detail')
    ax2.set_xlabel('Normalized Frequency (×π rad/sample)')
    ax2.set_ylabel('Magnitude')
    ax2.set_xlim(0, 0.5)
    ax2.set_ylim(0.9, 1.1)
    ax2.grid(True)
    
    # Plot stopband detail
    ax3 = plt.subplot(gs[2])
    ax3.plot(w/np.pi, 20*np.log10(np.abs(H) + 1e-10), 'b-', linewidth=2)
    ax3.set_title('Stopband Attenuation')
    ax3.set_xlabel('Normalized Frequency (×π rad/sample)')
    ax3.set_ylabel('Magnitude (dB)')
    ax3.set_xlim(0.5, 1)
    ax3.set_ylim(-80, 0)
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Plot impulse response
    plt.figure(figsize=(10, 4))
    plt.stem(np.arange(len(h)), h)
    plt.title(f'Impulse Response - Weight: {weight_name}, Transition Width: {transition_width:.2f}π')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare_transition_widths(order, cutoff, weight_func, weight_name):
    """Compare different transition band widths"""
    transition_widths = [0.05, 0.1, 0.2, 0.3]
    
    plt.figure(figsize=(12, 8))
    
    # Get ideal response for plotting
    H_ideal, w_ideal = ideal_lp(cutoff, 1024)
    
    for i, tw in enumerate(transition_widths):
        h, w, H = wls_fir_design(order, cutoff, weight_func, transition_width=tw)
        
        plt.subplot(2, 2, i+1)
        plt.plot(w/np.pi, H, 'b-', linewidth=2, label=f'WLS Filter')
        plt.plot(w_ideal/np.pi, H_ideal, 'r--', linewidth=1.5, label='Ideal')
        plt.title(f'Transition Width: {tw:.2f}π')
        plt.xlabel('Normalized Frequency (×π rad/sample)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.legend()
        plt.ylim(-0.1, 1.1)
    
    plt.suptitle(f'Effect of Transition Band Width - Weight: {weight_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    # Plot in dB scale for better stopband visualization
    plt.figure(figsize=(12, 8))
    
    for i, tw in enumerate(transition_widths):
        h, w, H = wls_fir_design(order, cutoff, weight_func, transition_width=tw)
        
        plt.subplot(2, 2, i+1)
        plt.plot(w/np.pi, 20*np.log10(np.abs(H) + 1e-10), 'b-', linewidth=2)
        plt.title(f'Transition Width: {tw:.2f}π')
        plt.xlabel('Normalized Frequency (×π rad/sample)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.ylim(-80, 5)
    
    plt.suptitle(f'Stopband Attenuation vs Transition Width - Weight: {weight_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def compare_weight_functions(order, cutoff, transition_width):
    """Compare different weight functions"""
    weight_funcs = [1, 2, 3, 4, custom_weight]
    weight_names = ['Equal', 'Higher in Stopband', 'Higher in Passband', 'Balanced', 'Custom Exponential']
    
    plt.figure(figsize=(15, 10))
    
    # Get ideal response for plotting
    H_ideal, w_ideal = ideal_lp(cutoff, 1024)
    
    for i, (wf, wn) in enumerate(zip(weight_funcs, weight_names)):
        h, w, H = wls_fir_design(order, cutoff, wf, transition_width=transition_width)
        
        plt.subplot(3, 2, i+1)
        plt.plot(w/np.pi, H, 'b-', linewidth=2, label=f'WLS Filter')
        plt.plot(w_ideal/np.pi, H_ideal, 'r--', linewidth=1.5, label='Ideal')
        plt.title(f'Weight: {wn}')
        plt.xlabel('Normalized Frequency (×π rad/sample)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.legend()
        plt.ylim(-0.1, 1.1)
    
    plt.suptitle(f'Effect of Weight Functions - Transition Width: {transition_width:.2f}π', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    # Plot in dB scale for better stopband visualization
    plt.figure(figsize=(15, 10))
    
    for i, (wf, wn) in enumerate(zip(weight_funcs, weight_names)):
        h, w, H = wls_fir_design(order, cutoff, wf, transition_width=transition_width)
        
        plt.subplot(3, 2, i+1)
        plt.plot(w/np.pi, 20*np.log10(np.abs(H) + 1e-10), 'b-', linewidth=2)
        plt.title(f'Weight: {wn}')
        plt.xlabel('Normalized Frequency (×π rad/sample)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.ylim(-80, 5)
    
    plt.suptitle(f'Stopband Attenuation with Different Weight Functions - Transition Width: {transition_width:.2f}π', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Main execution
if __name__ == "__main__":
    # Design parameters
    filter_order = 41    # Filter order (odd for linear phase)
    cutoff_freq = np.pi/2  # Cutoff frequency (π/2 radians)
    
    # Get ideal response for comparison
    H_ideal, w_ideal = ideal_lp(cutoff_freq, 1024)
    
    # Example of single filter design and visualization
    weight_func = 2  # Higher weight in stopband
    transition_width = 0.1  # Transition width in radians (as fraction of π)
    
    h, w, H = wls_fir_design(filter_order, cutoff_freq, weight_func, transition_width)
    
    plot_filter_response(h, w, H, H_ideal, "Higher in Stopband", transition_width)
    
    # Compare different transition widths
    compare_transition_widths(filter_order, cutoff_freq, weight_func, "Higher in Stopband")
    
    # Compare different weight functions
    compare_weight_functions(filter_order, cutoff_freq, transition_width)