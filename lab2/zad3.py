import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Given data
years = np.array([1790, 1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 
                  1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990])
population = np.array([3.9e6, 5.3e6, 7.2e6, 9.6e6, 12.9e6, 17.1e6, 23.1e6, 31.4e6, 38.6e6, 
                        50.2e6, 62.9e6, 76.0e6, 92.0e6, 105.7e6, 122.8e6, 131.7e6, 150.7e6, 
                        179.0e6, 205.0e6, 226.5e6, 248.7e6])

# Convert to log scale
y_log = np.log(population)
x_years = years - 1790  # Shift years to start from 0

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(x_years, y_log)

# Extract estimated growth rate r
r_estimated = slope
x0_estimated = np.exp(intercept)

# Print results
print(f"Estimated growth rate (r): {r_estimated:.6f}")
print(f"Estimated initial population (x0): {x0_estimated:.2f}")

# Plot data and regression line
plt.scatter(x_years, y_log, label='Data')
plt.plot(x_years, intercept + slope * x_years, color='red', label='Best Fit Line')
plt.xlabel("Years since 1790")
plt.ylabel("log(Population)")
plt.title("Log of Population vs. Time")
plt.legend()
plt.show()