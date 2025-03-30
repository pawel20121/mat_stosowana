import numpy as np
import matplotlib.pyplot as plt

# Lorenz system parameters
sigma = 10.0
r = 28.0
b = 8.0 / 3.0
dt = 0.01  # Time step
num_steps = 10000  # Number of steps

# Initialize arrays
x = np.zeros(num_steps)
y = np.zeros(num_steps)
z = np.zeros(num_steps)

# Initial conditions
x[0], y[0], z[0] = 1.0, 1.0, 1.0

# Euler method for solving the Lorenz system
for i in range(num_steps - 1):
    x[i + 1] = x[i] + dt * sigma * (y[i] - x[i])
    y[i + 1] = y[i] + dt * (x[i] * (r - z[i]) - y[i])
    z[i + 1] = z[i] + dt * (x[i] * y[i] - b * z[i])

# Plot the Lorenz attractor
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")
plt.show()

# Plot x, y, z as functions of time
time = np.linspace(0, num_steps * dt, num_steps)
fig, axs = plt.subplots(3, 1, figsize=(10, 7))

axs[0].plot(time, x, label='x(t)', color='b')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('X')
axs[0].legend()
axs[0].grid()

axs[1].plot(time, y, label='y(t)', color='r')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Y')
axs[1].legend()
axs[1].grid()

axs[2].plot(time, z, label='z(t)', color='g')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Z')
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()
