

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont

# Plate size and resolution
w = h = 1.0
nx, ny = 200, 200  # Increased resolution

# Diffusion coefficient
D = 1

# Time step (stability condition)
dx2, dy2 = (1.0 / nx) ** 2, (1.0 / ny) ** 2
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

# Initial and boundary conditions
Tcool, Thot = 0.0, 1.0
u0 = Tcool * np.ones((nx, ny))

# Generate text mask
text = "AGH"
img = Image.new('L', (nx, ny), color=0)
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("arial.ttf", 80)  # Increased font size
except OSError:
    font = ImageFont.load_default()  # Use default font if "arial.ttf" is not found

draw.text((30, 60), text, fill=255, font=font)  # Adjusted position for larger text
mask = np.array(img) > 128

# Apply initial heat distribution
u0[mask] = Thot
u = u0.copy()

def do_timestep(u0, u):
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * (
        (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2 +
        (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2
    )
    u0[:] = u
    return u0, u

# Number of timesteps
nsteps = 500
save_steps = [0, 10, 50, 100, 150, 200, 300, 400,500]

# Plot results
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
fignum = 0

for m in range(nsteps + 1):
    u0, u = do_timestep(u0, u)
    
    if m in save_steps:
        ax = axes[fignum // 3, fignum % 3]
        im = ax.imshow(gaussian_filter(u, sigma=1), cmap='jet', vmin=Tcool, vmax=Thot)
        ax.set_title(f't = {m} steps')
        ax.set_axis_off()
        fignum += 1

fig.colorbar(im, cax=cbar_ax)

cbar_ax.set_xlabel('T')
# plt.tight_layout()
plt.show()
