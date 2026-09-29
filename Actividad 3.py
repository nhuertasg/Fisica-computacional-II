import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
# -----------------------------------------------------------
# System parameters
# -----------------------------------------------------------
ml = 1.0
gl = 1.0
F0 = 0.01
om = 2/pi
T = 2*pi / om
# -----------------------------------------------------------
# Initial conditions: grid
# ----------------------------------------------------------- 
th0_values = np.arange(-3, 3.1, 0.3)
w0_values = np.arange(-2.5, 2.6, 0.25)
TH0, W0 = np.meshgrid(th0_values, w0_values)
th0_flat = TH0.ravel()
w0_flat = W0.ravel()
n_orbits = len(th0_flat)
y = np.concatenate([th0_flat, w0_flat])
# -----------------------------------------------------------
# Method parameters
# -----------------------------------------------------------
Trans = 0
Nperiods = 10000
steps_per_T = 300
dt = T / steps_per_T
# -----------------------------------------------------------
# Dynamics: Forced simple pendulum
# -----------------------------------------------------------
def dyn(t, y):
    th = y[:n_orbits]
    w = y[n_orbits:]
    dth = w
    dw = -gl*sin(th) + (F0/ml)*cos(om*t)
    return np.concatenate([dth, dw])
# -----------------------------------------------------------
# Fourth-order Runge-Kutta method
# -----------------------------------------------------------
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6
# -----------------------------------------------------------
# Stroboscopic storage
# -----------------------------------------------------------
def wrap(th):
    return (th + pi) % (2*pi) - pi

n_saved = Nperiods - Trans + 1
th_strobe = np.empty((n_saved, n_orbits))
w_strobe = np.empty((n_saved, n_orbits))
# Initial stroboscopic points
save_index = 0
if Trans == 0:
    th_strobe[save_index] = wrap(y[:n_orbits])
    w_strobe[save_index] = y[n_orbits:]
    save_index += 1
# -----------------------------------------------------------
# Integration
# -----------------------------------------------------------
total_steps = Nperiods * steps_per_T
for step in range(total_steps):
    current_time = step*dt
    y = rk4(dyn, current_time, y, dt)
    if (step + 1) % steps_per_T == 0:
        completed_period = (step + 1) // steps_per_T
        if completed_period >= Trans and save_index < n_saved:
            th_strobe[save_index] = wrap(y[:n_orbits])
            w_strobe[save_index] = y[n_orbits:]
            save_index += 1
# -----------------------------------------------------------
# Colors according to initial conditions & Stoboscopic map
# -----------------------------------------------------------
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
orbit_colors = [colors[i % len(colors)] for i in range(n_orbits)]
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(n_orbits):
    ax.scatter(th_strobe[:, i], w_strobe[:, i], s=1, color=orbit_colors[i], linewidths=0, rasterized=True)
ax.set_xlim(-pi, pi)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel(r"$\theta$", fontsize=22)
ax.set_ylabel(r"$\dot{\theta}$", fontsize=22)
ax.tick_params(axis="both", labelsize=18)
ax.set_box_aspect(0.70)
plt.tight_layout()
plt.savefig('MSPF.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()