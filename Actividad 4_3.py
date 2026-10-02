import numpy as np
from numpy import cos, pi
import matplotlib.pyplot as plt
# -----------------------------------------------------------
# System parameters
# -----------------------------------------------------------
delta = 0.15
alpha = 1
beta = 1
gamma = 0.3
om = 1.0
panel = "(b)"
T = 2*pi / om
# -----------------------------------------------------------
# Initial conditions: grid
# ----------------------------------------------------------- 
x0_values = np.linspace(-2, 2, 50)
v0_values = np.linspace(-2, 2, 50)
X0, V0 = np.meshgrid(x0_values, v0_values)
x0_flat = X0.ravel()
vo_flat = V0.ravel()
n_orbits = len(x0_flat)
y = np.concatenate([x0_flat, vo_flat])
# -----------------------------------------------------------
# Method parameters
# -----------------------------------------------------------
Trans = 200
Nperiods = 10000
steps_per_T = 200
dt = T / steps_per_T
# -----------------------------------------------------------
# Dynamics: Forced Duffing oscillator
# -----------------------------------------------------------
def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:]
    dx = v
    dv = gamma*cos(om*t) - delta*v + alpha*x - beta*x**3
    return np.concatenate([dx, dv])
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
n_saved = Nperiods - Trans + 1
x_strobe = np.empty((n_saved, n_orbits))
v_strobe = np.empty((n_saved, n_orbits))
# Initial stroboscopic points
save_index = 0
if Trans == 0:
    x_strobe[save_index] = y[:n_orbits]
    v_strobe[save_index] = y[n_orbits:]
    save_index += 1
# -----------------------------------------------------------
# Integration
# -----------------------------------------------------------
total_steps = Nperiods * steps_per_T
for step in range(total_steps):
    current_time = step*dt
    y = rk4(dyn, current_time, y, dt)
    completed_period = (step + 1) // steps_per_T
    if (step + 1) % steps_per_T == 0 and completed_period >= Trans:
        x_strobe[save_index] = y[:n_orbits]
        v_strobe[save_index] = y[n_orbits:]
        save_index += 1
# -----------------------------------------------------------
# Colors according to initial conditions & Stoboscopic map
# -----------------------------------------------------------
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
orbit_colors = [colors[i % len(colors)] for i in range(n_orbits)]
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(n_orbits):
    ax.scatter(x_strobe[:, i], v_strobe[:, i], s=1, color=orbit_colors[i],
               linewidths=0, rasterized=True)
ax.set_xlabel(r"$x$", fontsize=22)
ax.set_ylabel(r"$\dot{x}$", fontsize=22)
ax.tick_params(axis="both", labelsize=18)
ax.set_box_aspect(0.70)
ax.text(0.02, 0.97, panel, transform=ax.transAxes, ha='left', va='top', fontsize=18)
ax.text(0.98, 0.97, rf"$\delta$ = {delta}", transform=ax.transAxes, ha='right', va='top', fontsize=18)
plt.tight_layout()
plt.savefig('MS_ej3.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()