import numpy as np
from numpy import cos, pi, sin
import matplotlib.pyplot as plt
from fractions import Fraction
# -----------------------------------------------------------
# System parameters
# -----------------------------------------------------------
q = 2
F = 1.1799
om = Fraction(2, 3)
T = 2*pi / om
# -----------------------------------------------------------
# Initial conditions: grid
# ----------------------------------------------------------- 
x0_values = np.arange(-3.5, 3.5, 0.1)
v0_values = np.arange(-pi, pi, 0.5)
X0, V0 = np.meshgrid(x0_values, v0_values)
x0_flat = X0.ravel()
vo_flat = V0.ravel()
n_orbits = len(x0_flat)
y = np.concatenate([x0_flat, vo_flat])
# -----------------------------------------------------------
# Method parameters
# -----------------------------------------------------------
Trans = 150
Nperiods = 1000
steps_per_T = 300
dt = T / steps_per_T
# -----------------------------------------------------------
# Dynamics: Forced Duffing oscillator
# -----------------------------------------------------------
def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:]
    dx = v
    dv = -(1/q)*v - sin(x) + F*cos(om*t)
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
def wrap(x):
    return (x + pi) % (2*pi) - pi

n_saved = Nperiods - Trans + 1
x_strobe = np.empty((n_saved, n_orbits))
v_strobe = np.empty((n_saved, n_orbits))
# Initial stroboscopic points
save_index = 0
if Trans == 0:
    x_strobe[save_index] = wrap(y[:n_orbits])
    v_strobe[save_index] = y[n_orbits:]
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
            x_strobe[save_index] = wrap(y[:n_orbits])
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
ax.set_xlabel(rf"$\theta$", fontsize=22)
ax.set_ylabel(r"$\dot{theta}$", fontsize=22)
ax.tick_params(axis="both", labelsize=18)
ax.text(0.03,0.97,rf'$q = {q}$', transform=ax.transAxes,ha='left',va='top',fontsize=14) 
ax.text(0.97,0.97,rf'$\gamma = {F}$', transform=ax.transAxes,ha='right',va='top',fontsize=14)
ax.text(0.03,0.03,rf'$\omega = {om}$', transform=ax.transAxes,ha='left',va='bottom',fontsize=14)
ax.set_box_aspect(0.65)
plt.tight_layout()
plt.savefig('MS_Atractor.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()