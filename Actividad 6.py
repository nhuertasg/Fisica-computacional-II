import numpy as np
from numpy import sin, cos, pi, abs
import matplotlib.pyplot as plt
# -----------------------------------------------------------
# System parameters
# -----------------------------------------------------------
om, om0, alpha = 2, 1, 0.1
T = 2*pi/om
# -----------------------------------------------------------
# Initial conditions
# -----------------------------------------------------------
x0, v0 = 1, 1
# -----------------------------------------------------------
# Gamma values
# -----------------------------------------------------------
g_min = 0.00
g_max = 2.25
dg = 0.001
g_values = np.arange(g_min, g_max + dg, dg)
n_orbits = len(g_values)
# -----------------------------------------------------------
# Initial state
# -----------------------------------------------------------
x = np.full(n_orbits, x0)
v = np.full(n_orbits, v0)
y = np.concatenate([x, v])
# -----------------------------------------------------------
# Numerical method parameters
# -----------------------------------------------------------
Trans, Nkeep, steps_per_T = 150, 50, 150
dt = T / steps_per_T
# -----------------------------------------------------------
# Dynamics
# -----------------------------------------------------------
def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:]
    dx = v
    dv = -alpha*v - om0**2*sin(x) + g_values*cos(om*t)*sin(x)
    return np.concatenate([dx, dv])
# -----------------------------------------------------------
# Fourth-order Runge-Kutta method
# -----------------------------------------------------------
def rk4(f, t, y, h):
    k1 = h*f(t, y)
    k2 = h*f(t + h/2, y + k1/2)
    k3 = h*f(t + h/2, y + k2/2)
    k4 = h*f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6
# -----------------------------------------------------------
# Stroboscopic storage
# -----------------------------------------------------------
x_strobe = np.empty((Nkeep, n_orbits))
v_strobe = np.empty((Nkeep, n_orbits))
save_index = 0
# -----------------------------------------------------------
# Integration
# -----------------------------------------------------------
total_periods = Trans + Nkeep
total_steps = total_periods * steps_per_T
for step in range(total_steps):
    current_time = step*dt
    y = rk4(dyn, current_time, y, dt)
    completed_period = (step + 1) // steps_per_T
    if (step + 1) % steps_per_T == 0:
        if completed_period > Trans:
            x_strobe[save_index] = y[:n_orbits]
            v_strobe[save_index] = y[n_orbits:]
            save_index += 1
# -----------------------------------------------------------
# Bifurcation diagram & Figure format
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(n_orbits):
    ax.scatter(np.full(Nkeep, g_values[i]),
               abs(v_strobe[:, i]), s=0.6, color='blue', linewidths=0, rasterized=True)
ax.set_xlabel(r'$\gamma$', fontsize=16)
ax.set_ylabel(r'$\dot{|\theta|}$', fontsize=16)
ax.tick_params(axis='both', labelsize=12)
ax.set_xlim(g_min, g_max)
ax.set_box_aspect(0.65)
plt.tight_layout()
plt.savefig('Bif_A5.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()