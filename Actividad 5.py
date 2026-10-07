import numpy as np
from numpy import cos, pi
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# System parameters
# -----------------------------------------------------------
delta, alpha, beta, om = 0.1, 2.0, 2.0, 1.2
T = 2*pi/om

# -----------------------------------------------------------
# Initial conditions
# -----------------------------------------------------------
x0, v0 = 1.0, 1.0

# -----------------------------------------------------------
# Gamma values
# -----------------------------------------------------------
g_min, g_max, dg = 0.1, 7.0, 0.001
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
Trans, Nkeep, steps_per_T = 180, 50, 150
dt = T / steps_per_T

# -----------------------------------------------------------
# Dynamics
# -----------------------------------------------------------
def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:]
    dx = v
    dv = -delta*v + alpha*x - beta*x**3 + g_values*cos(om*t)
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
total_steps = (Trans + Nkeep) * steps_per_T
for step in range(total_steps):
    y = rk4(dyn, step*dt, y, dt)
    if (step + 1) % steps_per_T == 0:
        if (step + 1) // steps_per_T > Trans:   # ya pasó el transitorio
            x_strobe[save_index] = y[:n_orbits]
            v_strobe[save_index] = y[n_orbits:]
            save_index += 1

# -----------------------------------------------------------
# Bifurcation diagram & Figure format
# -----------------------------------------------------------
G = np.tile(g_values, Nkeep)   # misma forma que x_strobe.ravel()
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].scatter(G, x_strobe.ravel(), s=0.3, color='blue', linewidths=0, rasterized=True)
axs[0].set_ylabel(r'$x$', fontsize=16)
axs[1].scatter(G, v_strobe.ravel(), s=0.3, color='blue', linewidths=0, rasterized=True)
axs[1].set_ylabel(r'$\dot{x}$', fontsize=16)
for ax, label in zip(axs, ['(a)', '(b)']):
    ax.set_xlabel(r'$\gamma$', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(g_min, g_max)
    ax.text(0.02, 0.97, label, transform=ax.transAxes, fontsize=13,
            style='italic', va='top', ha='left')
plt.tight_layout()
plt.savefig('Bif_Duffing.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()