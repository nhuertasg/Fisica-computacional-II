"""
Péndulo con soporte deslizante y resorte -- Manim (Community Edition)
Compilar con extension de VScode y manim instalado, adjunto video 
de el resultado en el github.
"""

import numpy as np
from numpy import sin, cos
from manim import *

# -----------------------------------------------------------
# System parameters
# -----------------------------------------------------------
g = 9.81
l = 1.0      
m2 = 1.0      # masa del péndulo
m1 = 4.0    # masa del bloque
k = 50.0     

# -----------------------------------------------------------
# Initial conditions
# -----------------------------------------------------------
x0 = 0.2                  
v0 = 0.0                  
th0 = np.radians(120.0)    
ome0 = 0.0                 

# -----------------------------------------------------------
# method parameters
# -----------------------------------------------------------
tmax = 60
dt = 0.001


# -----------------------------------------------------------
# Dynamics
# -----------------------------------------------------------
def dyn(t, y):
    x, v, th, ome = y
    s, c = sin(th), cos(th)
    den = m1 / m2 + s**2
    a_x = ((g * c + l * ome**2) * s - (k / m2) * x) / den
    a_th = -(1.0 / l) * (g * (1 + m1 / m2) * s + c * (l * ome**2 * s - (k / m2) * x)) / den
    return np.array([v, a_x, ome, a_th])

# -----------------------------------------------------------
# RK4
# -----------------------------------------------------------
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h / 2, y + k1 / 2)
    k3 = h * f(t + h / 2, y + k2 / 2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6


# -----------------------------------------------------------
# Integration using RK4
# -----------------------------------------------------------
n = int(tmax / dt)
t_arr = np.linspace(0, n * dt, n + 1)
y = np.empty((n + 1, 4))
y[0] = np.array([x0, v0, th0, ome0])
for i in range(n):
    y[i + 1] = rk4(dyn, t_arr[i], y[i], dt)

x_arr = y[:, 0]
th_arr = y[:, 2]

# -----------------------------------------------------------
# Kinematics
# -----------------------------------------------------------
x_bob_arr = x_arr + l * sin(th_arr)
y_bob_arr = -l * cos(th_arr)

# -----------------------------------------------------------
# Escala física -> unidades de Manim
# -----------------------------------------------------------
SCALE = 2.2


def pos_at(t):
    """Interpola bloque y péndulo en el instante t (igual idea que recorrer el arreglo, pero continuo)."""
    xb = np.interp(t, t_arr, x_arr)
    xp = np.interp(t, t_arr, x_bob_arr)
    yp = np.interp(t, t_arr, y_bob_arr)
    return xb, xp, yp


def diagonal_hatch(rect, spacing=0.15):
    """Rayado diagonal recortado exactamente al área de `rect`."""
    x_min, x_max = rect.get_left()[0], rect.get_right()[0]
    y_min, y_max = rect.get_bottom()[1], rect.get_top()[1]
    diag = (x_max - x_min) + (y_max - y_min)
    n_lines = int(diag / spacing) + 2
    lines = VGroup()
    for i in range(-n_lines, n_lines):
        x0_ = x_min + i * spacing
        candidates = []
        for x in (x_min, x_max):
            yv = x - x0_ + y_min
            if y_min <= yv <= y_max:
                candidates.append((x, yv))
        for yv in (y_min, y_max):
            xv = x0_ + (yv - y_min)
            if x_min <= xv <= x_max:
                candidates.append((xv, yv))
        if len(candidates) >= 2:
            p1 = np.array([*candidates[0], 0])
            p2 = np.array([*candidates[1], 0])
            lines.add(Line(p1, p2, stroke_width=1.2, color="#f4d35e"))
    return lines


class PenduloResorte(Scene):
    def construct(self):
        self.camera.background_color = "#1b1023"

        R = l + max(abs(x_arr).max(), 0.5) + 0.3
        x_wall = -R + 0.15

        # ------------------------------------------------------
        # Eje de coordenadas de referencia
        # ------------------------------------------------------
        axes = Axes(
            x_range=[-R, R, 1],
            y_range=[-(l + 0.5), 1.5, 1],
            x_length=2 * R * SCALE,
            y_length=(l + 2.0) * SCALE,
            axis_config={"color": "#6b5b73", "stroke_width": 1.5, "include_tip": True,
                         "tip_width": 0.15, "tip_height": 0.15},
        ).move_to(ORIGIN)
        axes.add_coordinates(
            font_size=22,
            color="#a89ba8",
        )

        # ------------------------------------------------------
        # Pared
        # ------------------------------------------------------
        wall_width = 0.08 * SCALE
        wall = Rectangle(
            width=wall_width,
            height=5.5,
            fill_color="#3a2e46",
            fill_opacity=1,
            stroke_color="#f4d35e",
            stroke_width=2.5,
        ).move_to(np.array([x_wall * SCALE, 0, 0]))
        hatch = diagonal_hatch(wall)

        # ------------------------------------------------------
        # Bloque M (rectángulo)
        # ------------------------------------------------------
        block_w, block_h = 0.35 * SCALE, 0.30 * SCALE
        block = Rectangle(
            width=block_w,
            height=block_h,
            fill_color="#ee6c4d",
            fill_opacity=1,
            stroke_color="#fdf0d5",
            stroke_width=2.5,
        )
        block.move_to(np.array([x_arr[0] * SCALE, 0, 0]))

        # ------------------------------------------------------
        # Masa del péndulo
        # ------------------------------------------------------
        bob = Dot(
            radius=0.11 * SCALE,
            color="#f4d35e",
            fill_opacity=1,
            stroke_color="#3a2e46",
            stroke_width=2,
        )
        bob.move_to(np.array([x_bob_arr[0] * SCALE, y_bob_arr[0] * SCALE, 0]))

        # ------------------------------------------------------
        # Varilla
        # ------------------------------------------------------
        rod = always_redraw(
            lambda: Line(
                np.array([block.get_center()[0], -block_h / 2, 0]),
                bob.get_center(),
                stroke_width=3.5,
                color="#fdf0d5",
            )
        )

        # ------------------------------------------------------
        # Resorte helicoidal
        # ------------------------------------------------------
        l0_ref = abs(x_wall)

        def make_spring():
            x1 = wall.get_right()[0]
            x2 = block.get_left()[0]
            L = abs(x2 - x1) / SCALE + 1e-9
            n_coils = 12
            s_param = np.linspace(0, 1, 300)
            s = 0.5 * (1 - np.cos(np.pi * s_param))
            amp = np.clip(0.08 * np.sqrt(l0_ref / L), 0.04, 0.10) * SCALE
            wave = amp * np.sin(2 * np.pi * n_coils * s)
            xs = x1 + (x2 - x1) * s
            ys = wave
            pts = np.array([xs, ys, np.zeros_like(xs)]).T
            return VMobject(color="#e0b1cb", stroke_width=2.5).set_points_smoothly(pts)

        spring = always_redraw(make_spring)

        # ------------------------------------------------------
        # Rastro y contador de tiempo
        # ------------------------------------------------------
        trace = TracedPath(
            bob.get_center, stroke_color="#f4d35e", stroke_width=2, stroke_opacity=0.5
        )

        time_tracker = ValueTracker(0)
        clock_label = always_redraw(
            lambda: Text(
                f"t = {time_tracker.get_value():.1f} s", font_size=30, color="#fdf0d5"
            ).to_corner(UL, buff=0.4)
        )

        title = Text("Péndulo con soporte deslizante y resorte", font_size=32, color="#fdf0d5")
        title.to_edge(UP, buff=0.4)

        self.add(axes, wall, hatch, spring, block, rod, trace, bob, clock_label, title)

        # ------------------------------------------------------
        # Animación continua sobre la trayectoria física real
        # ------------------------------------------------------
        def update_scene(mob, alpha):
            ti = alpha * tmax
            xb, xp, yp = pos_at(ti)
            time_tracker.set_value(ti)
            block.move_to(np.array([xb * SCALE, 0, 0]))
            bob.move_to(np.array([xp * SCALE, yp * SCALE, 0]))

        self.play(
            UpdateFromAlphaFunc(block, update_scene),
            run_time=tmax,
            rate_func=linear,
        )
        self.wait(0.5)