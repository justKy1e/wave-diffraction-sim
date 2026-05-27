"""
wave_solver.py
--------------
Finite-Difference Time-Domain (FDTD) solver for the 2D scalar wave equation:

    d^2 u / dt^2 = c^2 (d^2 u/dx^2 + d^2 u/dy^2)

The leap-frog update is:

    u_next[i,j] = 2*u[i,j] - u_prev[i,j] + r^2 * laplacian(u)[i,j]

where r = c*dt/dx. For 2D explicit FDTD the Courant stability condition is
r <= 1/sqrt(2) ~= 0.707. We use r = 0.45 by default, comfortably below the
limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SimParams:
    nx: int = 640
    ny: int = 320
    wavelength: float = 10.0
    slit_width: int = 6
    slit_separation: int = 40
    screen_distance: int = 420
    mode: str = "Double Slit"  # or "Single Slit"
    dt: float = 0.45
    c: float = 1.0
    dx: float = 1.0
    source_amplitude: float = 0.35
    damping_layer: int = 18
    warmup_steps: int = 0  # 0 = auto from travel time


class WaveSimulation:
    """2D FDTD scalar-wave simulation with a slit barrier and a screen line.

    The grid (600 x 300) is sized so the screen can sit in the Fraunhofer
    (far-field) regime — d²/(λL) << 1 — while also keeping the diffraction
    angle small (λ/d ≤ ~0.25) so that the textbook Δy = λL/d formula is
    actually valid for comparison.
    """

    def __init__(
        self,
        nx: int = 640,
        ny: int = 320,
        wavelength: float = 10.0,
        slit_width: int = 6,
        slit_separation: int = 40,
        mode: str = "Double Slit",
        screen_distance: int = 420,
        dt: float = 0.45,
        c: float = 1.0,
        dx: float = 1.0,
        source_amplitude: float = 0.35,
        damping_layer: int = 32,
        warmup_steps: int = 0,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.wavelength = float(wavelength)
        self.slit_width = int(slit_width)
        self.slit_separation = int(slit_separation)
        self.mode = mode
        self.screen_distance = int(screen_distance)
        self.dt = float(dt)
        self.c = float(c)
        self.dx = float(dx)
        self.source_amplitude = float(source_amplitude)
        self.damping_layer = int(damping_layer)
        # Warmup placeholder; the true value is set below once source_x and
        # screen_x are known so it covers the full source→screen travel time.
        self._warmup_override = int(warmup_steps) if warmup_steps and warmup_steps > 0 else 0
        self.warmup_steps = 0

        # Courant number r = c*dt/dx (must be <= 1/sqrt(2) in 2D)
        self.r = self.c * self.dt / self.dx
        self.r2 = self.r * self.r

        # Wave-field arrays: previous, current, next.
        self.u_prev = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.u = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.u_next = np.zeros((self.nx, self.ny), dtype=np.float32)

        # Geometry: source on the left, barrier ~1/5 in so most of the
        # domain is available for far-field propagation to the screen.
        self.source_x = 14
        self.barrier_x = self.nx // 5
        # Keep the screen well clear of the damping layer.
        screen_max = self.nx - (self.damping_layer + 10)
        self.screen_x = min(screen_max, self.barrier_x + self.screen_distance)

        # Now that source_x and screen_x exist, finalize warmup steps:
        # wait long enough for the wave to physically reach the screen plus a
        # generous margin for steady-state oscillation to set in.
        if self._warmup_override:
            self.warmup_steps = self._warmup_override
        else:
            travel = (self.screen_x - self.source_x + 30) / (self.c * self.dt)
            self.warmup_steps = int(travel + 250)

        # Angular frequency for the sinusoidal source.
        self.omega = 2.0 * np.pi * self.c / self.wavelength

        # Time / step counters.
        self.t = 0.0
        self.step_count = 0

        # Build static masks.
        self.barrier_mask = self.create_barrier_mask()
        self.damping_mask = self.create_damping_mask()

        # Time-averaged intensity along the screen line.
        self.intensity_accum = np.zeros(self.ny, dtype=np.float64)
        self.intensity_count = 0

    # ------------------------------------------------------------------ masks
    def create_barrier_mask(self) -> np.ndarray:
        """Return a boolean mask that is True wherever the wall blocks the wave."""
        mask = np.zeros((self.nx, self.ny), dtype=bool)
        # The barrier is a thin vertical wall; 2 cells thick helps prevent
        # numerical tunneling at short wavelengths.
        wall_thickness = 2
        x0 = self.barrier_x
        x1 = self.barrier_x + wall_thickness

        # Start as a fully-closed wall, then carve openings.
        mask[x0:x1, :] = True

        cy = self.ny // 2
        sw = max(1, self.slit_width)

        if self.mode == "Single Slit":
            y_lo = cy - sw // 2
            y_hi = y_lo + sw
            mask[x0:x1, y_lo:y_hi] = False
        else:  # Double Slit
            d = max(sw + 1, self.slit_separation)  # avoid overlap
            top_center = cy - d // 2
            bot_center = cy + d // 2
            for center in (top_center, bot_center):
                y_lo = center - sw // 2
                y_hi = y_lo + sw
                y_lo = max(0, y_lo)
                y_hi = min(self.ny, y_hi)
                mask[x0:x1, y_lo:y_hi] = False
        return mask

    def create_damping_mask(self) -> np.ndarray:
        """Smooth damping toward the edges to absorb outgoing waves.

        Uses a quadratic ramp so cells deep inside are nearly untouched while
        cells near the boundary are aggressively damped. This is closer to a
        sponge layer / split-field PML in spirit, and prevents the back-edge
        reflection from setting up a 2D standing wave that would otherwise
        contaminate the time-averaged intensity profile on the screen.
        """
        d = self.damping_layer
        mask = np.ones((self.nx, self.ny), dtype=np.float32)
        if d <= 0:
            return mask
        # Quadratic ramp: 1.0 deep inside, ~0.70 right at the edge.
        # k = 0 is innermost edge of the layer, k = d-1 is the boundary.
        ks = np.arange(d, dtype=np.float32)
        ramp = 1.0 - 0.30 * ((d - 1 - ks) / (d - 1)) ** 2
        for k in range(d):
            val = ramp[k]
            mask[k, :] = np.minimum(mask[k, :], val)
            mask[-(k + 1), :] = np.minimum(mask[-(k + 1), :], val)
            mask[:, k] = np.minimum(mask[:, k], val)
            mask[:, -(k + 1)] = np.minimum(mask[:, -(k + 1)], val)
        return mask

    # ---------------------------------------------------------------- physics
    def step(self) -> None:
        """Advance the wave field one timestep using the FDTD update."""
        u = self.u
        up = self.u_prev
        un = self.u_next

        # 5-point Laplacian on the interior.
        lap = (
            u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4.0 * u[1:-1, 1:-1]
        )
        un[1:-1, 1:-1] = 2.0 * u[1:-1, 1:-1] - up[1:-1, 1:-1] + self.r2 * lap

        # Hard zero boundary on the outermost ring (damping handles the rest).
        un[0, :] = 0.0
        un[-1, :] = 0.0
        un[:, 0] = 0.0
        un[:, -1] = 0.0

        # Sinusoidal line source.
        self.t += self.dt
        un[self.source_x, :] += self.source_amplitude * np.sin(self.omega * self.t)

        # Enforce the barrier as a fixed wall.
        un[self.barrier_mask] = 0.0
        u[self.barrier_mask] = 0.0
        up[self.barrier_mask] = 0.0

        # Apply damping mask (gentle absorbing layer near edges).
        un *= self.damping_mask

        # Accumulate time-averaged intensity at the screen after warmup.
        self.step_count += 1
        if self.step_count > self.warmup_steps:
            screen_vals = un[self.screen_x, :]
            self.intensity_accum += screen_vals.astype(np.float64) ** 2
            self.intensity_count += 1

        # Rotate buffers: u_prev <- u, u <- u_next, u_next is scratch next time.
        self.u_prev, self.u, self.u_next = self.u, self.u_next, self.u_prev

    def run_steps(self, n: int) -> None:
        """Advance the simulation by n physics steps."""
        for _ in range(int(n)):
            self.step()

    # --------------------------------------------------------------- accessors
    def get_screen_intensity(self) -> np.ndarray:
        """Return the time-averaged squared amplitude along the screen line."""
        if self.intensity_count == 0:
            return np.zeros(self.ny, dtype=np.float64)
        return self.intensity_accum / self.intensity_count

    def get_state(self) -> np.ndarray:
        """Return a copy of the current wave field u(x, y)."""
        return self.u.copy()

    def reset(self) -> None:
        self.u_prev.fill(0.0)
        self.u.fill(0.0)
        self.u_next.fill(0.0)
        self.t = 0.0
        self.step_count = 0
        self.intensity_accum.fill(0.0)
        self.intensity_count = 0
