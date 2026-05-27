"""
visualization.py
----------------
Rendering helpers. Two paths are provided:

* `wave_field_to_rgb` -> a fast NumPy-only RGB image suitable for st.image,
  which is what the animation loop uses for a fair framerate.
* `plot_wave_field` / `plot_intensity_profile` -> Matplotlib figures for the
  final, higher-quality output.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from analysis import detect_intensity_peaks, theoretical_fringe_spacing


# A compact RdBu-like diverging colormap baked into a 256-entry lookup table.
def _build_rdbu_lut() -> np.ndarray:
    # 5 anchor colors from blue -> white -> red.
    anchors = np.array(
        [
            [5, 48, 97],
            [67, 147, 195],
            [247, 247, 247],
            [214, 96, 77],
            [103, 0, 31],
        ],
        dtype=np.float32,
    )
    xs = np.linspace(0.0, 1.0, anchors.shape[0])
    grid = np.linspace(0.0, 1.0, 256)
    lut = np.empty((256, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.clip(np.interp(grid, xs, anchors[:, c]), 0, 255).astype(np.uint8)
    return lut


_RDBU_LUT = _build_rdbu_lut()


def wave_field_to_rgb(
    sim,
    vmax: float = 1.0,
    show_markers: bool = True,
    upscale: int = 3,
) -> np.ndarray:
    """Render the wave field as an RGB image (H, W, 3) for st.image."""
    u = sim.u  # current wave field, shape (nx, ny)
    # Normalize to [0, 1] using a symmetric clip around 0.
    norm = np.clip(u / (2.0 * vmax) + 0.5, 0.0, 1.0)
    idx = (norm * 255).astype(np.uint8)
    rgb = _RDBU_LUT[idx]  # (nx, ny, 3)

    # Overlay barrier in dark gray.
    barrier = sim.barrier_mask
    rgb[barrier] = (40, 40, 40)

    if show_markers:
        # Source line: thin yellow stripe.
        sx = sim.source_x
        rgb[sx, :] = (255, 220, 80)
        # Screen line: thin green stripe.
        scx = sim.screen_x
        rgb[scx, :] = (60, 200, 120)

    # rgb is (nx, ny, 3) where nx -> x and ny -> y. For display we want a
    # standard image with width = nx (x left-to-right) and height = ny.
    # Transpose to (ny, nx, 3).
    img = np.transpose(rgb, (1, 0, 2))

    if upscale > 1:
        img = np.repeat(np.repeat(img, upscale, axis=0), upscale, axis=1)
    return img


def plot_wave_field(sim, show_markers: bool = True):
    """Matplotlib heatmap of the current wave field."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # imshow expects (rows, cols) = (y, x). u is (nx, ny) so transpose.
    field = sim.u.T
    im = ax.imshow(
        field,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
    )

    # Overlay barrier as dark scatter.
    if show_markers:
        bx, by = np.where(sim.barrier_mask)
        ax.scatter(bx, by, c="black", s=1, marker="s")
        ax.axvline(sim.source_x, color="gold", linewidth=1.0, alpha=0.8, label="Source")
        ax.axvline(
            sim.screen_x, color="lime", linewidth=1.0, alpha=0.8, label="Screen"
        )
        ax.legend(loc="upper right", fontsize=8)

    ax.set_title("Animated Wave Field  u(x, y)")
    ax.set_xlabel("x (grid units)")
    ax.set_ylabel("y (grid units)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="amplitude")
    fig.tight_layout()
    return fig


def plot_intensity_profile(
    intensity: np.ndarray,
    wavelength: Optional[float] = None,
    screen_distance: Optional[float] = None,
    slit_separation: Optional[float] = None,
    show_theory: bool = True,
    mode: str = "Double Slit",
):
    """Plot the time-averaged intensity along the screen, with optional peaks."""
    intensity = np.asarray(intensity, dtype=float)
    y = np.arange(intensity.size)

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    if intensity.max() > 0:
        norm = intensity / intensity.max()
    else:
        norm = intensity
    ax.plot(y, norm, color="#1f77b4", linewidth=1.4, label="Measured intensity")
    ax.fill_between(y, 0, norm, color="#1f77b4", alpha=0.15)

    peaks = detect_intensity_peaks(intensity)
    if peaks.size > 0:
        ax.plot(peaks, norm[peaks], "rx", markersize=8, label="Detected peaks")

    if (
        show_theory
        and mode == "Double Slit"
        and wavelength
        and screen_distance
        and slit_separation
    ):
        spacing = theoretical_fringe_spacing(wavelength, screen_distance, slit_separation)
        if np.isfinite(spacing) and spacing > 0:
            cy = intensity.size // 2
            k = 0
            while True:
                pos_right = cy + k * spacing
                pos_left = cy - k * spacing
                drew = False
                if 0 <= pos_right < intensity.size:
                    ax.axvline(
                        pos_right,
                        color="red",
                        linestyle="--",
                        alpha=0.35,
                        linewidth=0.9,
                        label="Predicted fringe" if k == 0 else None,
                    )
                    drew = True
                if k != 0 and 0 <= pos_left < intensity.size:
                    ax.axvline(pos_left, color="red", linestyle="--", alpha=0.35, linewidth=0.9)
                    drew = True
                if not drew:
                    break
                k += 1
                if k > 30:
                    break

    ax.set_title("Screen Intensity  I(y) = <u^2>_t")
    ax.set_xlabel("y (grid units along screen)")
    ax.set_ylabel("Normalized intensity")
    ax.set_xlim(0, intensity.size - 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig
