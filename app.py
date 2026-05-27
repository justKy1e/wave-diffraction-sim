"""
app.py
------
Streamlit front-end for the 2D wave-diffraction simulator.

Run locally with:

    streamlit run app.py
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from analysis import (
    detect_intensity_peaks,
    estimate_fringe_spacing,
    measure_central_peak_width,
    percent_error,
    theoretical_fringe_spacing,
    theoretical_single_slit_width,
)
from visualization import (
    plot_intensity_profile,
    plot_wave_field,
    wave_field_to_rgb,
)
from wave_solver import WaveSimulation


# ----------------------------------------------------------------- page setup
st.set_page_config(
    page_title="2D Wave Diffraction Simulator",
    layout="wide"
)

st.title("Slit Happens — A 2D Wave-Diffraction Simulator")
st.caption("ASTP Final Project · Kyle Hsiung · watching the wave equation make diffraction patterns")
st.markdown(
    """
This project simulates a wave traveling toward a wall with one or two
slits cut into it, and then creates a diffraction pattern on the screen. 
Instead of using the simple double-slit formula, I built the wave from scratch
step by step, which involved solving the 2D wave equation on a grid. 
The interference and diffraction patterns emerge on their own similar to
a real ripple tank.

Once the simulation finishes, the app measures the spacing between bright
fringes on the screen and compares it to the prediction done by the formula
we learned in class:

$$ \\Delta y = \\frac{\\lambda L}{d} $$

The formula is not used to generate
the picture, only to grade it.
"""
)

with st.expander("Equations and numerical methods", expanded=False):
    st.markdown("**Main equation (2D scalar wave equation):**")
    st.latex(r"\frac{\partial^2 u}{\partial t^2} = c^2\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)")
    st.markdown("After solving the PDE, we get the update rule for the wave field at each grid point:")
    st.latex(
        r"u^{n+1}_{i,j} = 2u^{n}_{i,j} - u^{n-1}_{i,j} + r^2\,\big(u^{n}_{i+1,j} + u^{n}_{i-1,j} + u^{n}_{i,j+1} + u^{n}_{i,j-1} - 4 u^{n}_{i,j}\big)"
    )
    st.markdown(r"**Courant stability (2D):**  $r = c\,\Delta t/\Delta x \le 1/\sqrt{2}\approx 0.707$.  We use $r = 0.45$.")
    st.markdown("For validation:")
    st.latex(r"\Delta y \;=\; \frac{\lambda L}{d}")


# ----------------------------------------------------------------- sidebar UI
st.sidebar.header("Controls")

mode = st.sidebar.radio("Simulation mode", ["Double Slit", "Single Slit"], index=0)
wavelength = st.sidebar.slider("Wavelength λ (grid units)", 6, 18, 8, 1)
slit_width = st.sidebar.slider("Slit width (grid units)", 3, 18, 10, 1)
slit_separation = st.sidebar.slider("Slit separation d (grid units)", 16, 60, 40, 1)
screen_distance = st.sidebar.slider("Screen distance L (grid units)", 200, 480, 460, 10)
n_frames = st.sidebar.slider("Displayed animation frames", 80, 400, 200, 10)
substeps = st.sidebar.slider("Physics steps per displayed frame", 2, 12, 8, 1)
show_theory = st.sidebar.checkbox("Show theoretical comparison", value=True)
show_markers = st.sidebar.checkbox("Show barrier / source / screen markers", value=True)

# Live Fresnel-number readout so the user can see whether the current
# settings sit in the far-field (F << 1) or near-field (F ≳ 1) regime.
_F = (slit_separation ** 2) / (wavelength * screen_distance) if wavelength * screen_distance > 0 else float("inf")
_regime = "✅ far-field" if _F < 0.4 else ("≈ borderline" if _F < 1.0 else "⚠️ near-field")
st.sidebar.caption(f"Fresnel number  F = d²/(λL) ≈ **{_F:.2f}**  {_regime}")

st.sidebar.markdown("---")
run_anim = st.sidebar.button("▶ Run Animation", use_container_width=True)
run_final = st.sidebar.button("Generate Final Pattern", use_container_width=True)
reset_clicked = st.sidebar.button("🗑 Reset / Clear Output", use_container_width=True)


# ----------------------------------------------------------------- simulation
NX, NY = 640, 320

if reset_clicked:
    st.session_state.pop("last_intensity", None)
    st.session_state.pop("last_mode", None)
    st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()


def make_sim() -> WaveSimulation:
    return WaveSimulation(
        nx=NX,
        ny=NY,
        wavelength=wavelength,
        slit_width=slit_width,
        slit_separation=slit_separation,
        mode=mode,
        screen_distance=screen_distance,
    )


# Layout: wave field on top, intensity below.
field_box = st.container()
intensity_box = st.container()
metrics_box = st.container()


def render_intensity_and_metrics(intensity: np.ndarray) -> None:
    fig = plot_intensity_profile(
        intensity,
        wavelength=wavelength,
        screen_distance=screen_distance,
        slit_separation=slit_separation,
        show_theory=show_theory,
        mode=mode,
    )
    with intensity_box:
        st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    # Use an expected-fringe-spacing-aware min distance so wave-scale ripples
    # don't get mistaken for interference peaks.
    predicted = theoretical_fringe_spacing(wavelength, screen_distance, slit_separation)
    expected_spacing = predicted if np.isfinite(predicted) and predicted > 0 else wavelength * 4
    min_dist = max(int(expected_spacing * 0.4), int(wavelength) + 2)
    smooth_win = max(5, int(wavelength) | 1)
    peaks = detect_intensity_peaks(intensity, min_distance=min_dist, smooth_window=smooth_win)
    measured = estimate_fringe_spacing(peaks)

    with metrics_box:
        st.subheader("Validation")
        if mode == "Double Slit":
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Δy", f"{predicted:.2f} gu")
            c2.metric(
                "Measured Δy",
                f"{measured:.2f} gu" if np.isfinite(measured) else "—",
            )
            err = percent_error(measured, predicted)
            c3.metric(
                "Percent error",
                f"{err:.1f} %" if np.isfinite(err) else "—",
            )
            if peaks.size < 3:
                st.warning(
                    "Only {} peak(s) detected. The pattern is visible, but the "
                    "automatic peak detector could not reliably measure fringe "
                    "spacing under these settings.".format(int(peaks.size))
                )
        else:
            # Single-slit mode: compare the measured FWHM of the central
            # maximum to the small-angle sinc²-pattern prediction 0.886 λL/w.
            predicted_w = theoretical_single_slit_width(wavelength, screen_distance, slit_width)
            measured_w = measure_central_peak_width(intensity, frac=0.5)
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted central FWHM 0.886 λL/w", f"{predicted_w:.1f} gu")
            c2.metric(
                "Measured central FWHM",
                f"{measured_w:.1f} gu" if np.isfinite(measured_w) else "—",
            )
            err = percent_error(measured_w, predicted_w)
            c3.metric(
                "Percent error",
                f"{err:.1f} %" if np.isfinite(err) else "—",
            )
            if predicted_w > NY * 0.9:
                st.warning(
                    "The predicted central FWHM ({:.0f} gu) is wider than the screen ({} gu), "
                    "so the first minima fall off-screen and the measured FWHM is bounded by the "
                    "screen size. Increase the slit width or reduce L to fit the pattern on screen.".format(
                        predicted_w, NY
                    )
                )
            else:
                st.caption(
                    "Single-slit test: the central diffraction maximum should narrow as the slit "
                    "widens (FWHM ∝ 1/w). Drag the slit-width slider to see this."
                )


# --------------------------------------------------------------- run animation
if run_anim:
    sim = make_sim()
    placeholder = field_box.empty()
    progress = st.progress(0.0)
    fps_text = st.empty()

    t0 = time.time()
    last_draw = t0
    frames_drawn = 0

    for frame in range(n_frames):
        sim.run_steps(substeps)
        img = wave_field_to_rgb(sim, vmax=1.0, show_markers=show_markers, upscale=2)
        placeholder.image(
            img,
            caption=f"Animated Wave Field — frame {frame + 1}/{n_frames}",
            use_container_width=True,
        )
        frames_drawn += 1
        progress.progress((frame + 1) / n_frames)

        now = time.time()
        if now - last_draw > 0.5:
            fps = frames_drawn / (now - t0)
            fps_text.caption(f"≈ {fps:.1f} displayed FPS")
            last_draw = now

    fps_text.caption(f"Done — average {frames_drawn / max(time.time() - t0, 1e-6):.1f} FPS")

    intensity = sim.get_screen_intensity()
    st.session_state["last_intensity"] = intensity
    st.session_state["last_mode"] = mode
    render_intensity_and_metrics(intensity)

elif run_final:
    sim = make_sim()
    # Need enough steps for the wave to travel to the screen AND average
    # an oscillating pattern for several wave periods afterwards.
    total_steps = max(n_frames * substeps, sim.warmup_steps + 800)
    with st.spinner(f"Running {total_steps} physics steps…"):
        sim.run_steps(total_steps)

    fig = plot_wave_field(sim, show_markers=show_markers)
    with field_box:
        st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    intensity = sim.get_screen_intensity()
    st.session_state["last_intensity"] = intensity
    st.session_state["last_mode"] = mode
    render_intensity_and_metrics(intensity)

elif "last_intensity" in st.session_state:
    with field_box:
        st.info("Press **Run Animation** to watch the wave evolve, or **Generate Final Pattern** for a fast static result.")
    render_intensity_and_metrics(st.session_state["last_intensity"])

else:
    with field_box:
        st.info(
            "Press **Run Animation** to watch the wave evolve in real time, or "
            "**Generate Final Pattern** for a fast static result if animation is "
            "slow on this machine."
        )