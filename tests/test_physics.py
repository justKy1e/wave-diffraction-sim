"""
Basic sanity tests for the wave-diffraction simulator.

Run with:

    pytest -q
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import (  # noqa: E402
    detect_intensity_peaks,
    percent_error,
    theoretical_fringe_spacing,
)
from wave_solver import WaveSimulation  # noqa: E402


def test_courant_number_is_stable():
    sim = WaveSimulation()
    assert sim.r <= 1.0 / math.sqrt(2.0)


def test_barrier_mask_has_openings_double():
    sim = WaveSimulation(mode="Double Slit", slit_width=10, slit_separation=45)
    col = sim.barrier_mask[sim.barrier_x, :]
    # The wall must not be fully closed.
    assert not col.all()
    assert col.any()


def test_single_slit_creates_one_opening():
    sim = WaveSimulation(mode="Single Slit", slit_width=10)
    col = sim.barrier_mask[sim.barrier_x, :]
    # Count contiguous False (open) regions.
    openings = 0
    in_gap = False
    for v in col:
        if not v and not in_gap:
            openings += 1
            in_gap = True
        elif v:
            in_gap = False
    assert openings == 1


def test_double_slit_creates_two_openings():
    sim = WaveSimulation(mode="Double Slit", slit_width=8, slit_separation=50)
    col = sim.barrier_mask[sim.barrier_x, :]
    openings = 0
    in_gap = False
    for v in col:
        if not v and not in_gap:
            openings += 1
            in_gap = True
        elif v:
            in_gap = False
    assert openings == 2


def test_intensity_array_length_matches_screen():
    sim = WaveSimulation()
    sim.run_steps(50)
    intensity = sim.get_screen_intensity()
    assert intensity.shape == (sim.ny,)


def test_theoretical_fringe_spacing_increases_with_wavelength():
    a = theoretical_fringe_spacing(10, 100, 40)
    b = theoretical_fringe_spacing(20, 100, 40)
    assert b > a


def test_theoretical_fringe_spacing_decreases_with_slit_separation():
    a = theoretical_fringe_spacing(15, 100, 30)
    b = theoretical_fringe_spacing(15, 100, 60)
    assert b < a


def test_percent_error_basic():
    assert percent_error(10.0, 10.0) == 0.0
    assert math.isclose(percent_error(11.0, 10.0), 10.0)
    assert math.isnan(percent_error(1.0, 0.0))


def test_simulation_does_not_blow_up():
    sim = WaveSimulation()
    sim.run_steps(400)
    assert np.all(np.isfinite(sim.u))
    assert np.max(np.abs(sim.u)) < 50.0  # comfortably bounded


def test_peak_detector_on_synthetic_pattern():
    y = np.arange(200)
    intensity = np.zeros(200)
    centers = [40, 80, 120, 160]
    for c in centers:
        intensity += np.exp(-((y - c) ** 2) / (2 * 4.0**2))
    peaks = detect_intensity_peaks(intensity)
    assert peaks.size == len(centers)
