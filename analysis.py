"""
analysis.py
-----------
Post-simulation analysis helpers: peak detection on the measured intensity
profile, an empirical fringe-spacing estimate, and the textbook far-field
double-slit prediction.

The far-field formula

    Delta y = lambda * L / d

is used ONLY here, for comparison. The simulation itself never uses it.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from scipy.signal import find_peaks  # type: ignore

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - SciPy is optional
    _HAS_SCIPY = False


def _fallback_find_peaks(y: np.ndarray, height: float, distance: int) -> np.ndarray:
    """Simple local-maximum peak finder used if SciPy is unavailable."""
    peaks: List[int] = []
    n = len(y)
    for i in range(1, n - 1):
        if y[i] <= height:
            continue
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            if peaks and (i - peaks[-1]) < distance:
                if y[i] > y[peaks[-1]]:
                    peaks[-1] = i
                continue
            peaks.append(i)
    return np.asarray(peaks, dtype=int)


def _smooth(y: np.ndarray, win: int) -> np.ndarray:
    """Simple symmetric boxcar smoothing. Used to wash out wave-scale ripples
    so the peak detector only locks onto true interference fringes."""
    win = max(1, int(win) | 1)  # force odd
    if win <= 1:
        return y
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(y, kernel, mode="same")


def detect_intensity_peaks(
    intensity: np.ndarray,
    min_height_frac: float = 0.15,
    min_distance: int = 8,
    smooth_window: int = 9,
) -> np.ndarray:
    """Return indices of significant peaks in the intensity profile.

    The profile is lightly smoothed first so that wave-scale ripples (which
    survive finite-time averaging) don't fool the detector into reporting
    spurious peaks every few cells.
    """
    intensity = np.asarray(intensity, dtype=float)
    if intensity.size == 0 or not np.any(intensity > 0):
        return np.asarray([], dtype=int)
    y = _smooth(intensity, smooth_window)
    height = min_height_frac * float(np.max(y))
    if _HAS_SCIPY:
        peaks, _ = find_peaks(y, height=height, distance=min_distance)
        return np.asarray(peaks, dtype=int)
    return _fallback_find_peaks(y, height=height, distance=min_distance)


def estimate_fringe_spacing(peaks: np.ndarray) -> float:
    """Return the mean spacing between adjacent peaks (NaN if fewer than 2)."""
    peaks = np.asarray(peaks)
    if peaks.size < 2:
        return float("nan")
    diffs = np.diff(np.sort(peaks))
    return float(np.mean(diffs))


def theoretical_fringe_spacing(
    wavelength: float, screen_distance: float, slit_separation: float
) -> float:
    """Far-field double-slit fringe spacing Delta y = lambda * L / d."""
    if slit_separation <= 0:
        return float("nan")
    return float(wavelength) * float(screen_distance) / float(slit_separation)


def theoretical_single_slit_width(
    wavelength: float, screen_distance: float, slit_width: float
) -> float:
    """Far-field single-slit central-maximum FWHM (full width at half max).

    The intensity pattern from a single slit is sinc^2(π w sinθ / λ). Setting
    sinc^2 = 0.5 gives π w sinθ / λ ≈ 1.392, so sinθ ≈ 0.443 λ / w, and the
    full width at half max on the screen (small-angle) is

        FWHM ≈ 2 · 0.443 · λ L / w  =  0.886 · λ L / w.
    """
    if slit_width <= 0:
        return float("nan")
    return 0.886 * float(wavelength) * float(screen_distance) / float(slit_width)


def measure_central_peak_width(intensity: np.ndarray, frac: float = 0.5) -> float:
    """Return the full width of the central maximum of an intensity profile
    measured at a given fraction of the peak height (default 0.5 = FWHM).
    Returns NaN if the profile is flat or has no clear central peak."""
    intensity = np.asarray(intensity, dtype=float)
    if intensity.size < 3 or not np.any(intensity > 0):
        return float("nan")
    smoothed = _smooth(intensity, max(5, intensity.size // 40 | 1))
    peak_idx = int(np.argmax(smoothed))
    threshold = frac * smoothed[peak_idx]
    # Walk left and right from the peak until we drop below threshold.
    left = peak_idx
    while left > 0 and smoothed[left] >= threshold:
        left -= 1
    right = peak_idx
    n = smoothed.size
    while right < n - 1 and smoothed[right] >= threshold:
        right += 1
    width = float(right - left)
    return width if width > 0 else float("nan")


def percent_error(measured: float, predicted: float) -> float:
    """Relative error in percent. Returns NaN if either value is invalid."""
    if (
        predicted is None
        or measured is None
        or not np.isfinite(measured)
        or not np.isfinite(predicted)
        or predicted == 0.0
    ):
        return float("nan")
    return 100.0 * abs(measured - predicted) / abs(predicted)
