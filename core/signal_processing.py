"""
Signal Processing Module
========================
Temporal smoothing (Savitzky-Golay, Butterworth) and frequency-domain
spectral decomposition (FFT, PSD) for tremor separation and motor assessment.
"""

from typing import Optional, Tuple, Union
import numpy as np
from scipy import signal


class SignalProcessor:
    """
    Mathematical and digital signal processing core for NeuroMotion.

    Provides Savitzky-Golay filtering for edge-safe jitter suppression,
    along with detrended windowed FFT/PSD estimation for tremor analysis.
    """

    @staticmethod
    def smooth_signal(
        data: np.ndarray,
        window_length: int = 7,
        polyorder: int = 2,
    ) -> np.ndarray:
        """
        Apply Savitzky-Golay filtering to eliminate tracking jitter without flattening
        sharp physiological peaks (e.g. finger tap excursions).

        Dynamically adapts window length for edge cases where the input series has
        fewer samples than window_length.

        Args:
            data: 1D or 2D array of kinematic time-series data.
            window_length: Ideal filter window size in samples (must be odd).
            polyorder: Degree of the fitting polynomial (default: 2).

        Returns:
            Smoothed array with identical shape and dtype float64.
        """
        arr = np.asarray(data, dtype=np.float64)
        n_samples = arr.shape[0]

        # Edge cases: empty array or insufficient data to fit polynomial
        if n_samples == 0 or n_samples <= polyorder:
            return arr.copy()

        # Savitzky-Golay requires an odd window length > polyorder
        effective_window = min(window_length, n_samples)
        if effective_window % 2 == 0:
            effective_window -= 1

        # If after reducing the window length is not strictly greater than polyorder,
        # fallback to the largest odd integer <= n_samples that is > polyorder
        if effective_window <= polyorder:
            # Cannot safely fit polyorder; return un-smoothed data
            return arr.copy()

        axis = 0 if arr.ndim > 1 else -1
        try:
            smoothed = signal.savgol_filter(
                arr,
                window_length=effective_window,
                polyorder=polyorder,
                axis=axis,
            )
            return smoothed
        except Exception:
            # Fallback to copy if filtering fails on ill-conditioned data
            return arr.copy()

    @staticmethod
    def compute_fft_spectrum(
        time_series: np.ndarray,
        timestamps: np.ndarray,
        tremor_band: Tuple[float, float] = (3.0, 12.0),
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Compute one-sided Power Spectral Density (PSD) via detrended, windowed rFFT.

        Extracts the dominant tremor frequency and peak power within the clinical
        tremor band (3.0 Hz to 12.0 Hz).

        Args:
            time_series: 1D kinematic signal array (e.g. distance or displacement).
            timestamps: Strictly increasing or relative timestamp array in seconds.
            tremor_band: Lower and upper frequency bounds for tremor search in Hz.

        Returns:
            Tuple of:
                - freqs: 1D array of frequency bins in Hz (from rfftfreq).
                - psd: 1D array of power spectral density values.
                - peak_freq: Dominant frequency within tremor_band in Hz (0.0 if none).
                - peak_power: Peak power magnitude within tremor_band (0.0 if none).
        """
        y = np.asarray(time_series, dtype=np.float64).flatten()
        t = np.asarray(timestamps, dtype=np.float64).flatten()
        n = len(y)

        # Require at least 4 samples for meaningful spectral decomposition
        if n < 4 or len(t) < 4:
            return (
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                0.0,
                0.0,
            )

        # 1. Calculate actual sampling frequency (Fs) from timestamps
        dt = np.diff(t)
        # Filter out non-positive dt to prevent division by zero
        dt_positive = dt[dt > 0]
        if len(dt_positive) == 0:
            fs = 30.0  # fallback default nominal frame rate
        else:
            mean_dt = float(np.mean(dt_positive))
            fs = 1.0 / mean_dt if mean_dt > 0 else 30.0

        # Guard against invalid or unrealistic sampling rate
        if fs <= 0.0 or not np.isfinite(fs):
            fs = 30.0

        # 2. Detrend the signal to eliminate low-frequency DC baseline drift
        try:
            detrended = signal.detrend(y, type="linear")
        except Exception:
            detrended = y - np.mean(y)

        # 3. Apply Hanning window to mitigate spectral leakage
        hanning_win = np.hanning(n)
        windowed_signal = detrended * hanning_win

        # 4. Compute real Fast Fourier Transform (rFFT)
        rfft_vals = np.fft.rfft(windowed_signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)

        # 5. Compute one-sided Power Spectral Density (PSD)
        win_sum_sq = np.sum(hanning_win**2)
        if win_sum_sq > 0 and fs > 0:
            psd = (2.0 / (fs * win_sum_sq)) * (np.abs(rfft_vals) ** 2)
            # Correct DC and Nyquist components
            psd[0] /= 2.0
            if n % 2 == 0 and len(psd) > 1:
                psd[-1] /= 2.0
        else:
            psd = (np.abs(rfft_vals) ** 2) / float(n)

        # 6. Extract dominant frequency & peak power in tremor band (3.0 - 12.0 Hz)
        low_f, high_f = tremor_band
        band_mask = (freqs >= low_f) & (freqs <= high_f)

        if np.any(band_mask):
            band_freqs = freqs[band_mask]
            band_psd = psd[band_mask]
            peak_idx = int(np.argmax(band_psd))
            peak_freq = float(band_freqs[peak_idx])
            peak_power = float(band_psd[peak_idx])
        else:
            peak_freq = 0.0
            peak_power = 0.0

        return freqs, psd, peak_freq, peak_power


class SignalFilter:
    """
    Applies real-time filtering and spectral decomposition to landmark trajectories.
    Maintained for backward compatibility and streaming low-pass filtering.

    Attributes:
        sampling_rate (float): Video capture frame rate in Hz.
        cutoff_freq (float): Low-pass cutoff frequency in Hz.
    """

    def __init__(self, sampling_rate: float = 30.0, cutoff_freq: float = 5.0) -> None:
        """
        Initialize the SignalFilter instance.

        Args:
            sampling_rate: Video sampling rate in Hz.
            cutoff_freq: Primary filter cutoff frequency.
        """
        self.sampling_rate = float(sampling_rate)
        self.cutoff_freq = float(cutoff_freq)

    def filter_trajectory(
        self, signal_data: np.ndarray, window_length: int = 7, polyorder: int = 2
    ) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing to time-series keypoint coordinates.

        Args:
            signal_data: 1D or 2D array of coordinate signals over time.
            window_length: Window size for Savitzky-Golay filter.
            polyorder: Polynomial order for smoothing.

        Returns:
            Filtered signal array.
        """
        return SignalProcessor.smooth_signal(
            signal_data, window_length=window_length, polyorder=polyorder
        )

    def lowpass_butterworth(
        self, signal_data: np.ndarray, order: int = 4
    ) -> np.ndarray:
        """
        Apply a zero-phase forward-backward Butterworth low-pass filter.

        Args:
            signal_data: 1D array of signal samples.
            order: Filter order (default: 4).

        Returns:
            Filtered array.
        """
        arr = np.asarray(signal_data, dtype=np.float64)
        if len(arr) < 3 * order:
            return arr.copy()

        nyq = 0.5 * self.sampling_rate
        normal_cutoff = min(self.cutoff_freq / nyq, 0.99)
        b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
        return signal.filtfilt(b, a, arr)
