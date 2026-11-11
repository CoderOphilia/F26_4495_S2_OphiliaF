"""Time-domain features for the power quality research project.
"""

import numpy as np


FEATURE_VERSION = 1
SAMPLE_RATE_HZ = 6000
SAMPLE_COUNT = 1200
FEATURE_NAMES = (
    "rms", "peak_amplitude", "crest_factor", "cycle_rms_min_ratio",
    "cycle_rms_max_ratio", "cycle_rms_std_ratio", "dominant_frequency_hz",
    "third_harmonic_power_ratio", "fifth_harmonic_power_ratio",
    "other_ac_power_ratio",
)


def _validated_voltage(voltage):
    if np.iscomplexobj(voltage):
        raise ValueError("Voltage samples must be real numbers.")
    try:
        values = np.asarray(voltage, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("Voltage must contain numeric samples.") from None
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Voltage must be a nonempty one-dimensional recording.")
    if not np.isfinite(values).all():
        raise ValueError("Voltage samples must be finite numbers.")
    return values


def calculate_rms(voltage):
    """Return the root mean square of a nonempty, real voltage recording.

    Values use the recording's amplitude units. This is a descriptive feature,
    not a classification result.
    """
    values = _validated_voltage(voltage)

    # Scaling avoids overflow when large voltage values are squared.
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return 0.0
    return scale * float(np.sqrt(np.mean((values / scale) ** 2)))


def calculate_peak_amplitude(voltage):
    """Return the largest distance from zero among the voltage samples."""
    values = _validated_voltage(voltage)
    return float(np.max(np.abs(values)))


def extract_features(voltage, sample_rate_hz):
    """Measure amplitude variation and FFT band power in a voltage window.

    Cycle windows use the project's nominal 60 Hz frequency. A Hann window
    reduces FFT leakage. The band ratios are descriptive features, not THD.
    Training and inference must call this same function in the same order.
    """
    values = _validated_voltage(voltage)
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, (int, float))
        or not np.isfinite(sample_rate_hz)
        or sample_rate_hz <= 600
    ):
        raise ValueError("Sample rate must exceed 600 Hz to include the fifth harmonic.")
    cycle_samples = int(round(sample_rate_hz / 60))
    if values.size < 2 * cycle_samples:
        raise ValueError("At least two nominal 60 Hz cycles are required.")

    peak = calculate_peak_amplitude(values)
    rms = calculate_rms(values)
    if peak == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    scaled = values / peak
    scaled_rms = float(np.sqrt(np.mean(scaled ** 2)))

    # All full-cycle windows, including those between fixed block boundaries.
    squared_sum = np.concatenate(([0.0], np.cumsum(scaled ** 2)))
    cycle_power = (squared_sum[cycle_samples:] - squared_sum[:-cycle_samples]) / cycle_samples
    cycle_rms = np.sqrt(np.maximum(cycle_power, 0)) / scaled_rms

    centered = scaled - np.mean(scaled)
    spectrum_power = np.abs(np.fft.rfft(centered * np.hanning(values.size))) ** 2
    frequencies = np.fft.rfftfreq(values.size, d=1 / sample_rate_hz)
    fundamental_band = (frequencies >= 45) & (frequencies <= 75)
    third_band = (frequencies >= 165) & (frequencies <= 195)
    fifth_band = (frequencies >= 285) & (frequencies <= 315)
    ac_power = float(spectrum_power[1:].sum())
    dominant_frequency = 0.0
    third_ratio = 0.0
    fifth_ratio = 0.0
    other_ratio = 0.0
    if ac_power > 0:
        dominant_frequency = float(frequencies[1:][np.argmax(spectrum_power[1:])])
        third_ratio = float(spectrum_power[third_band].sum() / ac_power)
        fifth_ratio = float(spectrum_power[fifth_band].sum() / ac_power)
        selected = fundamental_band | third_band | fifth_band
        selected[0] = True
        other_ratio = float(spectrum_power[~selected].sum() / ac_power)

    return np.array([
        rms, peak, 1 / scaled_rms,
        float(cycle_rms.min()), float(cycle_rms.max()), float(cycle_rms.std()),
        dominant_frequency, third_ratio, fifth_ratio, other_ratio,
    ], dtype=np.float64)
