"""Generate simplified, single-disturbance voltage recordings.

Developed with AI assistance for the CSIS 4495 power quality project.
Amplitude is relative, and phase is measured in radians.
"""

import numpy as np


SIGNAL_TYPES = (
    "normal",
    "sag",
    "swell",
    "interruption",
    "harmonic_distortion",
)


def generate_normal_signal(
    frequency_hz=60,
    sample_rate_hz=6000,
    duration_seconds=0.2,
    amplitude=1.0,
    phase=0.0,
):
    """Return matching arrays of timestamps and ideal sine-wave values."""
    values = (frequency_hz, sample_rate_hz, duration_seconds, amplitude, phase)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Signal parameters must be finite numbers.")
    if frequency_hz <= 0 or sample_rate_hz <= 0 or duration_seconds <= 0:
        raise ValueError("Frequency, sample rate, and duration must be positive.")
    if amplitude <= 0:
        raise ValueError("The reference amplitude must be positive.")
    if frequency_hz >= sample_rate_hz / 2:
        raise ValueError("Sample rate must exceed twice the signal frequency.")

    number_of_samples = int(sample_rate_hz * duration_seconds)
    if number_of_samples < 4:
        raise ValueError("A recording must contain at least four samples.")
    time = np.arange(number_of_samples) / sample_rate_hz

    voltage = amplitude * np.sin(
        2 * np.pi * frequency_hz * time + phase
    )

    return time, voltage


def generate_signal(
    signal_type,
    *,
    frequency_hz=60,
    sample_rate_hz=6000,
    duration_seconds=0.2,
    amplitude=1.0,
    phase=0.0,
    event_start_seconds=0.05,
    event_end_seconds=0.15,
    event_multiplier=None,
    third_harmonic_ratio=0.2,
    fifth_harmonic_ratio=0.1,
    third_harmonic_phase=0.0,
    fifth_harmonic_phase=0.0,
):
    """Return timestamps and one of the five supported signal examples.

    An event multiplier is the fraction of the reference amplitude retained
    during the event. Harmonic ratios are relative to the fundamental peak.
    These mathematical examples are not certified electrical measurements.
    """
    if signal_type not in SIGNAL_TYPES:
        raise ValueError(f"Unsupported signal type: {signal_type}")

    time, voltage = generate_normal_signal(
        frequency_hz, sample_rate_hz, duration_seconds, amplitude, phase
    )

    if signal_type == "normal":
        return time, voltage

    if signal_type == "harmonic_distortion":
        harmonic_parameters = (
            third_harmonic_ratio, fifth_harmonic_ratio,
            third_harmonic_phase, fifth_harmonic_phase,
        )
        if not all(np.isfinite(value) for value in harmonic_parameters):
            raise ValueError("Harmonic parameters must be finite numbers.")
        if third_harmonic_ratio <= 0 or fifth_harmonic_ratio <= 0:
            raise ValueError("Both harmonic amplitude ratios must be positive.")
        if 5 * frequency_hz >= sample_rate_hz / 2:
            raise ValueError("Sample rate must exceed twice the fifth harmonic.")
        voltage += third_harmonic_ratio * amplitude * np.sin(
            2 * np.pi * 3 * frequency_hz * time + third_harmonic_phase
        )
        voltage += fifth_harmonic_ratio * amplitude * np.sin(
            2 * np.pi * 5 * frequency_hz * time + fifth_harmonic_phase
        )
        return time, voltage

    if not np.isfinite(event_start_seconds) or not np.isfinite(event_end_seconds):
        raise ValueError("Event times must be finite numbers.")
    if not 0 < event_start_seconds < event_end_seconds < duration_seconds:
        raise ValueError("The event must begin and end inside the recording.")

    if event_multiplier is None:
        event_multiplier = {"sag": 0.5, "swell": 1.5, "interruption": 0.0}[signal_type]
    if not np.isfinite(event_multiplier):
        raise ValueError("The event multiplier must be finite.")
    valid_multiplier = {
        "sag": 0 < event_multiplier < 1,
        "swell": event_multiplier > 1,
        "interruption": 0 <= event_multiplier <= 0.05,
    }
    if not valid_multiplier[signal_type]:
        raise ValueError(f"Invalid amplitude multiplier for {signal_type}.")

    event = (time >= event_start_seconds) & (time < event_end_seconds)
    if not event.any() or not (time < event_start_seconds).any() or not (time >= event_end_seconds).any():
        raise ValueError("The recording must sample the event and both sides of it.")
    voltage[event] *= event_multiplier
    return time, voltage


if __name__ == "__main__":
    time, voltage = generate_normal_signal()

    print("Number of measurements:", len(voltage))
    print("First five values:", voltage[:5])
