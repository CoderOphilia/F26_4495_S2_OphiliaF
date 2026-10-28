"""Lesson: create a reusable normal voltage signal generator.

Developed with AI assistance for the CSIS 4495 power quality project.
Amplitude is relative, and phase is measured in radians.
"""

import numpy as np


def generate_normal_signal(
    frequency_hz=60,
    sample_rate_hz=6000,
    duration_seconds=0.2,
    amplitude=1.0,
    phase=0.0,
):
    """Return matching arrays of timestamps and ideal sine-wave values."""
    number_of_samples = int(sample_rate_hz * duration_seconds)
    time = np.arange(number_of_samples) / sample_rate_hz

    voltage = amplitude * np.sin(
        2 * np.pi * frequency_hz * time + phase
    )

    return time, voltage


if __name__ == "__main__":
    time, voltage = generate_normal_signal()

    print("Number of measurements:", len(voltage))
    print("First five values:", voltage[:5])
