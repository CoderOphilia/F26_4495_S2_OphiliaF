"""Time-domain features for the power quality research project.

Developed with AI assistance for CSIS 4495.
"""

import numpy as np


def calculate_rms(voltage):
    """Return the root mean square of a nonempty, real voltage recording.

    Values use the recording's amplitude units. This is a descriptive feature,
    not a classification result.
    """
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

    # Scaling avoids overflow when large voltage values are squared.
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return 0.0
    return scale * float(np.sqrt(np.mean((values / scale) ** 2)))
