"""Compare generated waveforms with supplied reference CSVs, without model fitting."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from signal_generator import generate_signal


ROOT = Path(__file__).resolve().parent.parent
REFERENCE_FILES = {
    "normal": "Pure_Sinusoidal.csv", "sag": "Sag.csv", "swell": "Swell.csv",
    "interruption": "Interruption.csv", "harmonic_distortion": "Harmonics.csv",
}


def validate_references(source=None, output=None):
    if source is None:
        # The standalone workspace and full repository use different layouts.
        source = ROOT / "RawData" if (ROOT / "RawData").is_dir() else ROOT.parent / "RawData"
    source = Path(source)
    output = Path(output) if output else ROOT / "Research" / "Reference"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Choose an empty reference output directory.")
    report = {
        "method": "Shape, finiteness and checksum audit; first row of each supplied reference CSV plotted against a project example at the reference metadata settings: 50 Hz, 5 kHz, 20 ms. Parameters and phase are not fitted to the reference rows. This is a qualitative shape comparison, not an accuracy evaluation.",
        "provenance": "Supplied RawData/Details.txt describes synthetic 17-class SEED-style recordings. The candidate Kaggle source matches the metadata, but the original download, licence and byte identity have not been verified. These files are not claimed as hardware measurements or IEEE DataPort data.",
        "candidate_source": "https://www.kaggle.com/datasets/sumairaziz/seed-power-quality-disturbance-dataset",
        "files": {},
    }
    figure, axes = plt.subplots(5, 2, figsize=(11, 11), layout="constrained")
    for row, (label, filename) in enumerate(REFERENCE_FILES.items()):
        path = source / filename
        reference = np.loadtxt(path, delimiter=",")
        if reference.shape != (1000, 100) or not np.isfinite(reference).all():
            raise ValueError(f"{filename}: expected a finite 1000 by 100 matrix.")
        time, generated = generate_signal(
            label, frequency_hz=50, sample_rate_hz=5000, duration_seconds=0.02,
            event_start_seconds=0.005, event_end_seconds=0.015,
        )
        report["files"][filename] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": list(reference.shape), "finite": True,
            "minimum": float(reference.min()), "maximum": float(reference.max()),
            "displayed_row": 0,
        }
        for col, values in enumerate((generated, reference[0])):
            axes[row, col].plot(time * 1000, values, color="#365477" if col == 0 else "#91605B")
            axes[row, col].set(xlabel="Time (ms)", ylabel="Relative voltage", ylim=(-2, 2),
                               title=f"{label.replace('_', ' ').title()} - {'project example' if col == 0 else 'supplied row 1'}")
            axes[row, col].grid(alpha=0.2)
    figure.suptitle("Qualitative waveform comparison at matched reference sampling settings")
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "reference_shapes.png", dpi=160)
    plt.close(figure)
    (output / "reference_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        validate_references(args.source, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
