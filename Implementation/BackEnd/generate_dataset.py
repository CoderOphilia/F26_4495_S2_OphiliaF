"""Build a reproducible, labelled dataset for the power quality project.

Developed with AI assistance for CSIS 4495. Uses synthetic mathematical
signals only; it does not import the unverified RawData reference files.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from signal_generator import SIGNAL_TYPES, generate_signal


DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "Data" / "Synthetic"
PARAMETER_FIELDS = (
    "frequency_hz", "sample_rate_hz", "duration_seconds", "amplitude", "phase",
    "event_start_seconds", "event_end_seconds", "event_multiplier",
    "third_harmonic_ratio", "fifth_harmonic_ratio",
    "third_harmonic_phase", "fifth_harmonic_phase",
)


def sample_parameters(signal_type, rng):
    """Vary the recording instead of repeatedly copying one fixed example."""
    parameters = {
        "frequency_hz": float(rng.uniform(59.0, 61.0)),
        "sample_rate_hz": 6000,
        "duration_seconds": 0.2,
        "amplitude": float(rng.uniform(0.9, 1.1)),
        "phase": float(rng.uniform(0, 2 * np.pi)),
    }

    if signal_type in ("sag", "swell", "interruption"):
        start = float(rng.uniform(0.03, 0.07))
        length = float(rng.uniform(0.04, 0.09))
        multiplier_ranges = {
            "sag": (0.2, 0.8),
            "swell": (1.15, 1.8),
            "interruption": (0.0, 0.03),
        }
        low, high = multiplier_ranges[signal_type]
        parameters.update({
            "event_start_seconds": start,
            "event_end_seconds": start + length,
            "event_multiplier": float(rng.uniform(low, high)),
        })

    if signal_type == "harmonic_distortion":
        parameters.update({
            "third_harmonic_ratio": float(rng.uniform(0.1, 0.3)),
            "fifth_harmonic_ratio": float(rng.uniform(0.05, 0.2)),
            "third_harmonic_phase": float(rng.uniform(0, 2 * np.pi)),
            "fifth_harmonic_phase": float(rng.uniform(0, 2 * np.pi)),
        })
    return parameters


def make_splits(samples_per_class, rng):
    """Assign whole recordings to approximately 70/15/15 percent groups."""
    validation_count = max(1, int(samples_per_class * 0.15))
    test_count = max(1, int(samples_per_class * 0.15))
    train_count = samples_per_class - validation_count - test_count
    splits = np.array(
        ["train"] * train_count
        + ["validation"] * validation_count
        + ["test"] * test_count
    )
    rng.shuffle(splits)
    return splits


def build_dataset(output_dir=DEFAULT_OUTPUT, samples_per_class=200, seed=42):
    """Write signals, labels, partitions, parameters, and five example CSVs.

    Existing nonempty folders are refused so reruns cannot overwrite work.
    The same seed and sample count reproduce the signal values and metadata.
    """
    if isinstance(samples_per_class, bool) or not isinstance(samples_per_class, int) or samples_per_class < 20:
        raise ValueError("Use an integer of at least 20 recordings per class.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("The random seed must be a nonnegative integer.")

    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Output already exists and is not empty. Choose a new --output folder.")

    # Keep generation and partition assignment on independent random streams.
    generation_sequence, partition_sequence = np.random.SeedSequence(seed).spawn(2)
    sample_sequences = generation_sequence.spawn(samples_per_class * len(SIGNAL_TYPES))
    partition_rng = np.random.default_rng(partition_sequence)

    count = samples_per_class * len(SIGNAL_TYPES)
    signals = np.empty((count, 1200), dtype=np.float64)
    labels = []
    signal_ids = []
    splits = []
    metadata = []
    example_indices = {}
    time = None

    for signal_type in SIGNAL_TYPES:
        class_splits = make_splits(samples_per_class, partition_rng)
        for class_index in range(samples_per_class):
            index = len(labels)
            sample_seed = int(sample_sequences[index].generate_state(1)[0])
            rng = np.random.default_rng(sample_seed)
            parameters = sample_parameters(signal_type, rng)
            time, values = generate_signal(signal_type, **parameters)
            signals[index] = values

            signal_id = f"seed{seed}_{signal_type}_{class_index:05d}"
            split = str(class_splits[class_index])
            labels.append(signal_type)
            signal_ids.append(signal_id)
            splits.append(split)
            metadata.append({
                "signal_id": signal_id, "label": signal_type, "split": split,
                "sample_seed": sample_seed, **parameters,
            })
            # Export a training example, leaving validation and test examples
            # untouched for their later model-selection/evaluation purposes.
            if split == "train" and signal_type not in example_indices:
                example_indices[signal_type] = index

    # Shuffle the dataset rows so class order cannot masquerade as a feature.
    order = partition_rng.permutation(count)
    metadata = [metadata[int(index)] for index in order]
    output_dir.mkdir(parents=True, exist_ok=True)
    example_dir = output_dir / "examples"
    example_dir.mkdir()

    archive = output_dir / "signals.npz"
    np.savez_compressed(
        archive,
        time_seconds=time,
        voltage_relative=signals[order],
        labels=np.asarray(labels)[order],
        signal_ids=np.asarray(signal_ids)[order],
        splits=np.asarray(splits)[order],
    )

    fields = ("signal_id", "label", "split", "sample_seed", *PARAMETER_FIELDS)
    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata)

    examples = {}
    for signal_type, index in example_indices.items():
        name = f"{signal_type}_signal.csv"
        np.savetxt(
            example_dir / name,
            np.column_stack((time, signals[index])),
            delimiter=",", header="time_seconds,voltage_relative", comments="",
            fmt="%.12g",
        )
        examples[signal_type] = {
            "path": f"examples/{name}", "signal_id": signal_ids[index], "split": "train",
        }

    split_counts = {name: splits.count(name) for name in ("train", "validation", "test")}
    manifest = {
        "schema_version": 1,
        "generator_version": "synthetic_v1",
        "seed": seed,
        "numpy_version": np.__version__,
        "source": "Synthetic mathematical waveforms; no imported or hardware measurements.",
        "samples_per_class": samples_per_class,
        "total_recordings": count,
        "measurements_per_recording": signals.shape[1],
        "sample_rate_hz": 6000,
        "duration_seconds": 0.2,
        "nominal_frequency_hz": 60,
        "added_noise": "none",
        "class_counts": {name: samples_per_class for name in SIGNAL_TYPES},
        "split_counts": split_counts,
        "parameter_ranges": {
            "frequency_hz": [59.0, 61.0], "relative_amplitude": [0.9, 1.1],
            "phase_radians": [0.0, float(2 * np.pi)],
            "event_start_seconds": [0.03, 0.07], "event_duration_seconds": [0.04, 0.09],
            "sag_multiplier": [0.2, 0.8], "swell_multiplier": [1.15, 1.8],
            "interruption_multiplier": [0.0, 0.03],
            "third_harmonic_ratio": [0.1, 0.3], "fifth_harmonic_ratio": [0.05, 0.2],
            "harmonic_phase_radians": [0.0, float(2 * np.pi)],
        },
        "signals_npz_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "examples": examples,
        "limitations": [
            "Simplified single events with abrupt envelope boundaries and no background noise.",
            "No combined events, hardware validation, or electrical-standard certification.",
            "Fixed 0.2-second windows and 6 kHz sampling around a 60 Hz fundamental.",
            "Performance on this generator does not establish real-device accuracy.",
        ],
        "evaluation_rule": "Keep every noisy copy with its original signal_id and saved split. Do not train on labels or metadata parameters as features.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        manifest = build_dataset(args.output, args.samples_per_class, args.seed)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"Created {manifest['total_recordings']} recordings ({manifest['samples_per_class']} per class).")
    print(f"Partitions: {manifest['split_counts']}")
    print(f"Saved to: {args.output.resolve()}")
    print("Upload a CSV from the examples folder to preview a class in the app.")


if __name__ == "__main__":
    main()
