"""Compare frozen clean-trained classifiers on the held-out noise conditions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from feature_extraction import FEATURE_NAMES, FEATURE_VERSION, extract_features
from train_models import DEFAULT_DATASET, DEFAULT_OUTPUT, SAMPLE_RATE_HZ, evaluation


NOISE_CONDITIONS = (None, 40, 30, 20)


def add_noise(voltage, snr_db, rng):
    """Add zero-mean Gaussian noise scaled to the specified measured SNR."""
    values = np.asarray(voltage, dtype=np.float64)
    noise = rng.normal(size=values.size)
    noise -= noise.mean()
    signal_rms = float(np.sqrt(np.mean(values ** 2)))
    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    return values + noise * signal_rms / (noise_rms * 10 ** (snr_db / 20))


def run_noise_experiment(dataset_dir=DEFAULT_DATASET, model_dir=DEFAULT_OUTPUT, output_dir=None, seed=2026):
    dataset_dir = Path(dataset_dir)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir) if output_dir is not None else model_dir / "noise_experiment"
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Experiment output must be empty. Choose a new --output folder.")
    bundle_path = model_dir / "classifiers.joblib"
    bundle = joblib.load(bundle_path)
    archive_path = dataset_dir / "signals.npz"
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if archive_hash != bundle["dataset_sha256"]:
        raise ValueError("Use the same dataset archive that produced these trained models.")
    if (
        bundle["feature_version"] != FEATURE_VERSION
        or bundle["feature_names"] != list(FEATURE_NAMES)
        or bundle["sklearn_version"] != sklearn.__version__
    ):
        raise ValueError("Models must match the current feature and scikit-learn versions.")
    with np.load(archive_path, allow_pickle=False) as archive:
        mask = archive["splits"] == "test"
        signals = archive["voltage_relative"][mask]
        labels = archive["labels"][mask]
        signal_ids = archive["signal_ids"][mask]
    if len(signals) == 0:
        raise ValueError("The dataset must contain a held-out test partition.")
    report = {
        "dataset_sha256": archive_hash,
        "model_bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "seed": seed, "numpy_version": np.__version__,
        "test_signal_ids": signal_ids.tolist(),
        "class_order": bundle["classes"],
        "method": "Frozen clean-trained models; same held-out IDs at every SNR. Per-record Gaussian noise is mean-centered and scaled to exact measured SNR. The same noise direction is reused across SNR levels. No fitting or model selection uses these results.",
        "conditions": {},
    }
    summary = []
    for snr in NOISE_CONDITIONS:
        condition = "clean" if snr is None else f"{snr}_db"
        measurements = signals
        if snr is not None:
            measurements = []
            for signal_id, values in zip(signal_ids, signals):
                # Stable IDs keep a noisy recording attached to its test parent.
                digest = hashlib.sha256(f"{seed}:{signal_id}".encode("utf-8")).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
                measurements.append(add_noise(values, snr, rng))
        features = np.stack([extract_features(values, SAMPLE_RATE_HZ) for values in measurements])
        report["conditions"][condition] = {}
        for name, model in bundle["models"].items():
            scores = evaluation(model, features, labels)
            report["conditions"][condition][name] = scores
            summary.append({
                "condition": condition, "model": name, "recordings": len(labels),
                "accuracy": scores["accuracy"], "macro_f1": scores["macro_f1"],
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    conditions = list(report["conditions"])
    for axis, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro F1")):
        for name, label in (("random_forest", "Random Forest"), ("svm", "SVM")):
            axis.plot(range(4), [report["conditions"][key][name][metric] for key in conditions], marker="o", label=label)
        axis.set_xticks(range(4), ["Clean", "40 dB", "30 dB", "20 dB"])
        axis.set_ylim(0, 1.05)
        axis.set_ylabel(title)
        axis.set_xlabel("Added noise condition")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"Noise robustness on {len(signals)} held-out synthetic recordings")
    figure.savefig(output_dir / "noise_comparison.png", dpi=160)
    plt.close(figure)
    lines = [
        "# Noise robustness experiment", "",
        f"Compared the frozen clean-trained models on the same {len(signals)} held-out recordings.",
        "Noise is added only to test copies. Models and preprocessing are not refitted.", "",
        "| Condition | Model | Accuracy | Macro F1 |", "| --- | --- | ---: | ---: |",
    ]
    for row in summary:
        lines.append(f"| {row['condition']} | {row['model']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |")
        print(f"{row['condition']}, {row['model']}: accuracy={row['accuracy']:.4f}, macro F1={row['macro_f1']:.4f}")
    lines.extend([
        "", "![Accuracy and macro F1 by added noise](noise_comparison.png)", "",
        "Gaussian noise is seeded per recording and mean-centered, then scaled to exact",
        "measured SNR. The browser uses a fresh random Gaussian draw at the target",
        "expected noise power, so individual browser predictions may differ.", "",
        "These results describe one noise realization per recording and this simplified",
        "generator. They do not establish field accuracy or cover non-Gaussian noise.",
        "Full confusion matrices, per-class scores, IDs, and checksums are in results.json.", "",
    ])
    (output_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--models", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    try:
        run_noise_experiment(args.dataset, args.models, args.output, args.seed)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
