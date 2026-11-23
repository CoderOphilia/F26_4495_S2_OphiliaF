"""Evaluate feature groups, repeated noise, and a second synthetic dataset seed."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from evaluate_noise import add_noise
from feature_extraction import FEATURE_NAMES, SAMPLE_COUNT, SAMPLE_RATE_HZ, extract_features
from generate_dataset import build_dataset
from signal_generator import SIGNAL_TYPES
from train_models import DEFAULT_DATASET, ROOT, evaluation, model_candidates


FEATURE_GROUPS = {
    "time": list(range(6)), "frequency": list(range(6, 10)),
    "combined": list(range(len(FEATURE_NAMES))),
}
CONDITIONS = (None, 40, 35, 30, 25, 20)
NOISE_SEEDS = (2026, 2027, 2028, 2029, 2030)


def load_archive(folder):
    folder = Path(folder)
    archive_path = folder / "signals.npz"
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if digest != manifest["signals_npz_sha256"]:
        raise ValueError("Dataset checksum does not match its manifest.")
    with np.load(archive_path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    signals = data["voltage_relative"]
    if signals.ndim != 2 or signals.shape[1] != SAMPLE_COUNT or not np.isfinite(signals).all():
        raise ValueError("Expected finite 1,200-sample recordings.")
    count = len(signals)
    if any(data[key].shape != (count,) for key in ("labels", "splits", "signal_ids")):
        raise ValueError("Dataset arrays must describe the same recordings.")
    if len(set(data["signal_ids"])) != count:
        raise ValueError("Recording IDs must be unique.")
    if set(data["splits"]) != {"train", "validation", "test"}:
        raise ValueError("Expected the saved train, validation, and test partitions.")
    for partition in ("train", "validation", "test"):
        if set(data["labels"][data["splits"] == partition]) != set(SIGNAL_TYPES):
            raise ValueError("Every partition must include all five classes.")
    times = data["time_seconds"]
    if times.shape != (SAMPLE_COUNT,) or not np.allclose(np.diff(times), 1 / SAMPLE_RATE_HZ, rtol=0.001, atol=1e-10):
        raise ValueError("Expected uniform 6,000 Hz timestamps.")
    return data, manifest, digest


def noisy_features(signals, ids, snr, seed):
    rows = []
    for values, signal_id in zip(signals, ids):
        if snr is not None:
            digest = hashlib.sha256(f"{seed}:{signal_id}".encode("utf-8")).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            values = add_noise(values, snr, rng)
        rows.append(extract_features(values, SAMPLE_RATE_HZ))
    return np.stack(rows)


def summarize_runs(runs):
    return {
        "accuracy_mean": float(np.mean([run["accuracy"] for run in runs])),
        "accuracy_std": float(np.std([run["accuracy"] for run in runs])),
        "macro_f1_mean": float(np.mean([run["macro_f1"] for run in runs])),
        "macro_f1_std": float(np.std([run["macro_f1"] for run in runs])),
        "per_class_recall": {
            label: float(np.mean([run["classification_report"][label]["recall"] for run in runs]))
            for label in SIGNAL_TYPES
        },
        "runs": runs,
    }


def run_validation(dataset=DEFAULT_DATASET, confirmation=None, output=None):
    confirmation = Path(confirmation) if confirmation else ROOT / "Data" / "Synthetic_seed43"
    output = Path(output) if output else ROOT / "Research" / "Results"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Research output must be empty. Choose a new --output folder.")
    if not confirmation.exists():
        build_dataset(confirmation, samples_per_class=200, seed=43)
    primary, primary_manifest, primary_hash = load_archive(dataset)
    confirm, confirm_manifest, confirm_hash = load_archive(confirmation)
    if primary_manifest["seed"] == confirm_manifest["seed"]:
        raise ValueError("The confirmation dataset must use a different generator seed.")
    if set(primary["signal_ids"]) & set(confirm["signal_ids"]):
        raise ValueError("Confirmation recording IDs must differ from the primary dataset.")
    primary_features = noisy_features(primary["voltage_relative"], primary["signal_ids"], None, 2026)
    train = primary["splits"] == "train"
    validation = primary["splits"] == "validation"
    models = {}
    selection = {}
    for group, columns in FEATURE_GROUPS.items():
        models[group] = {}
        selection[group] = {}
        for name, options in model_candidates().items():
            scores = []
            for candidate in options:
                candidate.fit(primary_features[train][:, columns], primary["labels"][train])
                scores.append(evaluation(candidate, primary_features[validation][:, columns], primary["labels"][validation]))
            chosen = max(range(len(options)), key=lambda index: scores[index]["macro_f1"])
            models[group][name] = options[chosen]
            selection[group][name] = {
                "candidates": [str(candidate) for candidate in options],
                "validation_scores": scores, "selected_candidate": chosen,
            }
    report = {
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "feature_groups": {group: [FEATURE_NAMES[index] for index in columns] for group, columns in FEATURE_GROUPS.items()},
        "noise_seeds": list(NOISE_SEEDS), "class_order": list(SIGNAL_TYPES),
        "method": "Exploratory feature ablation after the baseline experiment. Every estimator and SVM scaler fits only the primary training partition; settings are chosen using primary validation macro F1. Models remain frozen for both saved test partitions and every noise condition. Noise direction is shared across SNR levels within a recording and seed. Error bars show noise-realization standard deviation, not sampling confidence intervals.",
        "limitations": "Both datasets come from the same simplified generator. The second seed checks within-generator generalization; it is not hardware or external-dataset validation. The original test set has already been examined in the baseline study. No models are selected using these test results.",
        "selection": selection,
        "train_ids": primary["signal_ids"][train].tolist(),
        "validation_ids": primary["signal_ids"][validation].tolist(),
        "studies": {},
    }
    summary = []
    for study, data, manifest, checksum in (
        ("primary", primary, primary_manifest, primary_hash),
        ("confirmation", confirm, confirm_manifest, confirm_hash),
    ):
        test = data["splits"] == "test"
        signals, ids, labels = (data[key][test] for key in ("voltage_relative", "signal_ids", "labels"))
        study_report = {"generator_seed": manifest["seed"], "dataset_sha256": checksum,
                        "test_ids": ids.tolist(), "test_recordings": len(labels), "conditions": {}}
        for snr in CONDITIONS:
            condition = "clean" if snr is None else f"{snr}_db"
            runs = {group: {name: [] for name in options} for group, options in models.items()}
            seeds = NOISE_SEEDS[:1] if snr is None else NOISE_SEEDS
            for seed in seeds:
                features = noisy_features(signals, ids, snr, seed)
                for group, options in models.items():
                    for name, model in options.items():
                        measured = evaluation(model, features[:, FEATURE_GROUPS[group]], labels)
                        runs[group][name].append({"noise_seed": seed, **measured})
            study_report["conditions"][condition] = {}
            for group, options in runs.items():
                study_report["conditions"][condition][group] = {}
                for name, measurements in options.items():
                    aggregate = summarize_runs(measurements)
                    study_report["conditions"][condition][group][name] = aggregate
                    summary.append({"study": study, "condition": condition, "feature_group": group,
                                    "model": name, "recordings": len(labels), "noise_runs": len(seeds),
                                    **{key: aggregate[key] for key in ("accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std")}})
            print(f"Evaluated {study}: {condition}", flush=True)
        report["studies"][study] = study_report
    output.mkdir(parents=True, exist_ok=True)
    (output / "research_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "research_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    ticks = ["Clean", "40", "35", "30", "25", "20"]
    for row, study in enumerate(("primary", "confirmation")):
        for col, model_name in enumerate(("random_forest", "svm")):
            axis = axes[row, col]
            for group, label in (("time", "Time features"), ("frequency", "FFT features"), ("combined", "Combined")):
                entries = [report["studies"][study]["conditions"][key][group][model_name] for key in ("clean", "40_db", "35_db", "30_db", "25_db", "20_db")]
                axis.errorbar(range(6), [entry["macro_f1_mean"] for entry in entries],
                              yerr=[entry["macro_f1_std"] for entry in entries], marker="o", capsize=3, label=label)
            axis.set(title=f"{study.title()} - {model_name.replace('_', ' ').title()}", ylabel="Macro F1", xlabel="Added noise SNR (dB)", ylim=(0, 1.05))
            axis.set_xticks(range(6), ticks)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    figure.suptitle("Feature ablation with five seeded Gaussian noise realizations")
    figure.savefig(output / "feature_ablation.png", dpi=160)
    plt.close(figure)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        run_validation(args.dataset, args.confirmation, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
