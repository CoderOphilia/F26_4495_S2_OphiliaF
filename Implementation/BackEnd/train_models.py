"""Train and evaluate clean-signal baselines using saved recording partitions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_extraction import (
    FEATURE_NAMES, FEATURE_VERSION, SAMPLE_COUNT, SAMPLE_RATE_HZ, extract_features,
)
from signal_generator import SIGNAL_TYPES


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "Data" / "Synthetic"
DEFAULT_OUTPUT = ROOT / "Models"


def evaluation(model, features, labels):
    predicted = model.predict(features)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, labels=list(SIGNAL_TYPES), average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=list(SIGNAL_TYPES)).tolist(),
        "classification_report": classification_report(
            labels, predicted, labels=list(SIGNAL_TYPES), output_dict=True, zero_division=0,
        ),
    }


def train_models(dataset_dir=DEFAULT_DATASET, output_dir=DEFAULT_OUTPUT):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Model output must be empty. Choose a new --output folder.")
    archive_path = dataset_dir / "signals.npz"
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["signals_npz_sha256"] != archive_hash:
        raise ValueError("Dataset checksum does not match its manifest.")
    with np.load(archive_path, allow_pickle=False) as archive:
        signals = archive["voltage_relative"]
        times = archive["time_seconds"]
        labels = archive["labels"]
        splits = archive["splits"]
        signal_ids = archive["signal_ids"]
    if signals.ndim != 2 or signals.shape[1] != SAMPLE_COUNT:
        raise ValueError("Training requires 1,200 samples per recording.")
    count = signals.shape[0]
    if any(array.shape != (count,) for array in (labels, splits, signal_ids)):
        raise ValueError("Dataset labels, IDs, and partitions must align with recordings.")
    if len(set(signal_ids)) != count:
        raise ValueError("A recording ID must occur in exactly one partition.")
    if set(labels) != set(SIGNAL_TYPES) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("Expected five classes and train/validation/test partitions.")
    if times.shape != (SAMPLE_COUNT,) or not np.allclose(np.diff(times), 1 / SAMPLE_RATE_HZ, rtol=0.001, atol=1e-10):
        raise ValueError("Training requires uniform 6,000 Hz timestamps.")
    masks = {name: splits == name for name in ("train", "validation", "test")}
    for mask in masks.values():
        if set(labels[mask]) != set(SIGNAL_TYPES):
            raise ValueError("Every partition must include all five classes.")
    features = np.stack([extract_features(signal, SAMPLE_RATE_HZ) for signal in signals])
    if not np.isfinite(features).all():
        raise ValueError("Extracted features must be finite.")

    candidates = {
        "random_forest": [
            RandomForestClassifier(n_estimators=200, max_depth=depth, min_samples_leaf=2, random_state=42)
            for depth in (8, None)
        ],
        "svm": [make_pipeline(StandardScaler(), SVC(C=c, kernel="rbf", gamma="scale")) for c in (1, 10)],
    }
    report = {
        "dataset_sha256": archive_hash,
        "dataset_seed": manifest["seed"],
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "class_order": list(SIGNAL_TYPES),
        "split_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "partition_ids": {name: signal_ids[mask].tolist() for name, mask in masks.items()},
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
        },
        "selection_rule": "Highest validation macro F1; first candidate wins ties. Fit on train only. Evaluate test only after selection.",
        "noise_condition": "clean; no added noise",
        "limitations": "Synthetic single events at 6 kHz for 0.2 seconds, nominal 60 Hz and relative peak 1.0. These scores do not establish real-device accuracy.",
        "models": {},
    }
    selected_models = {}
    train, validation, test = (masks[name] for name in ("train", "validation", "test"))
    for name, options in candidates.items():
        validation_scores = []
        for model in options:
            model.fit(features[train], labels[train])
            validation_scores.append(evaluation(model, features[validation], labels[validation]))
        selected_index = max(range(len(options)), key=lambda index: validation_scores[index]["macro_f1"])
        selected = options[selected_index]
        selected_models[name] = selected
        report["models"][name] = {
            "candidates": [str(model) for model in options],
            "validation": validation_scores,
            "selected_candidate": selected_index,
            "test": evaluation(selected, features[test], labels[test]),
        }

    bundle = {
        "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
        "sample_rate_hz": SAMPLE_RATE_HZ, "sample_count": SAMPLE_COUNT,
        "classes": list(SIGNAL_TYPES), "dataset_sha256": archive_hash,
        "sklearn_version": sklearn.__version__, "models": selected_models,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # Publish both classifiers together so a request cannot read a partial pair.
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".tmp", delete=False) as stream:
        temporary_path = Path(stream.name)
    try:
        joblib.dump(bundle, temporary_path, compress=3)
        os.replace(temporary_path, output_dir / "classifiers.joblib")
    finally:
        temporary_path.unlink(missing_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Clean synthetic baseline", "",
        "Fitted on training recordings only. Candidate selection used validation macro F1.",
        "The saved test partition was evaluated after candidate selection.", "",
        "| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, result in report["models"].items():
        val = result["validation"][result["selected_candidate"]]
        test_result = result["test"]
        lines.append(f"| {name} | {val['accuracy']:.4f} | {val['macro_f1']:.4f} | {test_result['accuracy']:.4f} | {test_result['macro_f1']:.4f} |")
        print(f"{name}: test accuracy={test_result['accuracy']:.4f}, macro F1={test_result['macro_f1']:.4f}")
    lines.extend(["", report["limitations"], "", "Full confusion matrices, partition IDs, parameters, and versions are in evaluation.json.", ""])
    (output_dir / "evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        train_models(args.dataset, args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(f"Saved classifiers and evaluation to {args.output.resolve()}")


if __name__ == "__main__":
    main()
