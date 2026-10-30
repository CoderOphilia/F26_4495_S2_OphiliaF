# Synthetic dataset

This milestone implements reproducible examples for normal, sag, swell,
interruption, and harmonic distortion signals. Developed with AI assistance
for the CSIS 4495 project. These are educational mathematical models, not
certified measurements or proof of accuracy on real devices.

From the repository root, run:

```bash
python Implementation/BackEnd/generate_dataset.py
```

If VS Code is opened directly in the `Implementation` folder, run:

```bash
python BackEnd/generate_dataset.py
```

The default output is `Implementation/Data/Synthetic/` (relative to the
repository root). It contains 200 recordings per class, 1,000 total, and
1,200 voltage measurements per recording. Each recording spans a 0.2-second
window with a 6,000 Hz sample rate. The last timestamp is one sample interval
before 0.2 seconds. No measurement noise is added at this stage.

## Files

- `signals.npz`: NumPy archive with shared `time_seconds` (1,200 values),
  `voltage_relative` (1,000 by 1,200), and matching `labels`, `signal_ids`,
  and `splits` arrays (one value per recording). Load with
  `np.load(path, allow_pickle=False)`.
- `metadata.csv`: one row per recording, in the same order as the archive.
  Saves the label, partition, sample seed, and signal-generation parameters.
  Blank fields mean the parameter does not apply to that class.
- `manifest.json`: dataset seed, NumPy version, class and partition counts,
  parameter ranges, archive checksum, example IDs, and limitations.
- `examples/`: five two-column CSV recordings compatible with the current
  upload form. Each is a training example identified in the manifest.

The examples are `normal_signal.csv`, `sag_signal.csv`, `swell_signal.csv`,
`interruption_signal.csv`, and `harmonic_distortion_signal.csv`.
Upload these CSVs individually, rather than uploading the archive or metadata.

## Variation within each class

All classes vary the fundamental frequency (59 to 61 Hz), reference peak
amplitude (0.9 to 1.1 relative units), and phase (0 to 2 pi radians).
Sag, swell, and interruption examples vary event start (0.03 to 0.07 seconds)
and duration (0.04 to 0.09 seconds). Only the event interval is scaled.

| Class | Construction |
| --- | --- |
| Normal | Unmodified reference sine wave. |
| Sag | Event amplitude multiplier between 0.2 and 0.8. |
| Swell | Event amplitude multiplier between 1.15 and 1.8. |
| Interruption | Near-zero event multiplier between 0 and 0.03. |
| Harmonic distortion | Add third and fifth harmonics with ratios 0.1 to 0.3 and 0.05 to 0.2; vary each harmonic's phase. |

These ranges are project design choices. They are not a claim that every
possible disturbance, electrical standard, or recording condition is covered.
The sample IDs identify records within one dataset configuration; when
combining datasets, include the configuration or manifest identity as well.

## Reproducibility and future evaluation

The default master seed is 42. A separate seed for each recording is saved
alongside every applied parameter. Using the same seed, count, generator
version, and compatible NumPy environment reproduces the numerical data and
metadata. Runtime versions are recorded to make comparisons explicit.

The default partitions contain 700 training, 150 validation, and 150 test
recordings. Each class contributes 140, 30, and 30 recordings respectively.
Partition assignment happens before any future noise augmentation. Dataset
rows are shuffled, and metadata remains aligned with the stored signals.

Use training records to fit models, validation records for model choices,
and test records for the final evaluation. All future noisy copies of a
recording must retain its original ID and partition. Fit preprocessing on
training data only. Do not pass ground-truth labels, partition names, event
multipliers, or other generation metadata to a classifier as input features.

To make a separate dataset without overwriting the existing one:

```bash
python Implementation/BackEnd/generate_dataset.py --samples-per-class 200 --seed 43 --output Implementation/Data/Synthetic_seed43
```

The generator refuses nonempty output folders. Generated datasets are ignored
by Git and are recreated from the committed code. Keep dataset manifests with
any saved results. Imported `RawData` files are not used: their stated 50 Hz,
5 kHz, one-cycle format differs from this generator and their provenance still
needs verification.

## Limitations and next milestone

The waveforms contain simplified single events, abrupt amplitude transitions,
and no background noise, transients, combined disturbances, or sensor effects.
The records have not been validated against hardware recordings. High accuracy
on this dataset would not establish real-world diagnostic reliability.

Next: implement time-domain and FFT features, then train and evaluate Random
Forest and SVM models using the saved partitions. The Flask classifier is not
implemented by this dataset milestone.

Run the generation checks from the repository root:

```bash
python -m unittest discover -s Implementation/BackEnd/tests -v
```
