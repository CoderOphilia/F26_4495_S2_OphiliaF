# F26_4495_S2_OphiliaF
Machine learning classification of power quality disturbances (sag, swell, harmonics) from voltage signals, with a focus on how classification accuracy degrades under measurement noise.
---
Student Name: Ophilia Fradarick
---
Student ID: 300386351
---
Email ID: fradaricko@student.douglascollege.ca


## Run Power Quality Analyzer locally

Requirements: Python 3.10 or newer. Run these commands from the repository root:

```bash
python -m pip install -r Implementation/BackEnd/requirements.txt
python Implementation/BackEnd/app.py
```

Open http://127.0.0.1:5000 and keep the server terminal running.

1. Confirm that your CSV contains cleaned data using the checkbox.
2. Upload `Implementation/normal_signal.csv` or a compatible cleaned recording.
3. Explore the waveform, noise slider, model selection, RMS measurements, and CSV download.
4. Select Analyze Signal to request a normal-or-disturbance result.

Unchecking the cleaned-data confirmation clears the current recording and
blocks further upload and analysis. The confirmation is a user declaration;
CSV format checks do not automatically clean or certify a recording.
Missing or invalid values, duplicate timestamps, and irregular sampling are
rejected. Actual voltage disturbances should be preserved during data cleaning.

The Contact us bubble opens an email panel using the requested placeholder
address `ophiliaproject@gmail.com`. Its email links open the user's email
application; this is not a live-chat service and no email is sent automatically.

The Flask server now implements `POST /api/classify` using the committed
Random Forest and SVM models. Analyze Signal returns Normal signal or
Problem detected followed by the predicted disturbance, without a probability
score. Classification requires 1,200 samples at 6,000 Hz, a 0.2-second window
around a nominal 60 Hz signal, and relative voltage referenced to nominal peak
1.0. Other supported CSV sizes can still be previewed. Invalid or unsupported
requests show an explanation. Stop the server with Ctrl+C.

## Run the normal signal generator

```bash
python Implementation/BackEnd/signal_generator.py
```

The example prints the number of measurements and the first five signal values.
The reusable `generate_normal_signal` function accepts frequency, sample rate,
duration, relative amplitude, and phase in radians. It returns timestamp and
signal arrays. The default example contains 1,200 measurements of a 60 Hz sine
wave over a 0.2-second recording interval. No CSV is written by this module.

## Generate a labelled synthetic dataset

From the repository root:

```bash
python Implementation/BackEnd/generate_dataset.py
```

This creates 1,000 clean synthetic recordings (200 per class) for normal, sag,
swell, interruption, and harmonic distortion. Frequency, amplitude, phase,
event timing, and disturbance parameters vary between examples. The default
seed is 42; generation parameters and individual sample seeds are saved.

Output goes to `Implementation/Data/Synthetic/`: a NumPy archive, metadata CSV,
manifest, and five upload-compatible examples in `examples/`. The original
`Implementation/normal_signal.csv` is preserved. Output folders must be empty;
use `--output` to generate another dataset in a separate directory.

The default partitions contain 700 training, 150 validation, and 150 test
recordings. Future noisy copies must retain their original recording's
partition. Generated data stays local and is ignored by Git. These simplified
mathematical examples are not hardware-validated or certified measurements.
Time-domain/FFT features, both trained classifiers, and the Flask API are
implemented. Reproducible clean and noise comparisons are committed alongside
the models. These results apply to simplified synthetic signals.

See [dataset format, parameter ranges, and evaluation rules](Implementation/Data/README.md).

Run the generation checks:

```bash
python -m unittest discover -s Implementation/BackEnd/tests -v
```


## Train models and reproduce noise comparisons

The committed `Implementation/Models/classifiers.joblib` allows classification
without retraining. To reproduce training after generating the dataset:

```bash
python Implementation/BackEnd/train_models.py --output Implementation/Models_retrained
python Implementation/BackEnd/evaluate_noise.py --output Implementation/Models/noise_repeat
```

Existing nonempty output folders are preserved. The noise command above evaluates
the committed models against the original generated dataset. For retrained
models, also supply `--models Implementation/Models_retrained`.

Training uses the saved 700/150/150 train/validation/test recordings. SVM scaling
is fitted on training data only. Validation macro F1 selects model settings,
and the test partition is evaluated after selection. Frozen models are then
compared on the same test IDs at clean, 40 dB, 30 dB, and 20 dB SNR.

- [Classification setup, features, API, and limits](Implementation/BackEnd/README.md)
- [Clean baseline results](Implementation/Models/evaluation.md)
- [Noise results and chart](Implementation/Models/noise_experiment/results.md)

Full JSON results include confusion matrices, per-class scores, IDs, checksums,
and settings. The noise comparison uses one seeded Gaussian realization per
recording.

## Saved analysis history

Successful classifications now save their predicted class, selected model,
added noise setting, UTC timestamp, and analysis time to a local SQLite
database. Open http://127.0.0.1:5000/api/history after analyzing a recording to
inspect the newest 20 results. Use `/api/history?limit=5` for five records.

The database at `Implementation/Data/analysis_history.sqlite3` is created on
the first successful analysis and persists across server restarts. Its data
and journal files are ignored by Git. Failed saves return a service error.
Select View history in the result card to see the newest 20 saved analyses
in the app. The dialog shows the predicted result, model, noise setting, local
date/time, and analysis duration. Refresh history reloads the latest records;
Close or Escape returns to the workspace. Narrow screens support horizontal
table scrolling. The dialog includes empty history and connection error states.

See the classification workflow for timing, API details, and UI test commands.
