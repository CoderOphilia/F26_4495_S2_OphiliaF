# Classification workflow

Run commands below with VS Code opened in the `Implementation` folder.

```bash
python -m pip install -r BackEnd/requirements.txt
python BackEnd/app.py
```

Open http://127.0.0.1:5000. The committed model bundle makes classification
available immediately. Confirm cleaned data, upload `normal_signal.csv`,
select Random Forest or SVM, and select Analyze Signal. The result is
Normal signal or Problem detected followed by the predicted disturbance.

To try all five classes, generate the dataset once and upload CSVs from
`Data/Synthetic/examples/`.

```bash
python BackEnd/generate_dataset.py
```

An existing nonempty dataset folder is preserved. If it already exists,
use its examples or choose another output folder with `--output`.

## Supported recordings

Classification requires exactly 1,200 samples at 6,000 Hz, a 0.2-second
window around a nominal 60 Hz signal. The final timestamp is one sample
interval before the end of the window. Use relative voltage referenced to
a nominal peak of 1.0. A raw recording in volts needs its known nominal peak
as the reference. Dividing each recording by its own maximum would hide
amplitude differences and must be avoided.

CSV columns are `time_seconds,voltage_relative`. Values must be finite and
timestamps uniformly increasing. Magnitudes above 5 relative units and flat
recordings are outside this prototype's accepted range. Include the waveform
before and after an interruption. The checkbox records the user's confirmation;
it does not establish that data is cleaned. Preview and download still support
other uniform recordings with 4 to 50,000 samples. Unsupported classification
requests return an explanation.

## Features and training

`feature_extraction.py` produces a fixed list of ten waveform measurements.
They include RMS, absolute peak, crest factor, minimum/maximum/standard deviation
of sliding one-cycle RMS relative to whole-window RMS, dominant FFT frequency,
and third, fifth, and remaining AC band power fractions. The FFT uses a Hann
window and bands of 45–75, 165–195, and 285–315 Hz. Band fractions are features
for this dataset and are not a certified THD measurement.

To reproduce training without overwriting the committed model bundle:

```bash
python BackEnd/train_models.py --output Models_retrained
```

The dataset must first exist in `Data/Synthetic/`. The saved partitions are
700 train, 150 validation, and 150 test recordings. Model inputs contain only
waveform features. Labels, generation parameters, IDs, and partition names
are excluded. SVM scaling is fitted inside its pipeline on training data only.

Both Random Forest candidates use 200 trees, minimum leaf size 2, and seed 42;
maximum depth is selected from 8 and unrestricted. SVM uses an RBF kernel,
gamma `scale`, and C selected from 1 and 10. Validation macro F1 selects each
model, with the first candidate winning ties. The selected models retain their
training-only fit. Test recordings are used after selection to report accuracy,
macro F1, per-class scores, and confusion matrices.

`Models/classifiers.joblib` stores both selected classifiers together, including
SVM scaling and the feature contract. `Models/evaluation.json` records results,
partition IDs, the dataset checksum, and runtime versions. `evaluation.md`
summarizes the clean baseline. Only load model files produced by this trusted
project. Retraining with a different scikit-learn version requires matching
training and serving environments.

## API

The existing browser workflow sends JSON to `POST /api/classify`.

```json
{
  "time_seconds": [0.0, 0.0001666666667],
  "voltage_relative": [0.0, 0.0627905195],
  "sample_rate_hz": 6000,
  "model": "random_forest",
  "cleaned_data_confirmed": true,
  "added_noise_snr_db": null
}
```

The arrays above are abbreviated. A request needs all 1,200 values in both
arrays. `model` accepts `random_forest` or `svm`. Added noise metadata accepts
null, 40, 30, or 20. The browser already adds noise to the displayed samples;
the API classifies those supplied values without adding it again.

A successful response contains only the class, for example
`{"prediction":"sag"}`. Invalid JSON returns 400, unsupported recording or
settings return 422, oversized requests return 413, and missing/incompatible
trained models or unavailable history storage return 503. Request bodies are limited to 5 MB. The server
loads its configured local model bundle, checks its feature and runtime
versions, and does not accept model uploads.

## Verification and limits

```bash
python -m unittest discover -s BackEnd/tests -v
```

Tests cover analytical features, train-only scaling, model persistence,
all five CSV classes with both models, noise preview inference, invalid
requests, missing models, static files, and request limits.

This is a research prototype evaluated on simplified generated single events.
Its clean synthetic scores do not establish accuracy on hardware recordings,
combined disturbances, other sampling conditions, or arbitrary voltage scales.
SQLite persistence, history retrieval, and the website history view are
implemented.

## Saved analysis history

Every successful classification is saved to `Data/analysis_history.sqlite3`.
The database is created automatically on the first saved analysis. Each record
contains the predicted class, model, added noise setting, UTC timestamp, and
analysis time in milliseconds. Timing includes model loading, feature
extraction, and prediction, and excludes saving the record. A cold first
request can therefore take longer than later requests.

Select View history in the result card to see the newest 20 saved analyses.
The dialog shows the result, model, added noise, local date/time, and analysis
duration. Refresh history retrieves the latest records. Close or Escape returns
to the workspace. On narrow screens, scroll across the table to see every
column. Empty history and connection failures show a message with a next step.

Download shown history CSV exports the records currently displayed, up to 20,
to `analysis_history.csv`. It includes record IDs, UTC timestamps, predicted
classes, model keys, noise SNR, and full analysis durations in milliseconds.
A blank SNR cell means added noise was off. The button is enabled after saved
results load and is disabled during refresh or when history is empty or
unavailable. Refresh first to include newly saved analyses.

The dialog uses `GET /api/history?limit=20`. Open
http://127.0.0.1:5000/api/history to inspect results as JSON or use
`/api/history?limit=5` to retrieve five. Limits must be whole numbers between
1 and 100. An empty history returns `{"history":[]}`.

Records persist after the server restarts. Invalid inputs and failed model
requests do not create history records. A database failure returns 503 so the
caller knows that the analysis was not saved. Saves use a transaction and a
separate connection for each operation. Stored summaries do not include
uploaded waveform arrays. The local database and its journal files are ignored
by Git. Python includes SQLite support, so no additional package is needed.

If Node.js is installed, run the history UI regression checks from the
`Implementation` folder:

```bash
node --test FrontEnd/tests/history.test.cjs
```

These checks cover empty/populated history, failure and retry, stale requests
after closing/reopening, timeout recovery, CSV export and refresh behavior,
the waveform CSV format, and direct HTML preview guidance.

## Noise robustness experiment

```bash
python BackEnd/evaluate_noise.py
```

This evaluates the frozen clean-trained models on the same 150 test recording
IDs at clean, 40 dB, 30 dB, and 20 dB SNR. No fitting, scaling updates, or model
selection uses the noisy test results. Each recording has reproducible Gaussian
noise, centered and scaled to the exact measured SNR; its noise direction is
shared across conditions. Parent IDs remain in the test partition. Only one
noise realization is used per recording, so this is an initial robustness
comparison. The browser's fresh noise previews use expected noise power and
can produce different individual predictions.

Results, confusion matrices, per-class metrics, CSV comparison, and the chart
are saved in `Models/noise_experiment/`. Existing outputs are preserved. Use
`--output Models/noise_repeat` to rerun in another folder. Report the measured
scores as synthetic experiment results, with the generator's limitations.
