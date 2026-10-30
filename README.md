# F26_4495_S2_OphiliaF
Machine learning classification of power quality disturbances (sag, swell, harmonics) from voltage signals, with a focus on how classification accuracy degrades under measurement noise.
---
Student Name: Ophilia Fradarick
---
Student ID: 300386351
---
Email ID: fradaricko@student.douglascollege.ca


## Run Voltage Signal Checker locally

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

The Flask server currently serves the front end. Model training and the
`POST /api/classify` endpoint are not implemented yet, so Analyze Signal does not
produce real ML predictions at this stage. Once connected, results display
Normal signal or Problem detected followed by the disturbance name, without
showing a probability score. Stop the server with Ctrl+C.

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
Feature extraction, trained classifiers, and the API remain the next stage.

See [dataset format, parameter ranges, and evaluation rules](Implementation/Data/README.md).

Run the generation checks:

```bash
python -m unittest discover -s Implementation/BackEnd/tests -v
```
