# F26_4495_S2_OphiliaF
Machine learning classification of power quality disturbances (sag, swell, harmonics) from voltage signals, with a focus on how classification accuracy degrades under measurement noise.
---
Student Name: Ophilia Fradarick
---
Student ID: 300386351
---
Email ID: fradaricko@student.douglascollege.ca

## Run the web application locally

Requirements: Python 3.10 or newer. Run these commands from the repository root:

```bash
python -m pip install -r Implementation/BackEnd/requirements.txt
python Implementation/BackEnd/app.py
```

Open http://127.0.0.1:5000 in your browser and keep the server terminal running.
Upload `Implementation/normal_signal.csv` to try CSV validation, waveform display,
noise preview, RMS measurements, and CSV download. Stop the server with Ctrl+C.

The Flask server currently serves the front end. Model training and the
`POST /api/classify` endpoint are not implemented yet, so the Analyze Signal
button does not produce ML predictions at this stage.

## Run the normal signal generator

```bash
python Implementation/BackEnd/signal_generator.py
```

The example prints the number of measurements and the first five signal values.
The reusable `generate_normal_signal` function accepts frequency, sample rate,
duration, relative amplitude, and phase in radians. It returns timestamp and
signal arrays. The default example contains 1,200 measurements of a 60 Hz sine
wave over a 0.2-second recording interval. No CSV is written by this module.
