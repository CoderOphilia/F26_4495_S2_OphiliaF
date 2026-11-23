# Power Quality Analyzer demo and defense guide

This is a suggested six-to-eight-minute demonstration. Rehearse the controls
yourself before recording. Explain only what you understand and observed.

## Prepare

Open VS Code in `Implementation`, install the requirements and run
`python BackEnd/app.py`. Open http://127.0.0.1:5000. Generate the dataset with
`python BackEnd/generate_dataset.py` only if `Data/Synthetic` is absent.
Keep its `examples` folder ready in the file chooser. Use a clean browser tab
and leave the Flask terminal running. Saved history from earlier sessions is
expected; demonstration analyses are also saved.

## Script with actions

**0:00-0:45 - Purpose.** Say: "This is my Power Quality Analyzer research
prototype. It reads an existing voltage recording and predicts normal,
sag, swell, interruption or harmonic distortion. I compare Random Forest
and SVM, especially when measurement noise is added. It does not connect
to an electrical device or diagnose why the device failed."

**0:45-1:40 - Normal example.** Check the cleaned-data confirmation and choose
`Data/Synthetic/examples/normal_signal.csv`. Point to the waveform, 1,200
measurements, 6,000 Hz sample rate and RMS. Say: "The two CSV columns are
time in seconds and relative voltage. Voltage is referenced to a nominal
peak of one. The checkbox records my confirmation; it does not clean data."
Keep added noise Off and Random Forest selected. Click **Analyze Signal**.
Wait for **Normal signal**. Explain that a request sends the displayed values
to Flask, which computes ten features and applies the saved model.

**1:40-2:50 - Disturbances.** Upload `sag_signal.csv` and point to the smaller
wave during the event. Click **Analyze Signal** and wait for **Voltage sag**.
Change to Support Vector Machine and click Analyze again. Upload swell,
interruption and harmonic-distortion examples in turn if time allows; click
Analyze after each upload. Describe a larger wave, near-zero interval, and
changed shape. These demonstration CSVs are training examples, not the
held-out test recordings used to estimate accuracy.

**2:50-3:40 - Noise.** Return to normal or sag, move Added noise to 40 dB,
then click Analyze. Explain: "Lower SNR adds more noise. The website classifies
the displayed noisy preview. A single random preview can differ from the
averaged research results." Try 20 dB if useful, without promising a correct
prediction. Download signal CSV to retain the displayed preview. Return noise
to Off before the next clean example.

**3:40-4:30 - History.** Click **View history**. Identify predicted class,
model, added noise, date/time and analysis duration. Refresh to show the latest
records. Use Older analyses if more than 20 records exist. **Download shown
history CSV** exports just the visible page. Say: "SQLite stores these
summaries locally, not the uploaded waveform arrays. They survive a restart."
Close the dialog. Point out Contact us as an email panel with a demonstration
address, not a live analyst service.

**4:30-6:00 - Research results.** Open `Research/Results/feature_ablation.png`
and `Research/README.md`. Explain the 700/150/150 source-recording split,
training-only SVM scaling and validation-only candidate choice. The combined
models score 100% on clean simplified test signals; at primary 20 dB mean
accuracy is 78.27% RF and 73.20% SVM. Five noise seeds show variability.
FFT-only features are not consistently better than time-only features.
The second generator seed is a repeat within the same signal model, not a
real-device validation. Explain that normal signals are frequently mistaken
for swell at 20 dB. This is a limitation worth showing.

**6:00-7:00 - Finish.** Show the verification summary and source folders.
State that the scoped local workflow, two classifiers, noise experiment and
SQLite history are implemented. Future research would require independently
collected recordings, documented nominal voltage, more event combinations
and evaluation across instruments. Review the final report and actual work
log before submitting. Include the required assistance disclosure in academic
materials, using the real tools, versions and activities.

## Questions to be ready for

- **RMS?** Square each voltage value, average the squares, then take the square
  root. A clean unit-peak sine wave has RMS about 0.707. RMS alone can miss
  waveform shape, so the model also uses other features.
- **RF versus SVM?** RF votes across decision trees. SVM separates feature
  patterns using a scaled RBF decision boundary. Their noise weaknesses differ.
- **Why macro F1?** It averages each class's F1 equally. Accuracy alone hides
  which class fails; show the confusion matrices and per-class recall too.
- **Why three partitions?** Training fits parameters, validation chooses among
  candidates, and test estimates performance after selection. Noisy versions
  of a test recording remain in test.
- **What does the API do?** It validates the recording, extracts features,
  predicts with the selected model and saves a summary. History is a separate
  GET request; invalid input or storage failures produce an explanation.
- **Why SQLite?** It is a local file database included with Python. No separate
  server is needed. Records store IDs, UTC time, class, model, SNR and duration.
- **How is this useful?** It demonstrates offline screening of compatible
  voltage recordings. A suitable recorder must capture data before/during an
  event. The app cannot recover measurements that were never recorded.
- **Is 100% proof of reliability?** No. The simplified generator is easy to
  separate when clean. Low-noise success does not transfer automatically to
  actual equipment, different sampling, combined events or unknown classes.
- **What changed from the proposal?** The core scope remains; confidence
  display was removed, cleaned-data confirmation/contact guidance added, and
  the planned external waveform comparison uses supplied synthetic references
  with explicit provenance limitations. FFT was implemented; wavelet features
  mentioned among possible tools were not implemented.
