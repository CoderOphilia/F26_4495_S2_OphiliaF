# How training and testing work in this project

Machine learning learns patterns from examples. In this project, an example
is a voltage recording with a known signal type, called its label.

## Start with known examples

`BackEnd/generate_dataset.py` creates 1,000 recordings: 200 each of normal,
sag, swell, interruption and harmonic distortion. These are mathematical
examples, not measurements collected from equipment. The generator knows
each label because it constructed that signal type.

Each recording contains 1,200 voltage samples. Those samples describe one
recording; they are not 1,200 separate training examples.

## Calculate features

`BackEnd/feature_extraction.py` calculates ten useful measurements from each
recording. They include RMS, peak amplitude, changes in local RMS and frequency
information from the FFT. These measurements are called features.

The original waveform remains available. Feature extraction does not replace
the original dataset with ten voltage readings. It creates ten model inputs
for each recording. The correct label is stored separately from those inputs.

## Teach the models

`BackEnd/train_models.py` uses the saved partitions:

| Group | Recordings | What it is used for |
| --- | ---: | --- |
| Training | 700 | Learn relationships between features and known labels. |
| Validation | 150 | Choose between candidate model settings. |
| Test | 150 | Check predictions after model selection. |

Random Forest combines predictions from many decision trees. SVM learns
boundaries between feature patterns. The SVM pipeline also scales its inputs
using values learned from training records only.

The code trains two candidate settings for each model. Validation macro F1
selects a setting. The test answers are used to measure performance after
selection, not to teach the model or choose those settings.

## Understand the testing output

Suppose the known test label is sag and the model predicts sag. That is a
correct prediction. If it predicts normal instead, that is an error.

- Accuracy is the fraction of test predictions that are correct.
- Recall for sag is the fraction of actual sag recordings recognised as sag.
- Precision for sag is the fraction of sag predictions that actually are sag.
- F1 combines precision and recall. Macro F1 gives each class equal weight.
- A confusion matrix shows which known classes were predicted as other classes.

Read [the saved clean results](../Models/evaluation.md) and
[the initial noise comparison](../Models/noise_experiment/results.md).
The [additional research](../Research/README.md) compares feature groups and
repeats noisy evaluations. Noisy copies remain in their source test partition.

The strong clean synthetic results do not prove accuracy on actual equipment.
Independently labelled real recordings are needed to measure that performance.

## What happens when someone uploads a CSV

1. The app checks the recording format and sampling requirements.
2. Python calculates the same ten features used during training.
3. The selected saved model predicts a class.
4. The app displays the prediction and saves a summary in SQLite history.

The saved model bundle is `Models/classifiers.joblib`. The upload does not
retrain it. SQLite stores result summaries; it does not teach either model.

## Try the workflow yourself

From the Implementation folder, start `python BackEnd/app.py`, open
http://127.0.0.1:5000 and confirm that your data is cleaned. Upload a CSV
from `Data/Synthetic/examples`, keep noise Off, select a model and click
Analyze Signal. Repeat with the other model and try added noise.

The example CSVs are training examples provided for demonstrating the app.
Use the separate test reports when discussing measured model accuracy.
