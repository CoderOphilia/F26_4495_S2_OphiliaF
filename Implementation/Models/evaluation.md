# Clean synthetic baseline

Fitted on training recordings only. Candidate selection used validation macro F1.
The saved test partition was evaluated after candidate selection.

| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 |
| --- | ---: | ---: | ---: | ---: |
| random_forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| svm | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Synthetic single events at 6 kHz for 0.2 seconds, nominal 60 Hz and relative peak 1.0. These scores do not establish real-device accuracy.

Full confusion matrices, partition IDs, parameters, and versions are in evaluation.json.
