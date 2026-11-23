# Application verification - 22 November 2026

An isolated Windows Python 3.13.2 virtual environment was created and the
documented six pinned requirements were installed from scratch. `pip check`
reported no broken requirements. The application loaded its existing trusted
scikit-learn 1.6.1 model bundle without retraining.

| Check | Observed outcome |
| --- | --- |
| Python regression suite | 31 tests passed, including three research checks |
| JavaScript regression suite | 18 tests passed |
| Real HTTP workflow | All five analytical CSV classes classified correctly with both models |
| Resource serving | HTML, CSS and script served by Flask |
| Invalid requests | Rejected; no extra history records saved |
| History pagination | Two pages of five, ten distinct IDs |
| Persistence | All ten records remained after terminating and restarting the server process |
| Frontend handlers against live HTTP | Five generated CSVs with both models, 40 dB preview, short-file blocking, clear on unchecked confirmation |
| In-app browser | Empty history, refresh after eleven analyses, populated rows and enabled export control observed |

The frontend handler check executes the actual `script.js` using DOM/canvas
stubs and real HTTP requests. It is not a graphical browser test. The in-app
browser's file-chooser automation timed out, so a fresh manual file upload
through that browser was not verified in this pass. The parsed CSV workflow
and predictions were independently verified by the handler and HTTP checks.
The existing regression tests exercise CSV export content; a new browser file
download was not performed. Cross-browser/mobile hardware testing remains
outside this recorded pass.

[HTTP evidence](http_workflow.json) includes model hash and individual
predictions. [Browser screenshot](history_browser.jpg) shows the saved summaries.
Verification used private temporary databases. The user's application history
and serving model bundle were preserved. Times in the screenshot include
cold model loading and are not a controlled performance benchmark.

Re-run the non-UI checks from `Implementation`:

```bash
python -m pip check
python -m unittest discover -s BackEnd/tests -v
node --test FrontEnd/tests/history.test.cjs
python BackEnd/verify_application.py
```

The HTTP verifier creates a private temporary database, launches two local
server processes in sequence and removes its database afterwards. It needs
permission to open local loopback sockets. Its checks do not retrain models.
