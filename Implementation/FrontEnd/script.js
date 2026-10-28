// ---------- FIND THE HTML ELEMENTS ----------

const fileInput = document.getElementById("csvFile");
const fileName = document.getElementById("fileName");

const noiseSlider = document.getElementById("noiseSlider");
const noiseLabel = document.getElementById("noiseLabel");
const modelSelect = document.getElementById("modelSelect");

const analyzeButton = document.getElementById("analyzeButton");
const downloadButton = document.getElementById("downloadButton");

const sampleCount = document.getElementById("sampleCount");
const sampleRate = document.getElementById("sampleRate");
const rmsValue = document.getElementById("rmsValue");

const prediction = document.getElementById("prediction");
const resultMessage = document.getElementById("resultMessage");
const cleanedData = document.getElementById("cleanedData");
const contactButton = document.getElementById("contactButton");
const contactPanel = document.getElementById("contactPanel");
const contactClose = document.getElementById("contactClose");
const statusMessage = document.getElementById("status");

const canvas = document.getElementById("waveform");
const chartEmpty = document.getElementById("chartEmpty");
const context = canvas.getContext("2d");


// ---------- STORE THE CURRENT SIGNAL ----------

// Each position in these arrays describes one measurement.
let times = [];
let originalSignal = [];
let previewSignal = [];

let uploadedFileName = "";
let estimatedSampleRate = 0;
let busy = false;

// Slider positions: 0 = off, 1 = 40 dB, 2 = 30 dB, 3 = 20 dB.
const noiseConditions = [null, 40, 30, 20];


// ---------- GENERAL INTERFACE FUNCTIONS ----------

function setStatus(message, isError = false) {
    statusMessage.textContent = message;
    statusMessage.hidden = !message;
    statusMessage.classList.toggle("error", isError);
}

function resetResult() {
    prediction.textContent = "No result yet";

    resultMessage.textContent = !cleanedData.checked
        ? "Confirm your data is cleaned, then upload a CSV."
        : times.length > 0
            ? "Select Analyze Signal to check this recording."
            : "Upload a cleaned CSV recording to begin.";
}

function updateControls() {
    const hasSignal = cleanedData.checked && times.length > 0;

    cleanedData.disabled = busy;
    fileInput.disabled = busy || !cleanedData.checked;
    modelSelect.disabled = busy || !cleanedData.checked;

    noiseSlider.disabled = busy || !hasSignal;
    analyzeButton.disabled = busy || !hasSignal;
    downloadButton.disabled = busy || !hasSignal;
}

function clearRecording() {
    times = [];
    originalSignal = [];
    previewSignal = [];

    uploadedFileName = "";
    estimatedSampleRate = 0;

    fileName.textContent = "No file selected.";
    sampleCount.textContent = "No data";
    sampleRate.textContent = "No data";
    rmsValue.textContent = "No data";

    noiseSlider.value = "0";
    noiseLabel.textContent = "Off";

    resetResult();
    updateControls();
    drawWaveform();
}


// ---------- READ AND CHECK THE CSV ----------

function parseCSV(text) {
    // Remove an optional byte-order mark and surrounding whitespace.
    const cleanedText = text.replace(/^\uFEFF/, "").trim();

    if (cleanedText === "") {
        throw new Error("The CSV file is empty.");
    }

    const lines = cleanedText
        .split(/\r?\n/)
        .filter(line => line.trim() !== "");

    const header = lines[0]
        .split(",")
        .map(value => value.trim());

    if (
        header.length !== 2 ||
        header[0] !== "time_seconds" ||
        header[1] !== "voltage_relative"
    ) {
        throw new Error(
            "The CSV header must be: time_seconds,voltage_relative"
        );
    }

    const rows = lines.slice(1);

    if (rows.length < 4 || rows.length > 50000) {
        throw new Error(
            "Please upload between 4 and 50,000 measurements."
        );
    }

    const loadedTimes = [];
    const loadedSignal = [];

    rows.forEach((line, index) => {
        const cells = line
            .split(",")
            .map(value => value.trim());

        if (
            cells.length !== 2 ||
            cells.some(value => value === "")
        ) {
            throw new Error(
                `Measurement ${index + 1} must contain two numbers.`
            );
        }

        const timeValue = Number(cells[0]);
        const voltageValue = Number(cells[1]);

        if (
            !Number.isFinite(timeValue) ||
            !Number.isFinite(voltageValue)
        ) {
            throw new Error(
                `Measurement ${index + 1} contains an invalid number.`
            );
        }

        if (
            index > 0 &&
            timeValue <= loadedTimes[index - 1]
        ) {
            throw new Error(
                "Timestamps must increase with every measurement."
            );
        }

        loadedTimes.push(timeValue);
        loadedSignal.push(voltageValue);
    });

    // Estimate the time between consecutive measurements.
    const averageStep =
        (loadedTimes[loadedTimes.length - 1] - loadedTimes[0]) /
        (loadedTimes.length - 1);

    if (
        !Number.isFinite(averageStep) ||
        averageStep <= 0 ||
        !Number.isFinite(1 / averageStep)
    ) {
        throw new Error("The recording has an invalid time interval.");
    }

    const tolerance = Math.max(averageStep * 0.001, 1e-10);

    for (let index = 1; index < loadedTimes.length; index++) {
        const currentStep =
            loadedTimes[index] - loadedTimes[index - 1];

        if (Math.abs(currentStep - averageStep) > tolerance) {
            throw new Error(
                "This prototype requires evenly spaced timestamps."
            );
        }
    }

    return {
        times: loadedTimes,
        signal: loadedSignal,
        sampleRate: 1 / averageStep
    };
}


// ---------- HANDLE A FILE SELECTION ----------

fileInput.addEventListener("change", async () => {
    if (!cleanedData.checked) {
        fileInput.value = "";
        setStatus("Confirm that your CSV contains cleaned data before uploading.", true);
        return;
    }
    const file = fileInput.files[0];

    if (!file) {
        return;
    }

    busy = true;
    clearRecording();

    setStatus("Reading your voltage recording...");

    try {
        if (file.size > 5 * 1024 * 1024) {
            throw new Error("Please upload a file smaller than 5 MB.");
        }

        const text = await file.text();
        const recording = parseCSV(text);

        times = recording.times;
        originalSignal = recording.signal;
        estimatedSampleRate = recording.sampleRate;
        uploadedFileName = file.name;

        fileName.textContent = uploadedFileName;

        sampleCount.textContent =
            times.length.toLocaleString();

        sampleRate.textContent =
            estimatedSampleRate.toFixed(1) + " Hz";

        updatePreview();

        setStatus(
            `Loaded ${uploadedFileName}: ` +
            `${times.length.toLocaleString()} measurements.`
        );
    } catch (error) {
        clearRecording();
        fileInput.value = "";

        setStatus(error.message, true);
    } finally {
        busy = false;
        updateControls();
    }
});


// ---------- CALCULATE RMS AMPLITUDE ----------

function calculateRMS(values) {
    // Scale first to avoid squaring very large values directly.
    let largestMagnitude = 0;

    for (const value of values) {
        largestMagnitude = Math.max(
            largestMagnitude,
            Math.abs(value)
        );
    }

    if (largestMagnitude === 0) {
        return 0;
    }

    let normalizedSquares = 0;

    for (const value of values) {
        normalizedSquares += (value / largestMagnitude) ** 2;
    }

    return largestMagnitude *
        Math.sqrt(normalizedSquares / values.length);
}


// ---------- GENERATE GAUSSIAN NOISE ----------

function randomGaussian() {
    // Box-Muller transform: two uniform random numbers
    // produce one normally distributed random value.
    const first = Math.max(Math.random(), Number.MIN_VALUE);
    const second = Math.random();

    return Math.sqrt(-2 * Math.log(first)) *
        Math.cos(2 * Math.PI * second);
}


// ---------- UPDATE THE SIGNAL PREVIEW ----------

function updatePreview() {
    const sliderPosition = Number(noiseSlider.value);
    const snr = noiseConditions[sliderPosition];

    noiseLabel.textContent =
        snr === null ? "Off" : `${snr} dB`;

    if (snr === null) {
        // Copy the uploaded values without adding noise.
        previewSignal = [...originalSignal];
    } else {
        const signalRMS = calculateRMS(originalSignal);

        // Equivalent to noise_power = signal_power / 10^(SNR / 10).
        const noiseStandardDeviation =
            signalRMS / Math.pow(10, snr / 20);

        previewSignal = originalSignal.map(value => {
            return value +
                noiseStandardDeviation * randomGaussian();
        });
    }

    rmsValue.textContent =
        calculateRMS(previewSignal).toFixed(4);

    resetResult();
    drawWaveform();
}

noiseSlider.addEventListener("input", () => {
    if (times.length === 0) {
        return;
    }

    updatePreview();

    const noiseIsOff = Number(noiseSlider.value) === 0;

    setStatus(
        noiseIsOff
            ? "Showing the uploaded recording without added noise."
            : "Preview updated with a new random noise sample. Analyze again."
    );
});

modelSelect.addEventListener("change", () => {
    resetResult();

    if (times.length > 0) {
        setStatus(
            "Model selection changed. Analyze again to get a prediction."
        );
    }
});


// ---------- DRAW THE WAVEFORM ----------

function drawWaveform() {
    const bounds = canvas.getBoundingClientRect();
    const pixelRatio = window.devicePixelRatio || 1;

    // Higher-resolution canvas for high-density displays.
    canvas.width = Math.round(bounds.width * pixelRatio);
    canvas.height = Math.round(bounds.height * pixelRatio);

    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

    const width = bounds.width;
    const height = bounds.height;

    context.clearRect(0, 0, width, height);

    const hasSignal = times.length > 0;

    chartEmpty.hidden = hasSignal;

    if (!hasSignal) {
        canvas.setAttribute(
            "aria-label",
            "Voltage waveform. No recording loaded."
        );

        return;
    }

    canvas.setAttribute(
        "aria-label",
        `Voltage waveform with ${times.length} measurements. ` +
        `Added noise: ${noiseLabel.textContent}.`
    );

    const margin = {
        left: 55,
        right: 22,
        top: 20,
        bottom: 48
    };

    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    const startTime = times[0];
    const endTime = times[times.length - 1];

    // Keep a consistent minimum scale for the example signals.
    let amplitudeLimit = 1.7;

    for (let index = 0; index < times.length; index++) {
        amplitudeLimit = Math.max(
            amplitudeLimit,
            Math.abs(originalSignal[index]) * 1.1,
            Math.abs(previewSignal[index]) * 1.1
        );
    }

    function mapX(value) {
        return margin.left +
            ((value - startTime) / (endTime - startTime)) *
            plotWidth;
    }

    function mapY(value) {
        return margin.top +
            ((amplitudeLimit - value) / (2 * amplitudeLimit)) *
            plotHeight;
    }

    const chartStyle = getComputedStyle(canvas);
    const chartColor = name => chartStyle.getPropertyValue(name).trim();

    context.font = "11px Consolas, monospace";
    context.lineWidth = 1;

    // Horizontal grid lines and amplitude labels.
    context.textAlign = "right";

    for (let index = 0; index <= 4; index++) {
        const value =
            amplitudeLimit - (index / 4) * 2 * amplitudeLimit;

        const y = mapY(value);

        context.strokeStyle = chartColor("--chart-grid");
        context.beginPath();
        context.moveTo(margin.left, y);
        context.lineTo(width - margin.right, y);
        context.stroke();

        context.fillStyle = chartColor("--chart-label");
        context.fillText(
            value.toFixed(1),
            margin.left - 8,
            y + 4
        );
    }

    // Time labels along the horizontal axis.
    context.textAlign = "center";

    for (let index = 0; index <= 4; index++) {
        const value =
            startTime + (index / 4) * (endTime - startTime);

        context.fillText(
            value.toFixed(3),
            mapX(value),
            height - 29
        );
    }

    // Axis titles.
    context.fillText(
        "Time (seconds)",
        margin.left + plotWidth / 2,
        height - 8
    );

    context.save();
    context.translate(14, margin.top + plotHeight / 2);
    context.rotate(-Math.PI / 2);
    context.fillText("Relative amplitude", 0, 0);
    context.restore();

    function drawLine(values, color) {
        context.strokeStyle = color;
        context.lineWidth = 1.4;
        context.beginPath();

        for (let index = 0; index < values.length; index++) {
            const x = mapX(times[index]);
            const y = mapY(values[index]);

            if (index === 0) {
                context.moveTo(x, y);
            } else {
                context.lineTo(x, y);
            }
        }

        context.stroke();
    }

    drawLine(originalSignal, chartColor("--chart-original"));

    if (Number(noiseSlider.value) !== 0) {
        drawLine(previewSignal, chartColor("--chart-preview"));
    }
}

// Redraw when the chart changes size.
const chartObserver = new ResizeObserver(drawWaveform);
chartObserver.observe(canvas);


// ---------- DOWNLOAD THE CURRENT PREVIEW ----------

downloadButton.addEventListener("click", () => {
    if (times.length === 0) {
        return;
    }

    const rows = ["time_seconds,voltage_relative"];

    for (let index = 0; index < times.length; index++) {
        rows.push(`${times[index]},${previewSignal[index]}`);
    }

    const csvContent = rows.join("\n");

    const blob = new Blob(
        [csvContent],
        { type: "text/csv;charset=utf-8" }
    );

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    const baseName = uploadedFileName.replace(/\.csv$/i, "");

    link.href = url;
    link.download = `${baseName}_preview.csv`;

    document.body.appendChild(link);
    link.click();
    link.remove();

    setTimeout(() => {
        URL.revokeObjectURL(url);
    }, 1000);

    setStatus("The displayed signal was downloaded as a CSV.");
});


// ---------- REQUEST A PREDICTION FROM FLASK ----------

analyzeButton.addEventListener("click", async () => {
    if (!cleanedData.checked || times.length === 0 || busy) {
        return;
    }

    resetResult();

    // Local HTML preview supports upload, plotting, noise, and download.
    // Classification requires the page to be served through Flask.
    if (window.location.protocol === "file:") {
        setStatus(
            "To run ML analysis, open this page through the Flask " +
            "application. CSV preview and noise controls work here.",
            true
        );

        return;
    }

    busy = true;
    updateControls();

    analyzeButton.textContent = "Analyzing...";
    setStatus("Sending the displayed signal for classification...");

    const controller = new AbortController();

    const timeout = setTimeout(() => {
        controller.abort();
    }, 30000);

    try {
        const response = await fetch("/api/classify", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            signal: controller.signal,

            body: JSON.stringify({
                time_seconds: times,
                voltage_relative: previewSignal,
                sample_rate_hz: estimatedSampleRate,
                model: modelSelect.value,
                cleaned_data_confirmed: true,
                added_noise_snr_db:
                    noiseConditions[Number(noiseSlider.value)]
            })
        });

        if (!response.ok) {
            throw new Error(
                "The signal-check service is unavailable. Please try again later."
            );
        }

        const result = await response.json();

        const outcome = describePrediction(result.prediction);
        prediction.textContent = outcome;
        resultMessage.textContent = "";

        setStatus("Analysis completed.");
    } catch (error) {
        prediction.textContent = "Analysis unavailable";

        resultMessage.textContent =
            "No result was produced. Please try again when the signal-check service is available.";

        let message = error.message;

        if (error.name === "AbortError") {
            message = "Analysis timed out. Check the Flask server.";
        } else if (error instanceof TypeError) {
            message =
                "Could not reach the classification service. " +
                "Check that Flask is running.";
        }

        setStatus(message, true);
    } finally {
        clearTimeout(timeout);

        busy = false;
        analyzeButton.textContent = "Analyze Signal";

        updateControls();
    }
});


// ---------- INITIAL PAGE STATE ----------

clearRecording();
setStatus("");

// ---------- CLEAN-DATA CONFIRMATION ----------
cleanedData.addEventListener("change", () => {
    if (!cleanedData.checked) {
        fileInput.value = "";
        clearRecording();
        setStatus("");
    } else {
        resetResult();
        setStatus("Ready to upload a cleaned CSV recording.");
    }
    updateControls();
});

// ---------- SIMPLE SIGNAL RESULTS ----------
function describePrediction(label) {
    if (typeof label !== "string") {
        throw new Error("The service did not return a signal type.");
    }
    const key = label.trim().toLowerCase().replace(/[\s-]+/g, "_");
    if (["normal", "normal_signal", "pure_sinusoidal"].includes(key)) {
        return "Normal signal";
    }
    const problems = {
        sag: "Voltage sag", voltage_sag: "Voltage sag",
        swell: "Voltage swell", voltage_swell: "Voltage swell",
        interruption: "Voltage interruption", voltage_interruption: "Voltage interruption",
        harmonic: "Harmonic distortion", harmonics: "Harmonic distortion",
        harmonic_distortion: "Harmonic distortion"
    };
    if (!Object.hasOwn(problems, key)) {
        throw new Error("The service returned an unsupported signal type.");
    }
    return "Problem detected: " + problems[key];
}

// ---------- EMAIL CONTACT PANEL ----------
function setContactOpen(open) {
    contactPanel.hidden = !open;
    contactButton.setAttribute("aria-expanded", String(open));
    if (open) contactClose.focus();
    else contactButton.focus();
}
contactButton.addEventListener("click", () => setContactOpen(contactPanel.hidden));
contactClose.addEventListener("click", () => setContactOpen(false));
document.addEventListener("keydown", event => {
    if (event.key === "Escape" && !contactPanel.hidden) {
        setContactOpen(false);
    }
});
