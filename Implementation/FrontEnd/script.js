// The main application only displays a normal signal or a named disturbance.
const fileInput = document.getElementById("csvFile");
const fileName = document.getElementById("fileName");
const analyzeButton = document.getElementById("analyzeButton");
const resultPanel = document.getElementById("resultPanel");
const resultMark = document.getElementById("resultMark");
const prediction = document.getElementById("prediction");
const resultMessage = document.getElementById("resultMessage");
const statusMessage = document.getElementById("status");
const waveformDetails = document.getElementById("waveformDetails");
const canvas = document.getElementById("waveform");
const chartEmpty = document.getElementById("chartEmpty");
const context = canvas.getContext("2d");

let times = [];
let voltage = [];
let sampleRate = 0;
let busy = false;

function setStatus(message, isError = false) {
    statusMessage.textContent = message;
    statusMessage.classList.toggle("error", isError);
}

function setResult(state, title, message = "") {
    resultPanel.dataset.state = state;
    resultMark.textContent = { idle: "—", pending: "…", normal: "✓", problem: "!", error: "!" }[state];
    prediction.textContent = title;
    resultMessage.textContent = message;
}

function updateControls() {
    fileInput.disabled = busy;
    analyzeButton.disabled = busy || times.length === 0;
    analyzeButton.textContent = busy ? "Please wait…" : "Check signal";
}

// Keep model labels separate from the simple wording shown to the user.
function describePrediction(label) {
    if (typeof label !== "string") {
        throw new Error("The classification response did not contain a signal type.");
    }
    const key = label.trim().toLowerCase().replace(/[\s-]+/g, "_");
    if (["normal", "normal_signal", "pure_sinusoidal"].includes(key)) {
        return { state: "normal", title: "Normal signal" };
    }
    const problems = {
        sag: "Voltage sag", voltage_sag: "Voltage sag",
        swell: "Voltage swell", voltage_swell: "Voltage swell",
        interruption: "Voltage interruption", voltage_interruption: "Voltage interruption",
        harmonic: "Harmonic distortion", harmonics: "Harmonic distortion",
        harmonic_distortion: "Harmonic distortion"
    };
    if (!Object.hasOwn(problems, key)) {
        throw new Error("The classification service returned an unsupported signal type.");
    }
    return { state: "problem", title: `Problem detected: ${problems[key]}` };
}

// CSV format matches the files produced by the Python signal generator.
function parseCSV(text) {
    const cleaned = text.replace(/^\uFEFF/, "").trim();
    if (!cleaned) throw new Error("The selected CSV is empty.");
    const lines = cleaned.split(/\r?\n/).filter(line => line.trim() !== "");
    const header = lines.shift().split(",").map(cell => cell.trim());
    if (header.length !== 2 || header[0] !== "time_seconds" || header[1] !== "voltage_relative") {
        throw new Error("Use the CSV header: time_seconds,voltage_relative");
    }
    if (lines.length < 4 || lines.length > 50000) {
        throw new Error("The recording must contain between 4 and 50,000 measurements.");
    }
    const loadedTimes = [];
    const loadedVoltage = [];
    for (const [index, line] of lines.entries()) {
        const cells = line.split(",").map(cell => cell.trim());
        if (cells.length !== 2 || cells.some(cell => cell === "")) {
            throw new Error(`Measurement ${index + 1} must contain two numbers.`);
        }
        const time = Number(cells[0]);
        const value = Number(cells[1]);
        if (!Number.isFinite(time) || !Number.isFinite(value)) {
            throw new Error(`Measurement ${index + 1} contains an invalid number.`);
        }
        if (index > 0 && time <= loadedTimes[index - 1]) {
            throw new Error("Timestamps must increase with every measurement.");
        }
        loadedTimes.push(time);
        loadedVoltage.push(value);
    }
    const step = (loadedTimes.at(-1) - loadedTimes[0]) / (loadedTimes.length - 1);
    if (!Number.isFinite(step) || step <= 0 || !Number.isFinite(1 / step)) {
        throw new Error("The recording has an invalid time interval.");
    }
    const tolerance = Math.max(step * 0.001, 1e-10);
    for (let index = 1; index < loadedTimes.length; index++) {
        if (Math.abs(loadedTimes[index] - loadedTimes[index - 1] - step) > tolerance) {
            throw new Error("This prototype requires evenly spaced timestamps.");
        }
    }
    return { times: loadedTimes, voltage: loadedVoltage, sampleRate: 1 / step };
}

fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if (!file) return;
    busy = true;
    times = [];
    voltage = [];
    sampleRate = 0;
    fileName.textContent = "Reading recording…";
    setResult("idle", "No result yet", "Loading your recording.");
    setStatus("Reading the selected CSV.");
    updateControls();
    drawWaveform();
    try {
        if (file.size > 5 * 1024 * 1024) throw new Error("Choose a CSV smaller than 5 MB.");
        const recording = parseCSV(await file.text());
        times = recording.times;
        voltage = recording.voltage;
        sampleRate = recording.sampleRate;
        fileName.textContent = file.name;
        setResult("idle", "Ready to check", "Select Check signal to analyze this recording.");
        setStatus("Recording loaded. Ready to check.");
        drawWaveform();
    } catch (error) {
        fileInput.value = "";
        fileName.textContent = "No file selected.";
        setResult("error", "Unable to load recording", error.message);
        setStatus(error.message, true);
    } finally {
        busy = false;
        updateControls();
    }
});

analyzeButton.addEventListener("click", async () => {
    if (busy || times.length === 0) return;
    if (window.location.protocol === "file:") {
        setResult("error", "Signal check unavailable", "Open the application through its local server to check a recording.");
        setStatus("The signal-check service is not connected.", true);
        return;
    }
    busy = true;
    updateControls();
    setResult("pending", "Checking your signal…");
    setStatus("Checking the uploaded recording.");
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);
    try {
        const response = await fetch("/api/classify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: controller.signal,
            body: JSON.stringify({
                time_seconds: times,
                voltage_relative: voltage,
                sample_rate_hz: sampleRate,
                model: "random_forest",
                added_noise_snr_db: null
            })
        });
        if (!response.ok) throw new Error("The signal-check service is not available. Please try again later.");
        const result = await response.json();
        const outcome = describePrediction(result.prediction);
        setResult(outcome.state, outcome.title);
        setStatus("Signal check completed.");
    } catch (error) {
        const message = error.name === "AbortError"
            ? "The signal check timed out. Please try again."
            : error instanceof TypeError
                ? "Could not reach the signal-check service."
                : error instanceof SyntaxError
                    ? "The signal-check service returned an invalid response."
                    : error.message;
        setResult("error", "Signal check unavailable", message);
        setStatus(message, true);
    } finally {
        clearTimeout(timeout);
        busy = false;
        updateControls();
    }
});

// The waveform is optional; it does not replace the classifier's result.
function drawWaveform() {
    chartEmpty.hidden = times.length > 0;
    canvas.setAttribute("aria-label", times.length
        ? "Uploaded voltage waveform."
        : "Voltage waveform. No recording loaded.");
    const bounds = canvas.getBoundingClientRect();
    if (bounds.width < 100 || bounds.height < 100) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(bounds.width * ratio);
    canvas.height = Math.round(bounds.height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, bounds.width, bounds.height);
    if (!times.length) return;
    const left = 55, right = 20, top = 20, bottom = 45;
    const width = bounds.width - left - right;
    const height = bounds.height - top - bottom;
    const colors = getComputedStyle(canvas);
    let limit = 1.7;
    for (const value of voltage) limit = Math.max(limit, Math.abs(value) * 1.1);
    const x = time => left + (time - times[0]) / (times.at(-1) - times[0]) * width;
    const y = value => top + (limit - value) / (2 * limit) * height;
    context.font = "11px Consolas, monospace";
    context.fillStyle = colors.getPropertyValue("--blue-soft").trim();
    context.textAlign = "right";
    context.lineWidth = 1;
    for (let index = 0; index <= 4; index++) {
        const value = limit - index / 4 * 2 * limit;
        context.strokeStyle = colors.getPropertyValue("--navy").trim();
        context.beginPath();
        context.moveTo(left, y(value));
        context.lineTo(bounds.width - right, y(value));
        context.stroke();
        context.fillText(value.toFixed(1), left - 8, y(value) + 4);
    }
    context.textAlign = "center";
    for (let index = 0; index <= 4; index++) {
        const time = times[0] + index / 4 * (times.at(-1) - times[0]);
        context.fillText(time.toFixed(3), x(time), bounds.height - 26);
    }
    context.fillText("Time (seconds)", left + width / 2, bounds.height - 7);
    context.save();
    context.translate(13, top + height / 2);
    context.rotate(-Math.PI / 2);
    context.fillText("Relative amplitude", 0, 0);
    context.restore();
    context.strokeStyle = colors.getPropertyValue("--green").trim();
    context.lineWidth = 1.5;
    context.beginPath();
    voltage.forEach((value, index) => {
        if (index === 0) context.moveTo(x(times[index]), y(value));
        else context.lineTo(x(times[index]), y(value));
    });
    context.stroke();
}

waveformDetails.addEventListener("toggle", drawWaveform);
const chartObserver = new ResizeObserver(drawWaveform);
chartObserver.observe(canvas);
updateControls();
drawWaveform();
