// Exercise history rendering and request cancellation through the app script.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function loadApp(fetch, protocol = "http:") {
    const elements = new Map();
    const timers = new Map();
    const downloads = [];
    const blobs = new Map();
    let nextTimer = 0;
    const drawing = new Proxy({}, {
        get: (target, key) => target[key] || (() => {}),
        set: (target, key, value) => { target[key] = value; return true; }
    });

    function createElement() {
        return {
            value: "0", checked: false, textContent: "", open: false,
            hidden: true, disabled: false, children: [], handlers: {},
            classList: { toggle() {} },
            addEventListener(name, handler) { this.handlers[name] = handler; },
            replaceChildren(...children) { this.children = children; },
            appendChild(child) { this.children.push(child); },
            click() { downloads.push({ filename: this.download, blob: blobs.get(this.href) }); },
            remove() {},
            setAttribute() {}, focus() {},
            getContext() { return drawing; },
            getBoundingClientRect() { return { width: 800, height: 365 }; },
            showModal() { this.open = true; },
            close() { this.open = false; this.handlers.close(); }
        };
    }

    function element(id) {
        if (!elements.has(id)) elements.set(id, createElement());
        return elements.get(id);
    }

    const context = vm.createContext({
        document: { getElementById: element, createElement, body: createElement(), addEventListener() {} },
        window: { devicePixelRatio: 1, location: { protocol } },
        ResizeObserver: class { observe() {} },
        getComputedStyle: () => ({ getPropertyValue: () => "#111111" }),
        AbortController, fetch,
        Blob,
        URL: {
            createObjectURL(blob) {
                const url = `blob:test-${blobs.size}`;
                blobs.set(url, blob);
                return url;
            },
            revokeObjectURL(url) { blobs.delete(url); }
        },
        setTimeout(callback) { timers.set(++nextTimer, callback); return nextTimer; },
        clearTimeout(id) { timers.delete(id); }
    });
    const source = fs.readFileSync(path.join(__dirname, "..", "script.js"), "utf8");
    vm.runInContext(source, context);
    return { context, element, timers, downloads, blobs };
}

function response(history) {
    return { ok: true, json: async () => ({ history }) };
}

function record(overrides = {}) {
    return {
        id: 1, analyzed_at_utc: "2026-11-11T18:00:00+00:00", prediction: "sag",
        model: "random_forest", added_noise_snr_db: null, analysis_time_ms: 1.25,
        ...overrides
    };
}

test("renders empty history, then displays saved summaries after refresh", async () => {
    let history = [];
    const app = loadApp(async (url, options) => {
        assert.equal(url, "/api/history?limit=20");
        assert.equal(options.cache, "no-store");
        return response(history);
    });
    await app.context.loadHistory();
    assert.match(app.element("historyStatus").textContent, /No saved analyses/);
    assert.equal(app.element("historyTableContainer").hidden, true);
    assert.equal(app.element("historyDownload").disabled, true);
    history = [record(), record({ prediction: "normal", model: "svm", added_noise_snr_db: 20 })];
    await app.element("historyRefresh").handlers.click();
    const rows = app.element("historyRows").children;
    assert.equal(rows.length, 2);
    assert.equal(rows[0].children[1].textContent, "Problem detected: Voltage sag");
    assert.equal(rows[0].children[2].textContent, "Random Forest");
    assert.equal(rows[0].children[3].textContent, "Off");
    assert.equal(rows[0].children[4].textContent, "1.3 ms");
    assert.equal(rows[1].children[1].textContent, "Normal signal");
    assert.equal(rows[1].children[2].textContent, "SVM");
    assert.equal(rows[1].children[3].textContent, "20 dB");
    assert.equal(app.element("historyTableContainer").hidden, false);
    assert.equal(app.element("historyDownload").disabled, false);
    assert.equal(app.timers.size, 0);
});

test("a service failure clears stale rows and permits retry", async () => {
    let failing = false;
    const app = loadApp(async () => {
        if (failing) throw new Error("offline");
        return response([record()]);
    });
    await app.context.loadHistory();
    failing = true;
    await app.context.loadHistory();
    assert.equal(app.element("historyRows").children.length, 0);
    assert.equal(app.element("historyTableContainer").hidden, true);
    assert.match(app.element("historyStatus").textContent, /Could not load history/);
    assert.equal(app.element("historyRefresh").disabled, false);
    assert.equal(app.element("historyDownload").disabled, true);
    failing = false;
    await app.context.loadHistory();
    assert.equal(app.element("historyRows").children.length, 1);
});

test("closing and reopening prevents the previous request replacing fresh rows", async () => {
    let resolvePrevious;
    let calls = 0;
    const app = loadApp(async () => {
        calls++;
        if (calls === 1) return new Promise(resolve => { resolvePrevious = resolve; });
        return response([record({ prediction: "swell" })]);
    });
    const pending = app.context.loadHistory();
    app.element("historyDialog").handlers.close();
    await app.context.loadHistory();
    resolvePrevious(response([record({ prediction: "interruption" })]));
    await pending;
    assert.equal(app.element("historyRows").children[0].children[1].textContent, "Problem detected: Voltage swell");
    assert.equal(app.element("historyRefresh").disabled, false);
});

test("a timed out request offers retry and restores the refresh control", async () => {
    const app = loadApp((url, options) => new Promise((resolve, reject) => {
        options.signal.addEventListener("abort", () => reject({ name: "AbortError" }));
    }));
    const pending = app.context.loadHistory();
    const timeout = [...app.timers.values()][0];
    timeout();
    await pending;
    assert.match(app.element("historyStatus").textContent, /timed out/);
    assert.equal(app.element("historyRefresh").disabled, false);
});

test("a direct HTML preview explains how to reach saved history", async () => {
    const app = loadApp(() => { throw new Error("Unexpected network call"); }, "file:");
    await app.context.loadHistory();
    assert.match(app.element("historyStatus").textContent, /local server/);
    assert.equal(app.element("historyTableContainer").hidden, true);
});

test("exports exactly the shown records with UTC dates and unrounded durations", async () => {
    const app = loadApp(async () => response([
        record({ id: 9, analysis_time_ms: 1.23456 }),
        record({ id: 8, model: "svm", added_noise_snr_db: 30 })
    ]));
    await app.context.loadHistory();
    app.element("historyDownload").handlers.click();
    assert.equal(app.downloads.length, 1);
    const download = app.downloads[0];
    assert.equal(download.filename, "analysis_history.csv");
    assert.equal(download.blob.type, "text/csv;charset=utf-8");
    const lines = (await download.blob.text()).split("\r\n");
    assert.equal(lines.length, 3);
    assert.equal(lines[0], "id,analyzed_at_utc,prediction,model,added_noise_snr_db,analysis_time_ms");
    assert.equal(lines[1], '"9","2026-11-11T18:00:00+00:00","sag","random_forest","","1.23456"');
    assert.equal(lines[2], '"8","2026-11-11T18:00:00+00:00","sag","svm","30","1.25"');
    for (const callback of [...app.timers.values()]) callback();
    assert.equal(app.blobs.size, 0);
    app.element("historyDialog").handlers.close();
    assert.equal(app.element("historyDownload").disabled, true);
    app.element("historyDownload").handlers.click();
    assert.equal(app.downloads.length, 1);
});

test("refresh clears the previous export snapshot while a newer request is pending", async () => {
    let complete;
    let calls = 0;
    const app = loadApp(async () => {
        if (++calls === 1) return response([record()]);
        return new Promise(resolve => { complete = resolve; });
    });
    await app.context.loadHistory();
    const pending = app.context.loadHistory();
    assert.equal(app.element("historyDownload").disabled, true);
    app.element("historyDownload").handlers.click();
    assert.equal(app.downloads.length, 0);
    complete(response([]));
    await pending;
    assert.equal(app.element("historyDownload").disabled, true);
});

test("the shared download helper preserves the waveform CSV format and filename", async () => {
    const app = loadApp(async () => response([]));
    vm.runInContext("times = [0, 0.01]; previewSignal = [0, -0.5]; uploadedFileName = 'sample.csv';", app.context);
    app.element("downloadButton").handlers.click();
    assert.equal(app.downloads[0].filename, "sample_preview.csv");
    assert.equal(await app.downloads[0].blob.text(), "time_seconds,voltage_relative\n0,0\n0.01,-0.5");
});
