// Integration harness for real script.js handlers and a running local API.
// Uses DOM/canvas stubs; this does not replace manual browser upload checks.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..', '..');
const nodes = new Map();
const drawing = new Proxy({}, { get: (target, key) => target[key] || (() => {}), set: (target, key, value) => (target[key] = value, true) });
function element(id) {
  if (!nodes.has(id)) nodes.set(id, {
    value: '0', checked: false, textContent: '', hidden: true,
    disabled: false, files: [], handlers: {},
    classList: { toggle() {} },
    addEventListener(event, handler) { this.handlers[event] = handler; },
    setAttribute() {}, focus() {},
    getContext() { return drawing; },
    getBoundingClientRect() { return { width: 800, height: 365 }; },
  });
  return nodes.get(id);
}
const sandbox = {
  document: { getElementById: element, addEventListener() {} },
  window: { devicePixelRatio: 1, location: { protocol: 'http:' } },
  ResizeObserver: class { observe() {} },
  getComputedStyle: () => ({ getPropertyValue: () => '#111111' }),
  setTimeout, clearTimeout, AbortController,
  fetch: (url, options) => fetch((process.argv[2] || 'http://127.0.0.1:5000') + url, options),
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(root, 'FrontEnd/script.js'), 'utf8'), sandbox);
(async () => {
  element('cleanedData').checked = true;
  element('cleanedData').handlers.change();
  assert.equal(element('csvFile').disabled, false);
  const dir = path.join(root, 'Data/Synthetic/examples');
  for (const filename of fs.readdirSync(dir).filter(name => name.endsWith('.csv'))) {
    const content = fs.readFileSync(path.join(dir, filename), 'utf8');
    element('csvFile').files = [{ name: filename, size: Buffer.byteLength(content), text: async () => content }];
    await element('csvFile').handlers.change();
    assert.equal(element('sampleCount').textContent, '1,200');
    assert.equal(element('sampleRate').textContent, '6000.0 Hz');
    assert.equal(element('analyzeButton').disabled, false);
    assert.equal(element('chartEmpty').hidden, true);
    assert.ok(Number(element('rmsValue').textContent) > 0);
    const expected = {
      'normal_signal.csv': 'Normal signal',
      'sag_signal.csv': 'Problem detected: Voltage sag',
      'swell_signal.csv': 'Problem detected: Voltage swell',
      'interruption_signal.csv': 'Problem detected: Voltage interruption',
      'harmonic_distortion_signal.csv': 'Problem detected: Harmonic distortion',
    };
    for (const model of ['random_forest', 'svm']) {
      element('modelSelect').value = model;
      element('modelSelect').handlers.change();
      await element('analyzeButton').handlers.click();
      assert.equal(element('prediction').textContent, expected[filename], element('resultMessage').textContent);
      assert.equal(element('status').textContent, 'Analysis completed.');
      assert.equal(element('analyzeButton').disabled, false);
      console.log(`${filename}, ${model}: ${element('prediction').textContent}`);
    }
  }
  const content = fs.readFileSync(path.join(root, 'normal_signal.csv'), 'utf8');
  element('csvFile').files = [{name: 'normal_signal.csv', size: Buffer.byteLength(content), text: async () => content}];
  await element('csvFile').handlers.change();
  element('noiseSlider').value = '1';
  element('noiseSlider').handlers.input();
  await element('analyzeButton').handlers.click();
  assert.equal(element('status').textContent, 'Analysis completed.');
  console.log('Original normal CSV and noisy preview complete live inference.');
  const shortCsv = content.split('\n').slice(0, 101).join('\n');
  element('csvFile').files = [{name: 'short.csv', size: Buffer.byteLength(shortCsv), text: async () => shortCsv}];
  await element('csvFile').handlers.change();
  assert.equal(element('analyzeButton').disabled, true);
  assert.match(element('resultMessage').textContent, /1,200/);
  console.log('Unsupported CSV stays previewable and is blocked before classification.');
  element('cleanedData').checked = false;
  element('cleanedData').handlers.change();
  assert.equal(element('analyzeButton').disabled, true);
  assert.equal(element('sampleCount').textContent, 'No data');
  console.log('All five generated CSVs pass the app upload workflow; clearing still works.');
})().catch(error => { console.error(error); process.exitCode = 1; });
