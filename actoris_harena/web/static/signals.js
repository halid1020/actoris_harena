// The Signals tab: what the selected rig's devices are reporting, before any
// session. Everything here goes through /api/agent/*, which the console proxies
// to the rig's own agent process -- this page imports no robot and neither does
// the console. Reading only: the agent opens no control interface.
//
// A collection session holds the devices, so while one runs this tab says so
// and does nothing; two processes opening one camera is the failure that
// wastes an afternoon (the second open succeeds and delivers nothing).

let readTimer = null;
let readBusy = false;
let readingArm = false;
let camerasOpen = false;
let cameraNames = [];

const READ_PERIOD_MS = 200;

function signalsVisible() {
  return !document.querySelector('#pane-signals').hidden;
}

function sessionRunning() {
  return !!(sessionState && sessionState.running);
}

async function post(url) {
  return j(url, {method: 'POST', headers: {'Content-Type': 'application/json'},
                 body: '{}'});
}

function renderArm(body) {
  // armView reads either agent's dialect: one limb for the UR3e, left and
  // right for the dual SO-101.
  const view = armView(body);
  const sides = Object.keys(view.sides);
  const names = sides.length ? Object.keys(view.sides[sides[0]]) : [];
  const num = (v) => (typeof v === 'number') ? v.toFixed(2) : '--';
  $('#s-joints').innerHTML = !sides.length ? '' :
    '<tr><th></th>' + sides.map(s => `<th>${s}</th>`).join('') + '</tr>'
    + names.map(n => `<tr><th>${n}</th>`
      + sides.map(s => `<td>${num(view.sides[s][n])}</td>`).join('') + '</tr>').join('');
  $('#s-readings').innerHTML = view.readings.map(r => {
    const cls = r.ok === false ? 'lvl-FAIL' : (r.ok === true ? 'lvl-OK' : '');
    return `<tr class="${cls}"><th>${r.name}</th><td>${r.value}</td>`
      + `<td class="muted">${r.unit || ''}</td></tr>`;
  }).join('');
  $('#s-arm-state').textContent = !body || !view.running ? 'not reading'
    : !view.has ? 'connected — waiting for the first reading'
    : `reading · ${view.age_s === null ? '' : (view.age_s * 1000).toFixed(0) + ' ms old'}`;
  $('#s-err').textContent = view.problem;
}

function renderCameras(names) {
  if (names.join() === cameraNames.join()) return;
  cameraNames = names;
  const tiles = $('#s-tiles'); tiles.innerHTML = '';
  for (const name of names) {
    // One MJPEG response per camera. A multipart stream never completes, so
    // each holds one of the browser's few connections per origin -- fine for
    // a rig's handful of cameras, and why the src is set once, not per tick.
    const fig = document.createElement('figure');
    fig.className = 'tile';
    const img = document.createElement('img');
    img.alt = name;
    img.src = '/api/agent/cameras/' + encodeURIComponent(name);
    const cap = document.createElement('figcaption');
    cap.textContent = name;
    fig.append(img, cap);
    tiles.appendChild(fig);
  }
}

async function readOnce() {
  if (sessionRunning()) {
    $('#s-note').textContent =
      'a collection session is running and holds the devices — see the Collect tab';
    $('#s-arm').disabled = $('#s-cams').disabled = true;
    return;
  }
  $('#s-note').textContent = '';
  $('#s-arm').disabled = $('#s-cams').disabled = false;
  if (readingArm) renderArm(await j(await agentArmBase()));
  if (camerasOpen) {
    const cams = await j('/api/agent/cameras');
    renderCameras(cams.streams || []);
    const problem = cameraProblem(cams);
    $('#s-cam-state').textContent = problem && !(cams.streams || []).length ? problem
      : (cams.streams || []).length ? `open: ${cams.streams.join(', ')}`
      : 'opening…';
    if (problem && (cams.streams || []).length) $('#s-cam-state').textContent += ` — ${problem}`;
  }
}

async function readTick() {
  if (signalsVisible() && !readBusy) {
    readBusy = true;
    try { await readOnce(); }
    catch (e) { $('#s-err').textContent = e.message; }
    finally { readBusy = false; }
  }
  clearTimeout(readTimer);
  if (signalsVisible()) readTimer = setTimeout(readTick, READ_PERIOD_MS);
}

async function startArm() {
  $('#s-err').textContent = '';
  $('#s-arm-state').textContent = 'connecting…';
  try {
    const r = await post((await agentArmBase()) + '/start');
    readingArm = !!r.started;
    if (!r.started) $('#s-err').textContent = r.problem || 'the arm did not answer';
  } catch (e) {
    // The SO-101's agent answers a bus it cannot open with an HTTP 500.
    readingArm = false;
    $('#s-err').textContent = e.message;
  }
  if (!readingArm) $('#s-arm-state').textContent = 'not reading';
}

async function stopArm() {
  readingArm = false;
  try { await post((await agentArmBase()) + '/stop'); } catch (e) { /* agent may be gone */ }
  renderArm(null);
}

async function openCameras() {
  $('#s-cam-state').textContent = 'opening…';
  try {
    const r = await post('/api/agent/cameras/start');
    camerasOpen = true;
    const problem = cameraProblem(r);
    if (problem) $('#s-cam-state').textContent = problem;
  } catch (e) { camerasOpen = false; $('#s-cam-state').textContent = e.message; }
}

async function closeCameras() {
  camerasOpen = false;
  renderCameras([]);
  try { await post('/api/agent/cameras/stop'); } catch (e) { /* likewise */ }
  $('#s-cam-state').textContent = 'cameras closed';
}

function labelButtons() {
  $('#s-arm').textContent = readingArm ? 'Stop reading' : 'Read the arm';
  $('#s-cams').textContent = camerasOpen ? 'Close cameras' : 'Open cameras';
}

$('#s-arm').onclick = async () => {
  $('#s-arm').disabled = true;
  await (readingArm ? stopArm() : startArm());
  labelButtons();
  $('#s-arm').disabled = false;
  readTick();
};

$('#s-cams').onclick = async () => {
  $('#s-cams').disabled = true;
  await (camerasOpen ? closeCameras() : openCameras());
  labelButtons();
  $('#s-cams').disabled = false;
  readTick();
};

// Showing the tab is asking to see the rig: open both, read only. Not while a
// session holds the devices -- readOnce says so instead.
let autoOpening = false;
async function openOnShow() {
  if (autoOpening || sessionRunning()) return;
  autoOpening = true;
  try {
    // On a page load the picker may not have asked the server yet.
    const rig = selectedRig || (await j('/api/console')).rig;
    if (!rig || !signalsVisible()) return;
    await Promise.all([readingArm ? null : startArm(),
                       camerasOpen ? null : openCameras()]);
  } catch (e) { $('#s-err').textContent = e.message; }
  finally { autoOpening = false; }
  labelButtons();
  readTick();
}

// Leaving the tab lets go of everything it opened: a camera held here is one
// a collection session cannot have.
async function releaseReadings() {
  const was = readingArm || camerasOpen;
  if (!was) return;
  await Promise.all([stopArm(), closeCameras()]);
  labelButtons();
}

const _paneShown = window.onPaneShown;
window.onPaneShown = (name) => {
  if (_paneShown) _paneShown(name);
  if (name === 'signals') { readTick(); openOnShow(); }
  else releaseReadings();
};

// app.js restores the pane from the URL before this file is parsed, so a
// console reloaded on #signals never had the hook above called for it: no
// poll ran, and a button could start the arm with nothing reading it back.
window.addEventListener('load', () => {
  if (signalsVisible()) window.onPaneShown('signals');
});

// A different rig has different devices: drop the old one's.
const _rigChanged = window.onRigChanged;
window.onRigChanged = (rig) => {
  if (_rigChanged) _rigChanged(rig);
  readingArm = camerasOpen = false;
  renderCameras([]); renderArm(null); labelButtons();
  if (signalsVisible()) openOnShow();
};
