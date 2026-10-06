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
  const snap = body && body.snapshot;
  // Joints are per limb, the monitor's own layout; a flat {name: deg} from a
  // rig that reports one limb is drawn as that one limb.
  let joints = (snap && snap.joints) || {};
  if (Object.values(joints).some(v => typeof v === 'number')) joints = {arm: joints};
  const sides = Object.keys(joints);
  const names = sides.length ? Object.keys(joints[sides[0]]) : [];
  const num = (v) => (typeof v === 'number') ? v.toFixed(2) : '--';
  $('#s-joints').innerHTML = !sides.length ? '' :
    '<tr><th></th>' + sides.map(s => `<th>${s} (deg)</th>`).join('') + '</tr>'
    + names.map(n => `<tr><th>${n}</th>`
      + sides.map(s => `<td>${num(joints[s][n])}</td>`).join('') + '</tr>').join('');
  const readings = (snap && snap.readings) || [];
  $('#s-readings').innerHTML = readings.map(r => {
    const cls = r.ok === false ? 'lvl-FAIL' : (r.ok === true ? 'lvl-OK' : '');
    return `<tr class="${cls}"><th>${r.name}</th><td>${r.value}</td>`
      + `<td class="muted">${r.unit || ''}</td></tr>`;
  }).join('');
  const age = snap && snap.t_read ? Date.now() / 1000 - snap.t_read : null;
  $('#s-arm-state').textContent = !body || !body.running ? 'not reading'
    : !snap ? 'connected — waiting for the first reading'
    : `reading · ${age === null ? '' : (age * 1000).toFixed(0) + ' ms old'}`;
  $('#s-err').textContent = (body && body.problem) || '';
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
  if (readingArm) renderArm(await j('/api/agent/arm'));
  if (camerasOpen) {
    const cams = await j('/api/agent/cameras');
    renderCameras(cams.streams || []);
    $('#s-cam-state').textContent = cams.problem ? cams.problem
      : (cams.streams || []).length ? `open: ${cams.streams.join(', ')}`
      : 'opening…';
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

$('#s-arm').onclick = async () => {
  $('#s-arm').disabled = true;
  $('#s-err').textContent = '';
  try {
    if (readingArm) {
      await post('/api/agent/arm/stop');
      readingArm = false;
      renderArm(null);
    } else {
      $('#s-arm-state').textContent = 'connecting…';
      const r = await post('/api/agent/arm/start');
      readingArm = !!r.started;
      if (!r.started) $('#s-err').textContent = r.problem || 'the arm did not answer';
    }
  } catch (e) { $('#s-err').textContent = e.message; }
  $('#s-arm').textContent = readingArm ? 'Stop reading' : 'Read the arm';
  $('#s-arm').disabled = false;
};

$('#s-cams').onclick = async () => {
  $('#s-cams').disabled = true;
  try {
    if (camerasOpen) {
      await post('/api/agent/cameras/stop');
      camerasOpen = false;
      renderCameras([]);
      $('#s-cam-state').textContent = 'cameras closed';
    } else {
      $('#s-cam-state').textContent = 'opening…';
      const r = await post('/api/agent/cameras/start');
      camerasOpen = true;
      if (r.problem) $('#s-cam-state').textContent = r.problem;
    }
  } catch (e) { $('#s-cam-state').textContent = e.message; }
  $('#s-cams').textContent = camerasOpen ? 'Close cameras' : 'Open cameras';
  $('#s-cams').disabled = false;
};

// Leaving the tab lets go of everything it opened: a camera held here is one
// a collection session cannot have.
async function releaseReadings() {
  const was = readingArm || camerasOpen;
  readingArm = camerasOpen = false;
  renderCameras([]);
  $('#s-arm').textContent = 'Read the arm';
  $('#s-cams').textContent = 'Open cameras';
  $('#s-cam-state').textContent = 'cameras closed';
  if (!was) return;
  try { await post('/api/agent/arm/stop'); } catch (e) { /* agent may be gone */ }
  try { await post('/api/agent/cameras/stop'); } catch (e) { /* likewise */ }
}

const _paneShown = window.onPaneShown;
window.onPaneShown = (name) => {
  if (_paneShown) _paneShown(name);
  if (name === 'signals') readTick();
  else releaseReadings();
};

// A different rig has different devices: drop the old one's.
const _rigChanged = window.onRigChanged;
window.onRigChanged = (rig) => {
  if (_rigChanged) _rigChanged(rig);
  readingArm = camerasOpen = false;
  renderCameras([]); renderArm(null);
};
