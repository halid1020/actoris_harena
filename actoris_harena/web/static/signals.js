// The Signals tab: which device is which, and what it is called.
// The routes and the file on disk keep the older 'sensor' name
// (src/conf/sensor_map.yaml, and tool/test_sensor_rates.py writes the same map).
// Everything here opens a device, so it is refused while a session runs.

let sensors = null;
let probedPort = null;
let previewDevice = null;
let tickTimer = null;

function signalsVisible() {
  return !document.querySelector('#pane-signals').hidden;
}

async function loadSignals(scan) {
  try {
    sensors = await j('/api/sensors' + (scan ? '?scan=1' : ''));
  } catch (e) { $('#s-err').textContent = e.message; return; }
  probedPort = sensors.probe_port;
  renderMap();
  renderCandidates();
  $('#s-note').textContent = sensors.busy
    ? 'a session is running — assignment is disabled' : '';
  $('#s-scan').disabled = !!sensors.busy;
}

function renderMap() {
  const o = sensors.overview;
  const mark = (x) => x.device
    ? `<span class="${x.present ? 'ok' : 'bad'}">${x.present ? '●' : '○'}</span>`
    : '<span class="muted">—</span>';
  const short = (d) => d ? d.split('/').pop() : 'unassigned';
  const rows = [];
  rows.push('<h3 class="sub">Cameras</h3><table class="checks">');
  for (const c of o.cameras) {
    rows.push(`<tr><td>${mark(c)}</td><td>${c.name}</td>`
      + `<td class="muted" title="${c.device || ''}">${short(c.device)}</td>`
      + `<td>${c.device ? `<button class="link" data-clear="camera" data-key="${c.name}">clear</button>` : ''}</td></tr>`);
  }
  rows.push('</table><h3 class="sub">Arms</h3><table class="checks">');
  for (const a of o.arms) {
    const kind = a.role === 'follower' ? 'follower' : 'leader';
    rows.push(`<tr><td>${mark(a)}</td><td>${a.role} ${a.side}</td>`
      + `<td class="muted" title="${a.device || ''}">${short(a.device)}</td>`
      + `<td>${a.device ? `<button class="link" data-clear="${kind}" data-key="${a.side}">clear</button>` : ''}</td></tr>`);
  }
  rows.push('</table><h3 class="sub">Depth camera</h3><table class="checks">');
  const rs = o.realsense;
  rows.push(`<tr><td>${mark({device: rs.serial, present: rs.present})}</td>`
    + `<td>${rs.name || 'central'}</td><td class="muted">${rs.serial || 'unassigned'}</td>`
    + `<td>${rs.serial ? '<button class="link" data-clear="realsense" data-key="">clear</button>' : ''}</td></tr>`);
  rows.push('</table>');
  $('#s-map').innerHTML = rows.join('');
  document.querySelectorAll('#s-map [data-clear]').forEach(b => {
    b.onclick = () => clearAssignment(b.dataset.clear, b.dataset.key);
  });
}

// A chip carries the name its device already has, so identifying six
// identical cameras does not start again from nothing every time. The server
// resolves it: the map stores by-path aliases, the chips are /dev/videoN.
function chip(device, attr) {
  const name = (sensors.overview.bound || {})[device];
  const shown = device.split('/').pop();
  return `<button class="chip${name ? ' bound' : ''}" ${attr}="${device}"
    title="${device}">${shown}${name ? ` · ${name}` : ''}</button>`;
}

function boundName(device) {
  return device ? (sensors.overview.bound || {})[device] || null : null;
}

function renderCandidates() {
  const c = sensors.candidates;
  $('#s-cams').innerHTML = c.cameras.length
    ? c.cameras.map(d => chip(d, 'data-cam')).join('')
    : '<span class="muted">no capture devices found — press Scan devices</span>';
  document.querySelectorAll('#s-cams [data-cam]').forEach(b => {
    b.onclick = () => previewCamera(b.dataset.cam);
  });

  $('#s-ports').innerHTML = c.serial.length
    ? c.serial.map(d => chip(d, 'data-port')).join('')
    : '<span class="muted">no serial ports found</span>';
  document.querySelectorAll('#s-ports [data-port]').forEach(b => {
    b.onclick = () => probePort(b.dataset.port);
  });

  $('#s-rs').innerHTML = c.realsense.length
    ? c.realsense.map(d =>
        `<button class="chip${d.serial === sensors.overview.realsense.serial
          ? ' bound' : ''}" data-rs="${d.serial}">${d.name} (${d.serial})</button>`
      ).join('')
    : '<span class="muted">no RealSense device found</span>';
  document.querySelectorAll('#s-rs [data-rs]').forEach(b => {
    b.onclick = () => assignRealsense(b.dataset.rs);
  });

  const current = boundName(previewDevice);
  $('#s-cam-names').innerHTML = sensors.names.map(n =>
    `<button data-name="${n}"${n === current ? ' class="sel"' : ''}>${n}</button>`
  ).join('');
  document.querySelectorAll('#s-cam-names [data-name]').forEach(b => {
    b.onclick = () => assignCamera(b.dataset.name);
  });
  markArmButtons();
}

// The four role/side buttons under the ticks: the probed port's own binding is
// marked, so pressing another one visibly moves the mark to it.
function markArmButtons() {
  const current = boundName(probedPort);
  document.querySelectorAll('#s-ticks [data-role]').forEach(b => {
    const mine = `${b.dataset.role} ${b.dataset.side}` === current;
    b.classList.toggle('sel', mine);
  });
}

// ── Cameras ─────────────────────────────────────────────────────────────────

async function previewCamera(device) {
  $('#s-err').textContent = '';
  try {
    const r = await j('/api/sensors/camera/preview', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({device}),
    });
    previewDevice = device;
    const name = r.streams[0];
    $('#s-cam-view').hidden = false;
    $('#s-cam-cap').textContent = device;
    $('#s-cam-img').src = `/api/live/${encodeURIComponent(name)}.mjpg?t=${Date.now()}`;
    $('#s-work-title').textContent = `showing ${device}`
      + (boundName(device) ? ` — currently ${boundName(device)}` : ' — unassigned');
    $('#s-release').hidden = false;
    renderCandidates();  // the name this device already has is marked
  } catch (e) { $('#s-err').textContent = e.message; }
}

async function assignCamera(name) {
  if (!previewDevice) { $('#s-err').textContent = 'show a camera first'; return; }
  try {
    await j('/api/sensors/camera/assign', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({device: previewDevice, name}),
    });
  } catch (e) { $('#s-err').textContent = e.message; return; }
  await loadSignals(false);
}

// ── Arms ────────────────────────────────────────────────────────────────────

async function probePort(port) {
  $('#s-err').textContent = '';
  try {
    await j('/api/sensors/arm/probe', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({port}),
    });
  } catch (e) { $('#s-err').textContent = e.message; return; }
  probedPort = port;
  $('#s-ticks').hidden = false;
  $('#s-work-title').textContent = `reading ${port} — wiggle one arm`;
  $('#s-release').hidden = false;
  pollTicks();
}

async function pollTicks(rebase) {
  if (!probedPort) return;
  let body;
  try { body = await j('/api/sensors/arm/ticks' + (rebase ? '?rebase=1' : '')); }
  catch (e) { return; }
  $('#s-ticks-table').innerHTML = body.joints.map(r =>
    `<tr class="${r.moving ? 'lvl-OK' : ''}"><td>${r.joint}</td>`
    + `<td>${r.value}</td><td>${r.delta >= 0 ? '+' : ''}${r.delta}</td></tr>`).join('')
    + (body.error ? `<tr><td colspan="3" class="muted">${body.error}</td></tr>` : '');
  clearTimeout(tickTimer);
  if (signalsVisible()) tickTimer = setTimeout(() => pollTicks(false), 300);
}

$('#s-rebase').onclick = () => pollTicks(true);

document.querySelectorAll('#s-ticks [data-role]').forEach(b => {
  b.onclick = async () => {
    if (!probedPort) return;
    try {
      await j('/api/sensors/arm/assign', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({port: probedPort, role: b.dataset.role, side: b.dataset.side}),
      });
    } catch (e) { $('#s-err').textContent = e.message; return; }
    await loadSignals(false);
  };
});

// ── Depth camera, releasing, clearing ───────────────────────────────────────

async function assignRealsense(serial) {
  try {
    await j('/api/sensors/realsense/assign', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({serial}),
    });
  } catch (e) { $('#s-err').textContent = e.message; return; }
  await loadSignals(false);
}

async function clearAssignment(kind, key) {
  try {
    await j('/api/sensors/clear', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({kind, key}),
    });
  } catch (e) { $('#s-err').textContent = e.message; return; }
  await loadSignals(false);
}

async function releaseDevices() {
  clearTimeout(tickTimer);
  if (probedPort) {
    await j('/api/sensors/arm/release',
            {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'})
      .catch(() => {});
    probedPort = null;
  }
  if (previewDevice) {
    await j('/api/preview/stop',
            {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'})
      .catch(() => {});
    previewDevice = null;
    $('#s-cam-img').src = '';
  }
  $('#s-cam-view').hidden = true;
  $('#s-ticks').hidden = true;
  $('#s-release').hidden = true;
  $('#s-work-title').textContent = 'Pick a device to identify';
  await loadSignals(false);
}

$('#s-release').onclick = releaseDevices;
$('#s-scan').onclick = () => loadSignals(true);

// Leaving the tab lets go of whatever it was holding: a camera or an arm bus
// held open here is one a collection session cannot have.
const _paneShown = window.onPaneShown;
window.onPaneShown = (name) => {
  if (_paneShown) _paneShown(name);
  if (name === 'signals') loadSignals(false);
  else if (probedPort || previewDevice) releaseDevices();
};
