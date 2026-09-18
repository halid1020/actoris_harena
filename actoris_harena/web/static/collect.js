// The Collect tab: rig readiness, the live view, and one collection session.
// The console never opens a device while a session runs — every tile here is
// either the session's own frames, proxied, or the console's idle preview.

let sessionState = null;
let liveStreams = [];
let liveTiles = {};
let controlsByMode = {};
let collectTimer = null;
let tileTimer = null;
let tileBusy = false;

// The tiles are moving pictures, so they refresh far more often than the
// session status does. They can afford to: one request carries every camera.
const TILE_PERIOD_MS = 100;

function collectVisible() {
  return !document.querySelector('#pane-collect').hidden;
}

async function loadCollectConfig() {
  const cfg = await j('/api/collect/config');
  rigSchema = cfg.schema || null;
  controlsByMode = cfg.controls || {};
  renderControls(controlsByMode[$('#c-input').value]);
  const box = $('#c-streams');
  box.innerHTML = '<legend>Camera streams</legend>';
  for (const cam of cfg.cameras) {
    const id = 'cam-' + cam.name;
    const absent = cam.present === false;
    const label = document.createElement('label');
    label.className = 'row';
    // A camera whose assigned device node is gone is shown, but never ticked:
    // the recorder opens every selected camera before it creates the dataset,
    // so a session started on one exits at once. The box stays enabled, so
    // plugging it back in and pressing Check is all it takes.
    label.innerHTML = `<input type="checkbox" id="${id}" value="${cam.name}"`
      + `${cam.enabled && !absent ? ' checked' : ''}> ${cam.name}`
      + (absent ? ' <span class="stream-absent">— not connected</span>' : '');
    box.appendChild(label);
  }
}

function pickedStreams() {
  return [...document.querySelectorAll('#c-streams input:checked')].map(c => c.value);
}

function sessionRequest() {
  return {
    name: $('#c-name').value.trim(),
    task: $('#c-task').value.trim(),
    streams: pickedStreams(),
    depth: $('#c-depth').checked,
    ee: $('#c-ee').checked,
    input: $('#c-input').value,
    sensor_view: $('#c-view').checked,
  };
}

function describePlan(plan) {
  const bits = [
    plan.resuming ? 'resume' : 'new dataset',
    `cameras ${plan.cameras.join('+') || 'none'}`,
    `depth ${plan.depth ? 'on' : 'off'}`,
    `ee ${plan.ee ? 'on' : 'off'}`,
    `fps ${plan.fps || 'default'}`,
  ];
  return bits.join(' · ') + (plan.warnings.length ? '\n' + plan.warnings.join('\n') : '');
}

async function checkPlan() {
  $('#c-err').textContent = ''; $('#c-plan').textContent = '';
  let plan;
  try {
    plan = await j('/api/session/plan', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(sessionRequest()),
    });
  } catch (e) { $('#c-err').textContent = e.message; return null; }
  $('#c-plan').textContent = describePlan(plan);
  $('#c-err').textContent = plan.refusals.join('\n');
  // A resumed dataset dictates its own streams; showing them ticked keeps the
  // form honest about what will actually be recorded.
  if (plan.resuming) {
    document.querySelectorAll('#c-streams input').forEach(c => {
      c.checked = plan.cameras.includes(c.value);
      c.disabled = true;
    });
    $('#c-depth').checked = plan.depth; $('#c-ee').checked = plan.ee;
  } else {
    document.querySelectorAll('#c-streams input').forEach(c => { c.disabled = false; });
  }
  return plan;
}

$('#c-check').onclick = checkPlan;
$('#c-name').onchange = checkPlan;
$('#c-input').onchange = () => {
  renderControls(controlsByMode[$('#c-input').value]);
  if ($('#c-name').value.trim()) checkPlan();
};

$('#c-start').onclick = async () => {
  const plan = await checkPlan();
  if (!plan || plan.refusals.length) return;
  $('#c-start').disabled = true;
  try {
    await j('/api/session/start', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(sessionRequest()),
    });
  } catch (e) { $('#c-err').textContent = e.message; }
  $('#c-start').disabled = false;
  await pollSession();
};

async function pressEpisode() {
  try {
    await j('/api/session/episode',
            {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
  } catch (e) { alert('episode key failed: ' + e.message); }
  await pollSession();
}

// Enabling the arms moves them, so it lives where the operator can see them --
// the headset, in Quest mode. A leader session has no headset, and the session
// the console started has no keyboard of its own either (it is given no stdin,
// precisely so it cannot fight the console's terminal for one), so there the
// page is the only surface. The session decides: a mode that does not allow the
// key answers with its own refusal.
async function enableArms() {
  const ok = await confirmDialog({
    title: 'Enable both arms?',
    body: 'Torque goes on and both followers MOVE to the ready pose and hold '
      + 'it. Stand clear of the arms before confirming.',
    confirmLabel: 'Enable arms',
  });
  if (!ok) return;
  try {
    await j('/api/session/key', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({key: 'y'}),
    });
  } catch (e) { alert('enable failed: ' + e.message); }
  await pollSession();
}

// What ending the session does depends on what it is doing, and a confirmation
// that describes a session the operator can see they are not in is worse than
// none: it teaches them to click through the next one.
function stopBody(s) {
  const rec = (s && s.monitor && s.monitor.recorder) || {};
  const held = rec.episodes_done
    ? `${rec.episodes_done} recording(s) are already saved. `
    : '';
  return rec.recording
    ? `${held}The episode in progress (${rec.current_frames || 0} frames) is `
      + 'saved first, then the arms are parked and the dataset closed.'
    : `${held}No episode is in progress. The arms are parked and the dataset `
      + 'closed.';
}

async function stopSession() {
  const ok = await confirmDialog({
    title: 'End the collection session?',
    body: stopBody(sessionState),
    confirmLabel: 'Stop session',
  });
  if (!ok) return;
  try {
    await j('/api/session/stop',
            {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
  } catch (e) { alert('stop failed: ' + e.message); }
  await pollSession();
}

$('#c-episode').onclick = pressEpisode;
$('#c-enable').onclick = enableArms;
$('#c-stop').onclick = stopSession;

// The same three, typed. An operator driving a leader session is at this page
// with their hands on a keyboard, and the session no longer reads one of its
// own -- so a key here does exactly what its button does, confirmations
// included. Which keys count is the SESSION's answer, not ours: a Quest session
// ignores Y here for the same reason its monitor refuses it.
// Through the buttons rather than past them, so a key cannot do what a click
// cannot: a disabled button (no episode before the arms are enabled) ignores
// both alike, with no second copy of the rule to keep in step.
const KEY_BUTTONS = {y: '#c-enable', a: '#c-episode', q: '#c-stop'};

document.addEventListener('keydown', (e) => {
  if (!sessionState || !sessionState.running) return;
  if ($('#pane-collect').hidden) return;
  if (e.ctrlKey || e.altKey || e.metaKey) return;
  // Never take a keystroke away from something being typed into.
  if (e.target.closest('input, select, textarea, [contenteditable]')) return;
  const key = (e.key || '').toLowerCase();
  const allowed = (sessionState.monitor || {}).allowed_keys || [];
  if (!allowed.includes(key) || !KEY_BUTTONS[key]) return;
  const button = $(KEY_BUTTONS[key]);
  if (!button || button.hidden || button.disabled) return;
  e.preventDefault();
  button.click();
});

$('#c-arms').onclick = async () => {
  const on = $('#c-arms').dataset.on === '1';
  $('#c-arms').disabled = true;
  try {
    await j(on ? '/api/preview/arms/stop' : '/api/preview/arms/start',
            {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
  } catch (e) { alert(e.message); }
  $('#c-arms').disabled = false;
  await pollSession();
};

$('#c-preview').onclick = async () => {
  const on = $('#c-preview').dataset.on === '1';
  try {
    await j(on ? '/api/preview/stop' : '/api/preview/start',
            {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
  } catch (e) { alert(e.message); }
  await refreshTiles();
};

// ── Controls and signals ────────────────────────────────────────────────────

// The steps come from the session (common/recording/controls.py), which is also
// what the terminal prints, so this pane cannot describe a rig it is not
// driving. Anything that moves an arm names the surface it lives on.
function renderControls(steps) {
  if (!steps || !steps.length) return;
  $('#c-controls').innerHTML = steps.map(s => {
    const here = s.where.includes('this page');
    return `<tr class="${here ? 'lvl-OK' : ''}">
      <td>${s.key || '·'}</td><td>${s.what}</td>
      <td class="muted">${s.where}</td></tr>`;
  }).join('');
}

// The rig's own limbs and joints, from `rig.yaml` by way of /api/collect/config
// and, once a session is up, from the monitor's own payload. NOT written down
// here: this console serves a dual SO-101 with twelve channels and a single
// UR3e with seven, and a hard-coded column list draws one of them wrong.
let rigSchema = null;

function jointRows(schema) {
  const rows = (schema && schema.body_joints) ? schema.body_joints.slice() : [];
  if (!schema || schema.gripper !== false) rows.push('gripper');
  return rows;
}

function limbs(schema) {
  return (schema && schema.limbs && schema.limbs.length) ? schema.limbs : ['arm'];
}

// The table is drawn whether or not a session is running: an empty one saying
// where the numbers come from is honest, while a pane that simply omits the
// arm reads as a console that cannot see it.
function renderJoints(monitor) {
  const sig = monitor && monitor.joints;  // 'j' is the fetch helper here
  const preview = !!(monitor && monitor.source === 'preview');
  // The monitor reports the schema beside the joints, so a running session is
  // self-describing; before one starts, the rig's declaration is what we have.
  const schema = (monitor && monitor.schema) || rigSchema;
  const sides = limbs(schema);
  const num = (v) => (v === null || v === undefined) ? '--' : v.toFixed(2);
  // A command older than a frame is what makes the recorded action fall back to
  // the measured state, so a stale column is dimmed rather than left to look
  // like a live one.
  const cmdClass = (side) => (sig && sig[side] && sig[side].fresh) ? '' : ' class="rate"';
  const at = (side, half, name) =>
    (sig && sig[side] && sig[side][half]) ? sig[side][half][name] : null;
  let html = '<tr><th></th>'
    + sides.map(side => `<th colspan="2">${side}</th>`).join('') + '</tr>'
    + '<tr><th></th>'
    + sides.map(() => '<th>state</th><th>cmd</th>').join('') + '</tr>';
  for (const name of jointRows(schema)) {
    html += `<tr><th>${name}</th>` + sides.map(side =>
      `<td>${num(at(side, 'state', name))}</td>`
      + `<td${cmdClass(side)}>${num(at(side, 'command', name))}</td>`).join('')
      + '</tr>';
  }
  $('#c-joints').innerHTML = html;

  if (!sig) {
    $('#c-teleop').textContent =
      'no session — start one, or read the arms from here';
    $('#c-drift').textContent = '';
    return;
  }
  if (preview) {
    $('#c-teleop').textContent = 'reading the arms — torque off, nothing commanded';
    $('#c-drift').textContent =
      `last read ${((monitor.joint_drift_s || 0) * 1000).toFixed(0)} ms ago`;
    return;
  }
  const leader = sessionState && sessionState.input === 'leader';
  const one = sides.length === 1;
  $('#c-teleop').textContent = monitor.teleop_active
    ? (leader ? 'following the leaders' : 'teleoperating')
    : (leader ? 'not following — enable the arms first'
              : `not teleoperating — hold ${one ? 'the grip' : 'both grips'}`);
  const drift = monitor.joint_drift_s;
  const stale = sides.filter(side => sig[side] && !sig[side].fresh);
  $('#c-drift').textContent =
    (drift === null || drift === undefined
      ? 'joints --' : `joints ${(drift * 1000).toFixed(0)} ms`)
    + (stale.length ? ` · no fresh command: ${stale.join(', ')}` : '');
}

// ── Live tiles ──────────────────────────────────────────────────────────────

async function refreshTiles() {
  let body;
  try { body = await j('/api/live/frames'); }
  catch (e) { return; }
  const names = body.streams || [];
  const missing = body.missing || [];
  const preview = body.source === 'preview';
  // Say which camera is absent and why. A missing tile on its own looks like a
  // console fault; named, it is a camera to replug or a bus to unload.
  const note = $('#live-missing');
  if (note) {
    note.textContent = missing.length
      ? missing.map(m => `${m.name}: ${m.reason}`).join(' · ')
      : '';
    note.hidden = !missing.length;
  }
  $('#c-preview').dataset.on = (preview && names.length) ? '1' : '0';
  $('#c-preview').textContent = (preview && names.length)
    ? 'Stop preview' : 'Start preview';
  $('#c-preview').disabled = !preview;
  $('#live-source').textContent = names.length
    ? (preview ? 'preview (no session running)' : 'live from the session')
    : 'no live view';
  // Rebuild the tile elements only when the stream SET changes; the pictures
  // themselves are repainted every tick below.
  if (names.join() !== liveStreams.join()) {
    liveStreams = names;
    liveTiles = {};
    const tiles = $('#live-tiles'); tiles.innerHTML = '';
    for (const name of names) {
      const fig = document.createElement('figure');
      fig.className = 'tile';
      const img = document.createElement('img');
      img.alt = name;
      const cap = document.createElement('figcaption');
      cap.textContent = name;
      fig.append(img, cap);
      tiles.appendChild(fig);
      liveTiles[name] = img;
    }
  }
  // A name absent from the batch keeps whatever its tile last showed: a camera
  // merely between frames should not make its tile flicker.
  const frames = body.frames || {};
  for (const name in frames) {
    const img = liveTiles[name];
    if (img) img.src = 'data:image/jpeg;base64,' + frames[name];
  }
}

// One request in flight at a time, and none at all while the tab is off screen
// — the same discipline as collectTick, for the same reason: a slow tick must
// never be allowed to queue up behind itself.
async function tileTick() {
  if (collectVisible() && !tileBusy) {
    tileBusy = true;
    try { await refreshTiles(); }
    catch (e) { /* the next tick tries again */ }
    finally { tileBusy = false; }
  }
  clearTimeout(tileTimer);
  tileTimer = setTimeout(tileTick, TILE_PERIOD_MS);
}

// ── Session polling ─────────────────────────────────────────────────────────

function describeStatus(s) {
  if (!s.running) return 'no session';
  const m = s.monitor;
  if (!m) return `session running (pid ${s.pid}) — waiting for its live monitor`;
  const rec = m.recorder;
  const bits = [`arms ${m.arms}`];
  if (rec) {
    bits.push(rec.state);
    bits.push(rec.episodes_goal
      ? `episodes ${rec.episodes_done}/${rec.episodes_goal}`
      : `episodes ${rec.episodes_done}`);
    if (rec.recording) bits.push(`${rec.current_frames} frames`);
  }
  const stale = (m.streams || []).filter(x => x.age_s === null || x.age_s > 1.0);
  if (stale.length) bits.push(`stale: ${stale.map(x => x.name).join(', ')}`);
  return bits.join(' · ');
}

async function pollSession() {
  let s;
  try { s = await j('/api/session'); } catch (e) { return; }
  sessionState = s;
  $('#session-state').textContent = describeStatus(s);
  $('#session-form').hidden = s.running;
  $('#session-live').hidden = !s.running;
  if (s.running) {
    $('#c-running').textContent =
      `${s.resuming ? 'resuming' : 'recording into'} '${s.name}' — ${s.task}\n`
      + `cameras ${(s.cameras || []).join('+')} · depth ${s.depth ? 'on' : 'off'}`
      + ` · ee ${s.ee ? 'on' : 'off'}`;
    const m = s.monitor, rec = m && m.recorder;
    const armed = !!m && m.arms === 'ENABLED';
    const leading = s.input === 'leader';
    $('#c-enable').hidden = !leading;
    $('#c-enable').disabled = armed;
    $('#c-enable').title = armed ? 'the arms are already enabled' : '';
    $('#c-surface').textContent = leading
      ? 'This session is driven by the leader arms; its control keys are read '
        + 'from the terminal that started it, so the ones this page may press '
        + 'are here. Stand clear before enabling.'
      : 'Everything below that moves an arm stays on the headset or the '
        + 'keyboard at the rig — stand clear before it does.';
    $('#c-episode').disabled = !armed;
    $('#c-episode').title = armed ? ''
      : (leading ? 'enable the arms first'
                 : 'enable the arms first (button Y on the headset)');
    // The keys are worth saying: an operator with hands on a keyboard should
    // not have to discover that the page takes them.
    const keys = (m && m.allowed_keys ? m.allowed_keys : [])
      .filter(k => KEY_BUTTONS[k]).map(k => k.toUpperCase());
    $('#c-keys').textContent = keys.length
      ? `${keys.join(' · ')} work as keys on this page too` : '';
    $('#c-episode').textContent = (rec && rec.recording)
      ? 'Stop episode and save' : 'Start episode';
    $('#c-stop').disabled = !!s.stopping;
    $('#c-stop').textContent = s.stopping ? 'Stopping…' : 'Stop session';
  }
  renderJoints(s.monitor || s.preview_joints);
  $('#c-arms').hidden = !!s.running;
  $('#c-arms').dataset.on = s.preview_joints ? '1' : '0';
  $('#c-arms').textContent = s.preview_joints ? 'Stop reading' : 'Read the arms';
  if (s.running) renderControls(s.controls);
  $('#session-log').textContent = (s.tail || []).slice(-200).join('\n');
  $('#session-log').scrollTop = $('#session-log').scrollHeight;
}

async function loadPreflight() {
  $('#preflight-body').textContent = 'checking…';
  let body;
  try { body = await j('/api/preflight'); }
  catch (e) { $('#preflight-body').textContent = e.message; return; }
  const rows = body.checks.map(c =>
    `<tr class="lvl-${c.level}"><td>${c.level}</td><td>${c.name}</td>`
    + `<td>${c.detail}</td></tr>`).join('');
  $('#preflight-body').innerHTML =
    (body.hardware ? '' : '<p class="muted">devices are in use — file checks only</p>')
    + `<table class="checks">${rows}</table>`;
}

$('#preflight').ontoggle = () => { if ($('#preflight').open) loadPreflight(); };

// Poll only while the tab is on screen: the live tiles are streams, and the
// status is only interesting to someone looking at it. A running session is
// polled four times as often, because the joint table is worth watching move;
// one request is in flight at a time, so a slow monitor cannot queue them up.
let collectBusy = false;

async function collectTick() {
  if (collectVisible() && !collectBusy) {
    collectBusy = true;
    try { await pollSession(); }
    catch (e) { /* the next tick tries again */ }
    finally { collectBusy = false; }
  }
  const period = (sessionState && sessionState.running && collectVisible())
    ? 400 : 1500;
  clearTimeout(collectTimer);
  collectTimer = setTimeout(collectTick, period);
}

window.addEventListener('load', () => {
  loadCollectConfig().catch(e => { $('#c-err').textContent = e.message; });
  collectTimer = setTimeout(collectTick, 300);
  tileTimer = setTimeout(tileTick, 300);
  // app.js chooses the pane before this file is parsed, so a console opened
  // straight on #collect has to be told once, here.
  if (collectVisible()) window.onPaneShown('collect');
});

window.onPaneShown = (name) => {
  if (name === 'collect') {
    // The dataset names are the same list the Datasets tab shows.
    const list = $('#c-names');
    list.innerHTML = datasets.map(d => `<option value="${d.name}">`).join('');
    pollSession();
  }
};
