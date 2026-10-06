// The header's rig picker: which robot this console drives, and registering a
// new one by its directory. The console imports no robot -- a rig is a
// directory with a rig.yaml, reached through its own interpreter -- so all a
// page needs is the list /api/console gives and the two routes that change it.

const REGISTER = '__register__';
let consoleRigs = [];
let selectedRig = null;

function rigTitle(name) {
  const rig = consoleRigs.find(r => r.name === name);
  return rig ? rig.title : null;
}

function renderRigPicker() {
  const pick = $('#rigpick');
  // No silent default, the server's own rule: a console that picked the only
  // rig would look the same as one that picked the wrong one.
  const opts = selectedRig ? [] : ['<option value="">choose a rig…</option>'];
  for (const r of consoleRigs) {
    opts.push(`<option value="${r.name}"${r.name === selectedRig ? ' selected' : ''}>`
      + `${r.title} (${r.name})</option>`);
  }
  opts.push(`<option value="${REGISTER}">Register a rig…</option>`);
  pick.innerHTML = opts.join('');
  const title = rigTitle(selectedRig);
  $('#topbar .brand').textContent = title ? `Rig console — ${title}` : 'Rig console';
  document.title = title ? `${title} — rig console` : 'Rig console';
}

async function loadRigs() {
  const body = await j('/api/console');
  consoleRigs = body.rigs || [];
  selectedRig = body.rig;
  renderRigPicker();
}

async function selectRig(name) {
  const previous = selectedRig;
  try {
    await j('/api/console/rig', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({rig: name || null}),
    });
    selectedRig = name || null;
    armBase = null;
  } catch (e) {
    // A running job or session refuses the switch (409); say why, stay put.
    selectedRig = previous;
    $('#rigpick').title = e.message;
    alert(e.message);
  }
  renderRigPicker();
  if (selectedRig !== previous && window.onRigChanged) window.onRigChanged(selectedRig);
}

$('#rigpick').onchange = () => {
  const value = $('#rigpick').value;
  if (value === REGISTER) {
    renderRigPicker();  // put the select back on the current rig
    $('#rig-err').textContent = '';
    $('#dlg-rig').showModal();
    return;
  }
  selectRig(value);
};

$('#rig-go').onclick = async () => {
  $('#rig-err').textContent = '';
  let rig;
  try {
    rig = await j('/api/console/rigs', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({root: $('#rig-root').value.trim()}),
    });
  } catch (e) { $('#rig-err').textContent = e.message; return; }
  $('#dlg-rig').close();
  await loadRigs();
  // Registered in order to be used: select it, which is still only a choice --
  // it starts nothing until a tab asks the rig for something.
  await selectRig(rig.name);
};

// ── The rig's agent, as Signals and Collect both read it ───────────────────
// Agents are written by each rig and do not all speak alike. The UR3e serves
// /arm with {joints: {arm: {name: deg}}, readings: [...]}; the dual SO-101
// serves /arms with the monitor's own shape, {joints: {left: {state: {...}}}},
// reports cameras it could not open as `missing` rather than `problem`, and
// answers a failed arm start with an HTTP 500. The page learns both here, once,
// rather than either rig learning the other's.

let armBase = null;

// Which arm route this rig's agent serves, asked of its /status once per rig.
async function agentArmBase() {
  if (!armBase) {
    const status = await j('/api/agent/status');
    armBase = ('arms' in status && !('arm' in status))
      ? '/api/agent/arms' : '/api/agent/arm';
  }
  return armBase;
}

// Either agent's arm reply as {sides: {side: {joint: value}}, readings, age_s}.
function armView(body) {
  const snap = body && body.snapshot;
  let joints = (snap && snap.joints) || {};
  // A rig with one limb may report it flat, {name: value}.
  if (Object.values(joints).some(v => typeof v === 'number' || v === null)) {
    joints = {arm: joints};
  }
  const sides = {};
  for (const side in joints) {
    const entry = joints[side] || {};
    // The monitor's shape keeps the measured values under `state`.
    sides[side] = (entry.state && typeof entry.state === 'object') ? entry.state : entry;
  }
  let age = null;
  if (snap && snap.t_read) age = Math.max(0, Date.now() / 1000 - snap.t_read);
  else if (snap && typeof snap.joint_drift_s === 'number') age = snap.joint_drift_s;
  return {sides, readings: (snap && snap.readings) || [], age_s: age,
          has: !!snap, running: !!(body && body.running),
          problem: (body && body.problem) || ''};
}

// What an agent's camera reply says went wrong, in one line, or ''.
function cameraProblem(body) {
  if (!body) return '';
  if (body.problem) return body.problem;
  return (body.missing || [])
    .map(m => typeof m === 'string' ? m
      : `${m.name || 'camera'}: ${m.reason || m.problem || m.error || 'missing'}`)
    .join('; ');
}

window.addEventListener('load', () => {
  loadRigs().catch(e => { $('#rigpick').title = e.message; });
});
