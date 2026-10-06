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

window.addEventListener('load', () => {
  loadRigs().catch(e => { $('#rigpick').title = e.message; });
});
