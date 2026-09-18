// App shell: the tab bar and the header's collection-directory readout. Each
// pane keeps its own script; this file only decides which one is on screen.
const panes = () => document.querySelectorAll('.pane');

// A pane may be split into subviews, and one is: Training does two unrelated
// jobs -- starting a run and reading one -- which shared a border and nothing
// else. The mechanism is the tab bar's, one level down, so the hash addresses
// a subview too (#training/jobs) and a link to one survives a reload.
const subviews = (pane) =>
  document.querySelectorAll('#pane-' + pane + ' .subview');

function showSubview(pane, view) {
  const found = [...subviews(pane)];
  if (!found.length) return;
  const chosen = found.some(v => v.dataset.view === view) ? view
                                                          : found[0].dataset.view;
  found.forEach(v => { v.hidden = (v.dataset.view !== chosen); });
  document.querySelectorAll('#pane-' + pane + ' .subtab').forEach(t => {
    t.classList.toggle('sel', t.dataset.view === chosen);
  });
  location.hash = pane + '/' + chosen;
  if (window.onSubviewShown) window.onSubviewShown(pane, chosen);
}

function currentSubview(pane) {
  const shown = [...subviews(pane)].find(v => !v.hidden);
  return shown ? shown.dataset.view : null;
}

function showPane(name, view) {
  panes().forEach(p => { p.hidden = (p.id !== 'pane-' + name); });
  document.querySelectorAll('.tab').forEach(t => {
    t.classList.toggle('sel', t.dataset.pane === name);
  });
  location.hash = name;
  if (window.onPaneShown) window.onPaneShown(name);
  // After onPaneShown, so a pane's own loader has run before its subview is
  // told it is on screen. showSubview rewrites the hash to include the view.
  if (subviews(name).length) showSubview(name, view || currentSubview(name));
}

document.querySelectorAll('.tab').forEach(t => {
  t.onclick = () => showPane(t.dataset.pane);
});
document.querySelectorAll('.subtab').forEach(t => {
  t.onclick = () => showSubview(t.closest('.pane').id.slice(5), t.dataset.view);
});

// '#sensors' is what the tab used to be called; keep an old link working.
const [_hash, _view] = (location.hash || '#datasets').slice(1).split('/');
showPane(_hash === 'sensors' ? 'signals' : _hash, _view);

// One confirmation for the deletes that cannot be undone. Native confirm() is
// blocking and a browser may suppress it after a few; this one is neither, and
// it says the same things every time: what goes, how much of it, and that it
// does not come back. Marking an episode is NOT confirmed -- Restore undoes it.
function confirmDialog({title, body, confirmLabel = 'Delete'}) {
  const d = document.querySelector('#dlg-confirm');
  document.querySelector('#confirm-title').textContent = title;
  document.querySelector('#confirm-body').textContent = body;
  document.querySelector('#confirm-yes').textContent = confirmLabel;
  return new Promise(resolve => {
    d.returnValue = 'no';
    d.onclose = () => resolve(d.returnValue === 'yes');
    d.showModal();
  });
}
