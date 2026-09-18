// Which collection directory the console works on — one on this machine, or one
// on another reached over SSH. The page can open with none chosen, in which
// case this dialog is the first thing shown.

let rootInfo = null;
let browsePath = null;

function rootHeader() {
  const btn = $('#rootdir');
  const r = rootInfo || {};
  btn.dataset.path = r.root || '';
  if (!r.root) { btn.textContent = 'choose a collection directory…'; return; }
  const bits = [r.root];
  if (r.kind === 'ssh') bits.push('over SSH');
  if (!r.writable) bits.push('read-only');
  if (r.free_bytes) bits.push(`${humanSize(r.free_bytes)} free`);
  btn.textContent = bits.join(' · ');
}

async function loadRoots() {
  try { rootInfo = await j('/api/roots'); } catch (e) { rootInfo = null; }
  rootHeader();
  renderRecent();
  renderMounts();
  return rootInfo;
}

function renderRecent() {
  const bar = $('#root-recent');
  const recent = (rootInfo && rootInfo.recent) || [];
  bar.innerHTML = recent.length ? '<span class="muted">recent</span>' : '';
  for (const r of recent) {
    const b = document.createElement('button');
    b.textContent = r.target || r.path;
    b.title = r.path;
    b.onclick = () => useRoot(r.path);
    bar.appendChild(b);
  }
}

function renderMounts() {
  const box = $('#root-mounts');
  const mounts = (rootInfo && rootInfo.mounts) || [];
  box.innerHTML = '';
  if (!(rootInfo && rootInfo.sshfs)) {
    box.innerHTML = '<p class="err">sshfs is not installed here, and it is what '
      + 'mounts a remote directory: <code>sudo apt install sshfs</code>.</p>';
  }
  for (const m of mounts) {
    const row = document.createElement('div');
    row.className = 'bar wrap';
    row.innerHTML = `<span class="grow muted">${m.target} → ${m.path}</span>`;
    const b = document.createElement('button');
    b.textContent = 'Unmount';
    b.onclick = () => unmountRoot(m.path);
    row.appendChild(b);
    box.appendChild(row);
  }
}

async function browseTo(path) {
  let data;
  try { data = await j('/api/roots/browse?path=' + encodeURIComponent(path)); }
  catch (e) { $('#root-err').textContent = e.message; return; }
  browsePath = data.path;
  $('#root-path').value = data.path;
  $('#root-here').textContent = data.collection
    ? 'this directory holds datasets' : `${data.entries.length} sub-directory(s)`;
  $('#root-up').disabled = !data.parent;
  $('#root-up').dataset.path = data.parent || '';
  const ul = $('#root-list'); ul.innerHTML = '';
  for (const e of data.entries) {
    const li = document.createElement('li');
    li.textContent = e.name;
    if (e.collection) li.classList.add('collection');
    li.onclick = () => browseTo(e.path);
    li.ondblclick = () => useRoot(e.path);
    ul.appendChild(li);
  }
  if (!data.entries.length) ul.innerHTML = '<li class="muted">no sub-directories</li>';
}

async function afterRootChange(info) {
  rootInfo = info;
  rootHeader();
  renderRecent();
  renderMounts();
  dlg('dlg-root').close();
  resetSelection(null);
  if (window.onDatasetSelected) window.onDatasetSelected(null);
  await loadDatasets();
}

async function useRoot(path) {
  $('#root-err').textContent = '';
  try {
    const info = await j('/api/roots/use', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({path}),
    });
    await afterRootChange(info);
  } catch (e) { $('#root-err').textContent = e.message; }
}

async function unmountRoot(path) {
  $('#root-err').textContent = '';
  try {
    rootInfo = await j('/api/roots/unmount', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({path}),
    });
  } catch (e) { $('#root-err').textContent = e.message; return; }
  rootHeader(); renderRecent(); renderMounts();
  if (!rootInfo.root) { resetSelection(null); $('#ds-list').innerHTML = ''; }
}

$('#rootdir').onclick = async () => {
  $('#root-err').textContent = '';
  await loadRoots();
  openDialog('dlg-root');
  browseTo((rootInfo && rootInfo.root) || '/');
};

$('#root-up').onclick = () => {
  const up = $('#root-up').dataset.path;
  if (up) browseTo(up);
};

$('#root-path').onchange = () => browseTo($('#root-path').value.trim());
$('#root-use').onclick = () => useRoot($('#root-path').value.trim() || browsePath);

$('#root-mount').onclick = async () => {
  const target = $('#root-target').value.trim();
  const port = $('#root-port').value.trim();
  const identity = $('#root-key').value.trim();
  $('#root-err').textContent = 'mounting…';
  try {
    const info = await j('/api/roots/mount', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({target, port: port ? +port : null, identity}),
    });
    $('#root-err').textContent = '';
    await afterRootChange(info);
  } catch (e) { $('#root-err').textContent = e.message; }
};

// The page's first load: with a directory, list it; without one, ask for one.
loadRoots().then(info => {
  if (info && info.root) return loadDatasets();
  openDialog('dlg-root');
  return browseTo('/');
});
