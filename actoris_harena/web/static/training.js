// The Training tab: send a collected dataset to a GPU machine, and watch it.
//
// Nothing here decides whether a run can work. Every refusal comes from
// /api/training/plan, which is the same rule the terminal launcher applies, so
// the page cannot be more permissive than the thing that actually submits.

let trainingConfig = null;

function trainingVisible() {
  return !document.querySelector('#pane-training').hidden;
}

async function loadTrainingConfig() {
  const cfg = await j('/api/training/config');
  trainingConfig = cfg;

  const ds = $('#t-dataset');
  const chosen = ds.value;
  ds.innerHTML = cfg.datasets
    .map(d => `<option value="${d.name}">${d.name} (${d.episodes} ep)</option>`)
    .join('');
  if (chosen) ds.value = chosen;

  const dest = $('#t-dest');
  const chosenDest = dest.value;
  dest.innerHTML = cfg.destinations
    .map(d => `<option value="${d.name}">${d.name} — ${d.kind}</option>`)
    .join('');
  if (chosenDest) dest.value = chosenDest;

  renderPolicies(cfg.policies || []);
  describeDataset();
  describeDest();
}

// Two groups, because there are two implementations of the same three models
// and the choice between them is the experiment. `act` is LeRobot's; `so101_act`
// is this repo's port of it -- the same module tree moved, so the checkpoints
// are interchangeable and only the code that runs differs. Nine unstructured
// checkboxes said none of that, and a port read as a ninth unrelated policy.
function renderPolicies(policies) {
  const box = $('#t-policies');
  const kept = new Set(
    [...box.querySelectorAll('input:checked')].map(i => i.value));
  box.innerHTML = '<legend>Policies</legend>';

  const groups = [
    ['LeRobot', policies.filter(p => !p.local),
     'the installed LeRobot’s own implementations'],
    ['This repo', policies.filter(p => p.local),
     'src/so101_policies/ — trained by the same lerobot-train, through '
     + '--policy.discover_packages_path'],
  ];
  for (const [title, group, note] of groups) {
    if (!group.length) continue;
    const heading = document.createElement('h3');
    heading.className = 'sub';
    heading.textContent = title;
    heading.title = note;
    box.appendChild(heading);
    for (const p of group) {
      const label = document.createElement('label');
      label.className = 'row';
      // A policy the installed LeRobot has never heard of is SHOWN and left
      // untickable: hiding it would make the gate look like a missing feature,
      // and the refusal names the two files to change.
      label.innerHTML = `<input type="checkbox" value="${p.name}"`
        + `${p.available ? '' : ' disabled'}`
        + `${kept.has(p.name) ? ' checked' : ''}> ${p.name}`
        + (p.ported_from
            ? ` <span class="muted">— port of ${p.ported_from}</span>` : '')
        + (p.available
            ? '' : ` <span class="stream-absent">— ${p.local
                ? 'not importable here' : 'not in this LeRobot'}</span>`);
      if (!p.available) label.title = p.problem;
      box.appendChild(label);
    }
  }
}

function describeDataset() {
  const d = (trainingConfig.datasets || []).find(x => x.name === $('#t-dataset').value);
  $('#t-dataset-detail').textContent = d
    ? `${d.episodes} episodes, ${d.frames} frames — ${d.cameras.join(', ')}`
    : '';
  const composites = Object.entries(trainingConfig.composites || {});
  $('#t-cameras-help').textContent =
    "'all' is the dataset's own cameras. "
    + composites.map(([n, parts]) => `${n} tiles ${parts.join(' + ')} into one`).join('; ');
}

function describeDest() {
  const d = (trainingConfig.destinations || []).find(x => x.name === $('#t-dest').value);
  if (!d) { $('#t-dest-detail').textContent = ''; return; }
  const limits = Object.entries(d.limits || {})
    .map(([p, caps]) => `${p} ${Object.entries(caps).map(([k, v]) => `${k}≤${v}`).join(' ')}`);
  const live = machine(d.name);
  const measured = live && live.measured;
  $('#t-dest-detail').textContent =
    `${d.kind === 'local' ? 'this machine' : d.ssh} — stages to ${d.stage}`
    + (limits.length ? ` — measured ceilings: ${limits.join(', ')}` : '')
    + (live && !live.ok ? ` — ${live.problem}` : '')
    + (measured && measured.ok ? ` — ${measured.gpu || 'no GPU'}, ${measured.why}` : '');
  // Only a machine that was ASKED can offer this, and only when the answer is
  // that it would not use its GPU. A run that falls back to the CPU does not
  // fail, it just never finishes, so it has to be asked for.
  $('#t-allow-cpu-row').hidden = !(measured && measured.device === 'cpu');
}

// Reaching a machine costs an SSH, so this is asked for in the background
// once the tab is open and the detail line is filled in when it lands -- the
// form stays usable off the VPN, which is the whole reason Check touches
// nothing.
let trainingMachines = [];

async function loadMachines() {
  trainingMachines = await j('/api/training/machines');
  describeDest();
}

function machine(name) {
  return trainingMachines.find(m => m.name === name);
}

function trainingRequest() {
  return {
    dataset: $('#t-dataset').value,
    dest: $('#t-dest').value,
    policies: [...document.querySelectorAll('#t-policies input:checked')].map(c => c.value),
    cameras: $('#t-cameras').value.trim() || 'all',
    slots: $('#t-slots').value.trim() || '-',
    steps: $('#t-steps').value.trim() || '-',
    batch: $('#t-batch').value.trim() || '-',
    hours: $('#t-hours').value.trim() || null,
    restage: $('#t-restage').checked,
    allow_cpu: $('#t-allow-cpu').checked,
  };
}

function renderPlan(plan) {
  const table = $('#t-rows');
  table.innerHTML = '';
  for (const row of plan.rows) {
    const tr = document.createElement('tr');
    tr.className = 'lvl-OK';
    tr.innerHTML = `<td>${row.policy}</td><td>${row.cameras}</td>`
      + `<td>${row.resolved_steps} steps</td><td>batch ${row.resolved_batch}</td>`
      + `<td>${row.hours} h</td>`;
    table.appendChild(tr);
  }
  $('#t-err').textContent = (plan.refusals || []).join('\n');
  return (plan.refusals || []).length === 0;
}

async function checkTraining() {
  $('#t-err').textContent = 'checking…';
  const plan = await j('/api/training/plan', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(trainingRequest()),
  });
  return renderPlan(plan);
}

async function startTraining() {
  const request = trainingRequest();
  const ok = await checkTraining();
  if (!ok) return;
  const yes = await confirmDialog({
    title: `Train on ${request.dest}?`,
    body: `${request.policies.join(', ')} on ${request.dataset}. The dataset is `
      + `copied to ${request.dest} first, which can take several minutes, and `
      + `the run then holds that machine's GPU for hours.`,
    confirmLabel: 'Start training',
  });
  if (!yes) return;
  await j('/api/training/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(request),
  });
  // Staging outlives this request, so the dock is where it is watched.
  if (window.pollJobs) pollJobs();
  $('#t-state').textContent = `staging ${request.dataset} to ${request.dest}…`;
  // The run appears in the panel as soon as the machine has a log to show, so
  // go and look at it: staging is minutes, and a form that sits there having
  // apparently done nothing is indistinguishable from a launch that hung.
  if (window.showSubview) showSubview('training', 'jobs');
  if (window.loadRunList) loadRunList(true).catch(() => {});
}

async function testReach() {
  // Re-asks every machine and rebuilds the detail line from the answer, rather
  // than replacing it with "answered" — what a machine measured about itself
  // is the more useful half of the reply, and it is already in there.
  $('#t-err').textContent = 'connecting…';
  trainingMachines = await j('/api/training/machines?refresh=1');
  const live = machine($('#t-dest').value);
  $('#t-err').textContent = !live || live.ok ? '' : live.problem;
  describeDest();
}

$('#t-check').onclick = () =>
  checkTraining().catch(e => { $('#t-err').textContent = e.message; });
$('#t-reach').onclick = () =>
  testReach().catch(e => { $('#t-err').textContent = e.message; });
$('#t-start').onclick = () =>
  startTraining().catch(e => { $('#t-err').textContent = e.message; });
$('#t-dataset').onchange = describeDataset;
$('#t-dest').onchange = describeDest;

const _trainingPaneShown = window.onPaneShown;
window.onPaneShown = (name) => {
  if (_trainingPaneShown) _trainingPaneShown(name);
  if (name === 'training') {
    loadTrainingConfig().catch(e => { $('#t-err').textContent = e.message; });
    loadMachines().catch(() => {});
  }
};

window.addEventListener('load', () => {
  if (trainingVisible()) window.onPaneShown('training');
});
