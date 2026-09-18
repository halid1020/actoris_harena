// Training Jobs: what every machine is training, and how it is going.
//
// The Runs column used to list what THIS console had launched and offer one
// button that returned one line. Almost nothing was in that list: nineteen of
// the twenty-one run directories on these machines were started from a
// terminal. So the list is discovered, and this view reads it the way an
// experiment is actually read -- runs grouped into a project the operator
// named, in a table that sorts, with the configuration difference between the
// selected runs and a panel per metric.
//
// Two costs shape everything here, and they are the reason this is not simply
// "fetch everything and render". Asking a machine anything is a whole SSH, and
// a filtered log is a couple of hundred kilobytes. So the LIST is cheap (one
// call per machine, cached on the server) and carries only what a directory
// listing knows; a run's CURVE is fetched only for what is being watched -- the
// rows opened, the rows ticked, and whatever was written to in the last few
// minutes, which is how a live run is recognised without reading it. Nothing
// polls while the tab is off screen.

const RUN_LIST_MS = 45000;
const RUN_WATCH_MS = 20000;
// A log written to this recently is being written to now. Discovery reports the
// age against the machine's OWN clock, so this holds across timezones.
const LIVE_S = 300;

// Which panel block a metric belongs in, and the order they read best in: what
// the run is optimising, then what that means on held-out data, then what it
// predicts, then the machine. Anything unrecognised still gets drawn, under
// its own namespace -- that is the whole point of the metric channel.
const GROUPS = [
  ['train', 'Training'],
  ['eval', 'Evaluation'],
  ['pred', 'Prediction'],
];

let runList = [];
let runProblems = {};
let projectStore = {projects: [], builtin: []};
let currentProject = '__all__';
let sortKey = 'mtime';
let sortDesc = true;
const runDetail = new Map(); // key -> the last /progress answer
const runOpen = new Set();
const runPicked = new Set();
let runTimer = null;
let runBusy = false;

const runKey = (r) => `${r.dest}|${r.run}|${r.policy}`;

// ── What is on screen ───────────────────────────────────────────────────────

const jobsVisible = () =>
  !document.querySelector('#pane-training').hidden
  && !document.querySelector('#pane-training .subview[data-view="jobs"]').hidden;

function projectRuns(name) {
  const project = projectStore.projects.find((p) => p.name === name);
  return project ? new Set(project.runs) : new Set();
}

function inProject(entry) {
  if (currentProject === '__all__') return true;
  if (currentProject === '__unassigned__') return projectsOf(entry).length === 0;
  return projectRuns(currentProject).has(runKey(entry));
}

function matchesFilter(entry) {
  const text = ($('#t-filter').value || '').trim().toLowerCase();
  if (!text) return true;
  return [entry.run, entry.policy, entry.dest, entry.dataset]
    .filter(Boolean).join(' ').toLowerCase().includes(text);
}

const visibleRuns = () => runList.filter((r) => inProject(r) && matchesFilter(r));

// ── State, and what is worth reading ────────────────────────────────────────

function stateOf(entry) {
  const detail = runDetail.get(runKey(entry));
  if (detail && detail.ok && detail.summary) return detail.summary.state;
  // Nothing has been read yet, so only the listing is known: a log written to
  // in the last few minutes is being written to now, and anything else is a
  // run that is not currently going. Which of "done", "failed" or "stalled" it
  // was needs the log, and the log is only fetched for what is being watched.
  return entry.age_s !== null && entry.age_s < LIVE_S ? 'running' : 'idle';
}

function isLive(entry) {
  return entry.age_s !== null && entry.age_s !== undefined && entry.age_s < LIVE_S;
}

function watched() {
  return runList.filter(
    (r) => runOpen.has(runKey(r)) || runPicked.has(runKey(r)) || isLive(r)
  );
}

const summaryOf = (entry) => {
  const detail = runDetail.get(runKey(entry));
  return (detail && detail.ok && detail.summary) || null;
};

// ── The axis and the series ─────────────────────────────────────────────────

const xMode = () => $('#t-xaxis').value;
const xUnit = () => (xMode() === 'step' ? '' : 'h');
const xLabel = () => ({step: 'step', hours: 'hours', elapsed: 'hours since start'})[xMode()];
const smoothing = () => Number($('#t-smooth').value || 0) / 100;

function xOf(point, first) {
  if (xMode() !== 'step' && point.time && first && first.time) {
    return (point.time - first.time) / 3600;
  }
  return point.step;
}

// A named curve out of the /progress answer. Every metric arrives the same
// shape -- lerobot's tracker fields, our own channel and the sim rollout
// results alike -- so nothing here knows which parser produced which.
function seriesFor(detail, metric) {
  const entries = (detail && detail.series && detail.series[metric]) || [];
  const first = entries[0];
  return entries
    .filter((p) => p.value !== null && p.value !== undefined)
    .map((p) => ({x: xOf(p, first), y: p.value}));
}

function metricsOfSelection(entries) {
  const names = new Set();
  for (const entry of entries) {
    const detail = runDetail.get(runKey(entry));
    for (const name of (detail && detail.metrics) || []) names.add(name);
  }
  return [...names];
}

const colourFor = (index) => {
  const palette = chartPalette().series;
  return palette[index % palette.length];
};

// The runs the charts and the diff are about: the ticked ones, or -- when
// nothing is ticked -- the ones that are open, so opening a row shows its
// curves without also having to tick it.
function selected() {
  const picked = visibleRuns().filter((r) => runPicked.has(runKey(r)));
  return picked.length ? picked : visibleRuns().filter((r) => runOpen.has(runKey(r)));
}

// ── Projects ────────────────────────────────────────────────────────────────

async function loadProjects() {
  try {
    projectStore = await j('/api/training/projects');
  } catch (e) {
    projectStore = {projects: [], builtin: []};
  }
  renderProjects();
}

function renderProjects() {
  const list = $('#t-project-list');
  const rows = [
    ...(projectStore.builtin || []).map((b) => ({name: b.name, label: b.label, builtin: true})),
    ...projectStore.projects.map((p) => ({name: p.name, label: p.name, count: p.runs.length})),
  ];
  list.innerHTML = rows.map((r) => `
    <li data-name="${r.name}" class="${r.name === currentProject ? 'sel' : ''}">
      <span class="grow">${r.label}</span>
      <span class="muted">${r.builtin ? '' : `${r.count} run(s)`}</span>
    </li>`).join('');
  for (const node of list.children) {
    node.onclick = () => {
      currentProject = node.dataset.name;
      renderProjects();
      renderRuns();
    };
  }
  const custom = !currentProject.startsWith('__');
  $('#t-project-actions').hidden = !custom;
  const project = projectStore.projects.find((p) => p.name === currentProject);
  $('#t-project-detail').textContent = custom && project
    ? `${project.runs.length} run(s)${project.created ? ` · created ${project.created.slice(0, 10)}` : ''}`
    : 'A project is a name you give a set of runs. Deleting one removes the '
      + 'label, never the runs.';
  renderMissing(project);
}

// Members no machine answered for. Reported rather than dropped: a machine off
// the VPN this morning has not deleted anything, and a list that silently
// shrank would be the one way this view could misreport what was run.
function renderMissing(project) {
  const node = $('#t-project-missing');
  if (!project) { node.textContent = ''; return; }
  const present = new Set(runList.map(runKey));
  const missing = project.runs.filter((k) => !present.has(k));
  node.textContent = missing.length
    ? `${missing.length} run(s) in this project were not found on any machine `
      + `just now: ${missing.join(', ')}`
    : '';
}

async function projectAction(body, method = 'PATCH', name = currentProject) {
  try {
    await j(`/api/training/projects/${encodeURIComponent(name)}`,
            {method, headers: {'content-type': 'application/json'},
             body: body === null ? undefined : JSON.stringify(body)});
    await loadProjects();
    renderRuns();
  } catch (e) { $('#t-runs-err').textContent = e.message; }
}

// ── The runs table ──────────────────────────────────────────────────────────

// Membership is computed here rather than served with the run: discovery is
// cached for forty-five seconds and a project changes the moment it is clicked,
// so a server-side tag would be visibly stale exactly when it is being used.
function projectsOf(entry) {
  const key = runKey(entry);
  return projectStore.projects.filter((p) => p.runs.includes(key)).map((p) => p.name);
}

const COLUMNS = [
  {key: 'projects', label: 'project', get: (e) => projectsOf(e).join(', ')},
  {key: 'run', label: 'run', get: (e) => e.run},
  {key: 'policy', label: 'policy', get: (e) => e.policy},
  {key: 'dataset', label: 'dataset', get: (e) => e.dataset || ''},
  {key: 'dest', label: 'machine', get: (e) => e.dest},
  {key: 'state', label: 'state', get: stateOf},
  {key: 'step', label: 'step', get: (e) => (summaryOf(e) || {}).step},
  {key: 'loss', label: 'loss', get: (e) => (summaryOf(e) || {}).loss_last},
  {key: 'best', label: 'best', get: (e) => (summaryOf(e) || {}).loss_min},
  {key: 'eval', label: 'eval', get: (e) => (summaryOf(e) || {}).eval_loss_last},
  {key: 'mem', label: 'GB', get: (e) => (summaryOf(e) || {}).gpu_mem_gb_max},
  {key: 'mtime', label: 'written', get: (e) => e.mtime || 0},
];

function cell(column, entry) {
  const value = column.get(entry);
  if (column.key === 'state') {
    return `<span class="runstate ${value}">${value}</span>`;
  }
  if (column.key === 'mtime') return fmtSeconds(entry.age_s) + ' ago';
  if (column.key === 'policy') {
    // A port and the policy it was ported from are the same model run by
    // different code; the badge is what stops the two reading as one row
    // duplicated. `policy_name` is set for a sim cell, whose log is named for
    // the whole <mode>_<task>_<policy> triple.
    const name = entry.policy_name || entry.policy;
    const badge = name.startsWith('so101_')
      ? '<span class="chip">repo</span>' : '';
    return `${entry.policy} ${badge}`;
  }
  if (typeof value === 'number') {
    return column.key === 'step' ? fmtInt(value) : fmtValue(value);
  }
  return value === undefined || value === null ? '–' : String(value);
}

function sortRuns(entries) {
  const column = COLUMNS.find((c) => c.key === sortKey) || COLUMNS[COLUMNS.length - 1];
  const sorted = [...entries].sort((a, b) => {
    const x = column.get(a), y = column.get(b);
    // A run whose log has not been read has no step and no loss. Those sort to
    // the end whichever way the column is pointing, rather than pretending to
    // be zero and heading a "lowest loss" sort.
    const nx = x === undefined || x === null, ny = y === undefined || y === null;
    if (nx || ny) return nx && ny ? 0 : (nx ? 1 : -1);
    if (typeof x === 'number' && typeof y === 'number') return x - y;
    return String(x).localeCompare(String(y));
  });
  return sortDesc ? sorted.reverse() : sorted;
}

function renderRuns() {
  const entries = sortRuns(visibleRuns());
  const table = $('#t-runs-table');
  const head = COLUMNS.map((c) => {
    const arrow = c.key === sortKey ? (sortDesc ? ' ▾' : ' ▴') : '';
    return `<th data-sort="${c.key}">${c.label}${arrow}</th>`;
  }).join('');
  table.innerHTML = `<thead><tr><th class="tick"><input type="checkbox" id="t-pick-all"></th>`
    + head + `<th></th></tr></thead><tbody>`
    + entries.map((entry) => {
      const key = runKey(entry);
      return `<tr data-key="${key}" class="${runPicked.has(key) ? 'sel' : ''}">
        <td class="tick"><input type="checkbox" class="pick"
            ${runPicked.has(key) ? 'checked' : ''}></td>`
        + COLUMNS.map((c) => `<td>${cell(c, entry)}</td>`).join('')
        + `<td><button class="link toggle">${runOpen.has(key) ? 'hide' : 'open'}</button>`
        + `${entry.id ? '<button class="danger stop">Stop</button>' : ''}</td></tr>`;
    }).join('')
    + '</tbody>';
  if (!entries.length) {
    table.innerHTML += `<tbody><tr><td colspan="${COLUMNS.length + 2}" class="muted">`
      + (runList.length ? 'no runs match' : 'no runs found on any machine')
      + '</td></tr></tbody>';
  }
  wireTable();

  const live = runList.filter(isLive).length;
  $('#t-runstate').textContent = runList.length
    ? `${entries.length} of ${runList.length} run(s) on `
      + `${new Set(runList.map((r) => r.dest)).size} machine(s)`
      + (live ? `, ${live} being written to now` : '')
    : 'no runs found on any machine';
  $('#t-runs-err').textContent = Object.entries(runProblems)
    .map(([name, problem]) => `${name}: ${problem}`).join('\n');
  renderSelection();
  renderDetails();
}

function wireTable() {
  const table = $('#t-runs-table');
  table.querySelectorAll('th[data-sort]').forEach((th) => {
    th.onclick = () => {
      if (sortKey === th.dataset.sort) sortDesc = !sortDesc;
      else { sortKey = th.dataset.sort; sortDesc = true; }
      renderRuns();
    };
  });
  const all = table.querySelector('#t-pick-all');
  if (all) {
    all.onclick = async () => {
      for (const entry of visibleRuns()) {
        if (all.checked) runPicked.add(runKey(entry)); else runPicked.delete(runKey(entry));
      }
      renderRuns();
      if (all.checked) await refreshWatched();
    };
  }
  table.querySelectorAll('tr[data-key]').forEach((row) => {
    const key = row.dataset.key;
    const entry = runList.find((r) => runKey(r) === key);
    row.querySelector('.pick').onclick = async (event) => {
      event.stopPropagation();
      if (event.target.checked) runPicked.add(key); else runPicked.delete(key);
      renderRuns();
      if (event.target.checked) await refreshWatched();
    };
    const toggle = row.querySelector('.toggle');
    if (toggle) toggle.onclick = async () => {
      if (runOpen.has(key)) runOpen.delete(key); else runOpen.add(key);
      renderRuns();
      if (runOpen.has(key)) await refreshWatched();
    };
    const stop = row.querySelector('.stop');
    if (stop) stop.onclick = async () => {
      const yes = await confirmDialog({
        title: `Stop ${entry.run}?`,
        body: `This cancels the run on ${entry.dest}. Checkpoints already `
          + `written are kept, but training does not resume by itself.`,
        confirmLabel: 'Stop it',
      });
      if (!yes) return;
      try {
        await j(`/api/training/runs/${entry.id}/stop`, {method: 'POST'});
      } catch (e) { $('#t-runs-err').textContent = e.message; }
    };
  });
}

// ── The config diff, and the panel grid ─────────────────────────────────────

// Only the fields that DIFFER. A resolved train_config.json is 143 keys, and
// the whole question when two runs are side by side is which of them is the
// thing being varied -- everything identical is noise in front of it.
function renderDiff(entries) {
  const node = $('#t-diff');
  const configs = entries.map((e) => (runDetail.get(runKey(e)) || {}).config)
    .filter(Boolean);
  if (entries.length < 2 || configs.length < 2) { node.innerHTML = ''; return; }
  const keys = new Set();
  for (const config of configs) for (const key of Object.keys(config)) keys.add(key);
  const rows = [...keys].sort().filter((key) => {
    const values = configs.map((c) => JSON.stringify(c[key]));
    return values.some((v) => v !== values[0]);
  });
  if (!rows.length) {
    node.innerHTML = '<p class="muted">The selected runs were configured '
      + 'identically — the only difference is which machine ran them.</p>';
    return;
  }
  node.innerHTML = `<h3 class="sub">Configuration — ${rows.length} field(s) differ</h3>`
    + '<div class="scroll"><table class="runs"><thead><tr><th>field</th>'
    + entries.map((e, i) =>
        `<th><span class="swatch" style="background:${colourFor(i)}"></span>`
        + `${e.policy}</th>`).join('')
    + '</tr></thead><tbody>'
    + rows.map((key) => `<tr><td>${key}</td>`
        + configs.map((c) => `<td>${c[key] === undefined ? '–' : String(c[key])}</td>`).join('')
        + '</tr>').join('')
    + '</tbody></table></div>'
    + (configs.length < entries.length
        ? '<p class="muted">Runs with no checkpoint yet have no configuration '
          + 'written to disk, so they are not in this comparison.</p>' : '');
}

// ── What the world model predicts ───────────────────────────────────────────

// Only a world model has this section, so for ACT, diffusion and pi0.5 the
// panel is simply absent -- they predict actions and not observations, and an
// empty "prediction" heading over every run would read as a missing result
// rather than an inapplicable question.
//
// The column that matters is `beats holding`, NOT PSNR. Holding the last
// observed frame is a strong predictor of a scene that barely changes, and
// four of this rig's five cameras are gel images that do exactly that until
// contact. A tactile camera at 40 dB that loses to holding on every horizon
// step has taught the model nothing; the overhead camera at 30 dB that wins
// has. Sorting or colouring by PSNR would put those two in the wrong order,
// so the table is ordered by the verdict and PSNR rides behind it.
function renderPrediction(entries) {
  const node = $('#t-prediction');
  const scored = entries
    .map((entry) => [entry, (runDetail.get(runKey(entry)) || {}).predictions || []])
    .filter(([, list]) => list.length);
  if (!scored.length) { node.innerHTML = ''; return; }

  node.innerHTML = scored.map(([entry, list]) => {
    // The last checkpoint scored: parse_predictions sorts them forwards, and
    // an older score is kept in the file rather than shown, because two rows
    // for one camera with no step beside them would be unreadable.
    const latest = list[list.length - 1];
    const at = latest.step === null ? 'an unnumbered checkpoint' : `step ${latest.step}`;
    const cameras = Object.entries(latest.cameras)
      .sort((a, b) => (b[1].beats / b[1].horizon) - (a[1].beats / a[1].horizon));
    const rows = cameras.map(([camera, v]) => {
      const won = v.horizon ? v.beats / v.horizon : 0;
      const bar = `<span class="predbar"><span style="width:${(won * 100).toFixed(0)}%"></span></span>`;
      return `<tr><td>${camera}</td>`
        + `<td>${v.beats}/${v.horizon} ${bar}</td>`
        + `<td>${mean(v.psnr).toFixed(2)}</td><td>${mean(v.psnr_baseline).toFixed(2)}</td>`
        + `<td>${mean(v.ssim).toFixed(3)}</td><td>${mean(v.ssim_baseline).toFixed(3)}</td></tr>`;
    }).join('');
    const total = cameras.reduce((n, [, v]) => n + (v.beats ? 1 : 0), 0);
    return `<h3 class="sub">Future prediction — ${entry.policy} at ${at}</h3>`
      + '<div class="scroll"><table class="runs"><thead><tr>'
      + '<th>camera</th><th>beats holding</th><th>PSNR</th><th>held</th>'
      + '<th>SSIM</th><th>held</th></tr></thead><tbody>'
      + rows + '</tbody></table></div>'
      + `<p class="muted">${total} of ${cameras.length} cameras beat the held-last-frame `
      + `baseline on at least one horizon step, over ${latest.frames} frames of `
      + `episode(s) ${latest.episodes.join(', ') || '—'}. “held” is that baseline: `
      + 'the last observed frame repeated. A camera that does not beat it has not '
      + 'been learned, whatever its PSNR says.</p>';
  }).join('');
}

function mean(values) {
  if (!values || !values.length) return NaN;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function renderSelection() {
  const entries = selected();
  renderDiff(entries);
  renderPrediction(entries);
  const grid = $('#t-panels');
  if (!entries.length) {
    grid.innerHTML = '<p class="muted">Tick a run to draw its curves; tick '
      + 'two or more to overlay them.</p>';
    return;
  }
  const names = metricsOfSelection(entries);
  if (!names.length) {
    grid.innerHTML = '<p class="muted">reading…</p>';
    return;
  }
  const blocks = [];
  const seen = new Set();
  for (const [prefix, label] of GROUPS) {
    const group = names.filter((n) => n.startsWith(prefix + '/')).sort();
    group.forEach((n) => seen.add(n));
    if (group.length) blocks.push([label, group]);
  }
  const rest = names.filter((n) => !seen.has(n)).sort();
  if (rest.length) blocks.push(['Other', rest]);

  grid.innerHTML = blocks.map(([label, group]) =>
    `<h3 class="sub">${label}</h3><div class="panels">`
    + group.map((name) => `
      <figure class="plot" data-metric="${name}">
        <figcaption class="muted runnums">${name}</figcaption>
        <canvas></canvas>
        <figcaption class="muted runnums readout"></figcaption>
      </figure>`).join('')
    + '</div>').join('')
    + '<div class="bar wrap" id="t-legend"></div>';

  $('#t-legend').innerHTML = entries.map((e, i) =>
    `<span class="muted"><span class="swatch" style="background:${colourFor(i)}"></span>`
    + `${e.dest}/${e.run}/${e.policy}</span>`).join('');

  for (const figure of grid.querySelectorAll('figure[data-metric]')) {
    const metric = figure.dataset.metric;
    const canvas = figure.querySelector('canvas');
    const readout = figure.querySelector('.readout');
    const paint = () => drawCurves(canvas, entries.map((entry, i) => ({
      label: entry.policy,
      colour: colourFor(i),
      points: seriesFor(runDetail.get(runKey(entry)), metric),
    })), {
      // lr and the timing metrics are near-constant, and a log axis on a
      // constant draws a flat line with a misleading label; loss is the one
      // that spans three decades and needs one.
      logY: $('#t-logy').checked && /loss|grad_norm|place_err/.test(metric),
      smooth: smoothing(),
      xLabel: xLabel(), xUnit: xUnit(),
      empty: 'nothing logged yet',
    });
    paint();
    attachHover(canvas, readout, paint);
  }
}

// ── The per-run detail ──────────────────────────────────────────────────────

function renderDetails() {
  const box = $('#t-runs');
  const open = visibleRuns().filter((e) => runOpen.has(runKey(e)));
  box.innerHTML = open.map((entry) => {
    const detail = runDetail.get(runKey(entry)) || {};
    const s = detail.summary || {};
    const problems = (detail.problems || []).join('\n');
    const config = detail.config || {};
    const rows = Object.keys(config).sort()
      .map((k) => `<tr><td>${k}</td><td>${String(config[k])}</td></tr>`).join('');
    return `<div class="runcard open">
      <div class="bar wrap">
        <b>${entry.run}</b>
        <span class="chip on">${entry.policy}</span>
        <span class="runstate ${stateOf(entry)}">${stateOf(entry)}</span>
        <span class="grow muted">${entry.dest}${entry.episodes ? ` · ${entry.episodes} ep` : ''}</span>
      </div>
      <div class="runbar ${stateOf(entry)}">
        <span style="width:${((s.fraction || 0) * 100).toFixed(1)}%"></span></div>
      <div class="muted runnums">
        ${fmtInt(s.step)} / ${fmtInt(s.total_steps)}
        ${s.eta_s ? `· ${fmtSeconds(s.eta_s)} left` : ''}
        ${s.checkpoint ? `· checkpoint ${fmtInt(s.checkpoint)}` : ''}
        ${s.resumes ? `· resumed ${s.resumes}×` : ''}
        ${detail.selected ? `· selected step ${fmtInt(detail.selected.step)}` : ''}
      </div>
      ${detail.ok === false ? `<p class="err">${detail.problem}</p>` : ''}
      ${problems ? `<p class="err">${problems}</p>` : ''}
      <pre class="logtail">${(detail.tail || '').replace(/[<&]/g, (c) => c === '<' ? '&lt;' : '&amp;')}</pre>
      ${rows ? `<details><summary class="muted">resolved configuration
        (${Object.keys(config).length} fields)</summary>
        <div class="scroll"><table class="runs"><tbody>${rows}</tbody></table></div>
        </details>` : ''}
    </div>`;
  }).join('');
}

// ── CSV ─────────────────────────────────────────────────────────────────────

// What is plotted, not what was fetched: an export that quietly held more than
// the chart would disagree with the figure it was taken to support.
function exportCsv() {
  const entries = selected();
  const names = metricsOfSelection(entries);
  const lines = ['dest,run,policy,metric,x,value'];
  for (const entry of entries) {
    const detail = runDetail.get(runKey(entry));
    for (const name of names) {
      for (const point of seriesFor(detail, name)) {
        lines.push([entry.dest, entry.run, entry.policy, name, point.x, point.y].join(','));
      }
    }
  }
  const blob = new Blob([lines.join('\n')], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'training_runs.csv';
  a.click();
  URL.revokeObjectURL(url);
}

// ── Talking to the machines ─────────────────────────────────────────────────

async function loadRunList(fresh = false) {
  const body = await j(`/api/training/discovered${fresh ? '?refresh=1' : ''}`);
  runList = body.runs || [];
  runProblems = body.problems || {};
  renderProjects();
  renderRuns();
}

async function refreshWatched(fresh = false) {
  // Sequential on purpose: each of these is an SSH, and firing six at a machine
  // at once to draw six lines is not a trade worth making.
  for (const entry of watched()) {
    const key = runKey(entry);
    const query = new URLSearchParams({
      dest: entry.dest, run: entry.run, policy: entry.policy,
    });
    // Discovery already established which tree the run is in and, for a sim
    // cell, where its rollout results are. Passing them costs nothing and
    // saves the server a round trip to find out.
    if (entry.tree) query.set('tree', entry.tree);
    if (entry.cell) query.set('cell', entry.cell);
    if (fresh) query.set('refresh', '1');
    try {
      runDetail.set(key, await j(`/api/training/progress?${query}`));
    } catch (e) {
      runDetail.set(key, {ok: false, problem: e.message});
    }
    renderRuns();
  }
}

async function runsTick() {
  if (jobsVisible() && !runBusy) {
    runBusy = true;
    try {
      await loadRunList();
      await refreshWatched();
    } catch (e) {
      $('#t-runs-err').textContent = e.message;
    } finally { runBusy = false; }
  }
  clearTimeout(runTimer);
  runTimer = setTimeout(runsTick, watched().length ? RUN_WATCH_MS : RUN_LIST_MS);
}

$('#t-refresh').onclick = async () => {
  $('#t-runs-err').textContent = 'asking every machine…';
  try {
    await loadProjects();
    await loadRunList(true);
    await refreshWatched(true);
    $('#t-runs-err').textContent = '';
  } catch (e) { $('#t-runs-err').textContent = e.message; }
};
$('#t-logy').onchange = renderRuns;
$('#t-xaxis').onchange = renderRuns;
$('#t-smooth').oninput = renderSelection;
$('#t-filter').oninput = renderRuns;
$('#t-export').onclick = exportCsv;

$('#t-project-new').onclick = async () => {
  const name = prompt('Name this project');
  if (!name) return;
  try {
    await j('/api/training/projects', {
      method: 'POST', headers: {'content-type': 'application/json'},
      body: JSON.stringify({name}),
    });
    currentProject = name.trim();
    await loadProjects();
    renderRuns();
  } catch (e) { $('#t-runs-err').textContent = e.message; }
};
$('#t-project-add').onclick = () =>
  projectAction({add: [...runPicked]});
$('#t-project-remove').onclick = () =>
  projectAction({remove: [...runPicked]});
$('#t-project-rename').onclick = () => {
  const name = prompt('Rename the project', currentProject);
  if (!name || name === currentProject) return;
  const from = currentProject;
  currentProject = name.trim();
  projectAction({name}, 'PATCH', from);
};
$('#t-project-delete').onclick = async () => {
  const yes = await confirmDialog({
    title: `Delete the project ${currentProject}?`,
    body: 'This removes the label only. Every run stays exactly where it is on '
      + 'the machine that trained it, and stays in the list under All runs.',
    confirmLabel: 'Delete the label',
  });
  if (!yes) return;
  const name = currentProject;
  currentProject = '__all__';
  projectAction(null, 'DELETE', name);
};

window.addEventListener('resize', () => renderRuns());

const _jobsSubviewShown = window.onSubviewShown;
window.onSubviewShown = (pane, view) => {
  if (_jobsSubviewShown) _jobsSubviewShown(pane, view);
  if (pane === 'training' && view === 'jobs') {
    loadProjects().catch(() => {});
    runsTick();
  }
};

window.addEventListener('load', () => {
  if (jobsVisible()) { loadProjects().catch(() => {}); runsTick(); }
  else runTimer = setTimeout(runsTick, RUN_LIST_MS);
});
