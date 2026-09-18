const $ = (s) => document.querySelector(s);
let curDataset = null, curEpisode = null, lastIntegrity = null;
let datasets = [];

async function j(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
}

function humanSize(bytes) {
  if (!bytes) return '';
  const u = ['B', 'kB', 'MB', 'GB', 'TB']; let i = 0, n = bytes;
  while (n >= 1000 && i < u.length - 1) { n /= 1000; i++; }
  return `${n < 10 && i ? n.toFixed(1) : Math.round(n)} ${u[i]}`;
}

function datasetLine(d) {
  // Second line: enough to tell two similarly named datasets apart without
  // opening either one.
  const bits = [];
  bits.push(d.stillborn ? 'no episodes yet' : `${d.episodes} ep`);
  if (d.pending) bits.push(`${d.pending}✗ marked`);
  if (d.bytes) bits.push(humanSize(d.bytes));
  if (d.fps) bits.push(`${d.fps} fps`);
  if (d.streams && d.streams.length) bits.push(d.streams.join('+'));
  return bits.join(' · ');
}

async function loadDatasets() {
  datasets = await j('/api/datasets');
  const ul = $('#ds-list'); ul.innerHTML = '';
  for (const d of datasets) {
    const li = document.createElement('li');
    li.title = (d.tasks || []).join(' / ');
    li.innerHTML = `<span class="grow"><span class="dsname">${d.name}</span>
                    <span class="muted dsline">${datasetLine(d)}</span></span>`;
    li.onclick = () => selectDataset(d.name, li, d);
    if (d.name === curDataset) li.classList.add('sel');
    ul.appendChild(li);
  }
  if (!datasets.length) ul.innerHTML = '<li class="muted">no datasets found</li>';
  if (window.onDatasetsLoaded) window.onDatasetsLoaded(datasets);
}

async function selectDataset(name, li, d) {
  curDataset = name; curEpisode = null;
  document.querySelectorAll('#ds-list li').forEach(x => x.classList.remove('sel'));
  if (li) li.classList.add('sel');
  $('#ep-title').textContent = name;
  if (window.onDatasetSelected) window.onDatasetSelected(name);
  if (d && d.stillborn) {
    // Created, then quit before recording: there is nothing to list, but the
    // directory is real and the operator most likely wants to delete it.
    $('#ep-list').innerHTML = '';
    showPending(0); lastIntegrity = null; showDamaged([], null);
    $('#viewer').innerHTML = '<p class="muted">This dataset holds no saved '
      + 'episodes — a session that stopped before recording. Resume it from the '
      + 'Collect tab, or delete it.</p>';
    return;
  }
  await loadEpisodes();
  $('#viewer').innerHTML = '<p class="muted">Select a recording.</p>';
}

function resetSelection(name) {
  // After a dataset is renamed away or deleted, the recording pane must not go
  // on showing the episodes of something that is no longer there.
  curDataset = name || null; curEpisode = null;
  $('#ep-title').textContent = name || 'Recordings';
  $('#ep-list').innerHTML = '';
  showPending(0); lastIntegrity = null; showDamaged([], null);
  $('#viewer').innerHTML =
    '<p class="muted">Select a recording to play its sensor view.</p>';
  if (name) loadEpisodes();
}

// Two quite different faults look the same in the list -- an episode with no
// length ("?f") -- and they need opposite remedies. An unreadable metadata file
// still has its recording behind it; an episode the dataset counted and never
// wrote has nothing at all, and until it is dropped no rewrite of this dataset
// can run.
function showDamaged(files, integrity) {
  const bar = $('#damaged-bar');
  const bad = files && files.length;
  const gap = integrity && !integrity.ok;
  bar.hidden = !(bad || gap);
  $('#repair').hidden = !(integrity && integrity.repairable);
  if (bar.hidden) return;
  const parts = [];
  if (bad) parts.push(
    `${files.length} unreadable metadata file(s) — some episode lengths are `
    + `unknown ("?f"). The episodes themselves may still play. (${files.join(', ')})`);
  if (gap) parts.push(
    integrity.summary
    + (integrity.repairable
        ? '. Repair drops the empty slot(s) and renumbers the rest.'
        : '.'));
  $('#damaged-text').textContent = parts.join(' · ');
}

$('#repair').onclick = async () => {
  const integrity = lastIntegrity || {};
  const n = (integrity.phantom || []).length;
  const ok = await confirmDialog({
    title: `Repair ${curDataset}?`,
    body: `${curDataset} counts ${n} episode(s) that were never recorded `
      + `(${(integrity.phantom || []).join(', ')}). Repairing forgets them and `
      + `renumbers the recordings after them, so their numbers change — what is `
      + `now recording 5 may become recording 4. Nothing recorded is lost. `
      + `Until this is done, no episode of this dataset can be deleted or merged.`,
    confirmLabel: 'Repair',
  });
  if (!ok) return;
  await j(`/api/datasets/${curDataset}/repair`, {method: 'POST'});
  pollJobs();
};

function showPending(n) {
  $('#pending-bar').hidden = !n;
  if (n) $('#pending-text').textContent =
    `${n} marked for deletion — still on disk, so NOT yet excluded from training.`;
}

async function loadEpisodes() {
  const data = await j(`/api/datasets/${curDataset}/episodes`);
  const eps = data.episodes;
  showPending(data.pending);
  lastIntegrity = data.integrity;
  showDamaged(data.damaged, data.integrity);
  const ul = $('#ep-list'); ul.innerHTML = '';
  for (const e of eps) {
    const li = document.createElement('li');
    li.dataset.idx = e.index;
    const len = (e.length === null || e.length === undefined) ? '?' : e.length;
    li.innerHTML =
      `<input type="checkbox" class="pick" onclick="event.stopPropagation()">
       <span class="grow">episode ${e.index}</span>
       <span class="muted">${len}f</span>
       <span class="del" title="delete">🗑</span>`;
    li.onclick = () => selectEpisode(e.index, li);
    li.querySelector('.del').onclick = (ev) => {
      ev.stopPropagation(); doDelete([e.index]);
    };
    li.querySelector('.pick').onchange = updateSelCount;
    ul.appendChild(li);
  }
  $('#all').checked = false; updateSelCount();
}

function picked() {
  return [...document.querySelectorAll('#ep-list .pick')]
    .filter(c => c.checked).map(c => +c.closest('li').dataset.idx);
}
function updateSelCount() {
  $('#del-sel').disabled = picked().length === 0;
}

// Playing an episode does not need a rendered video: the recorded camera files
// are already the frames to show, so they are played where they lie and the
// joint columns are drawn beside them from the same numbers the live view
// displayed. That makes the first click as fast as the browser can start a
// video, instead of as slow as compositing one. The recorded mp4 is still
// rendered on demand for a dataset this cannot cover (depth is stored as images,
// not as a playable stream) and for downloading an episode.
const JOINTS = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'];
let sync = null;  // the running viewer, so a new selection can stop the old one

function selectEpisode(idx, li) {
  curEpisode = idx;
  document.querySelectorAll('#ep-list li').forEach(x => x.classList.remove('sel'));
  if (li) li.classList.add('sel');
  if (sync) { sync.stop(); sync = null; }
  $('#viewer').innerHTML = '<p class="muted">opening…</p>';
  openEpisode(curDataset, idx);
}

function renderedFallback(name, idx, why) {
  $('#viewer').innerHTML =
    `<video controls autoplay muted src="/api/datasets/${name}/episodes/${idx}.mp4"></video>
     <p class="muted">episode ${idx} — ${why} Showing the composited view, which is
     rendered on first play.</p>`;
}

async function openEpisode(name, idx) {
  let info;
  try {
    info = await j(`/api/datasets/${name}/episodes/${idx}/playback`);
  } catch (e) {
    $('#viewer').innerHTML = `<p class="muted">${e.message}</p>`;
    return;
  }
  if (name !== curDataset || idx !== curEpisode) return;  // a newer click won
  if (info.has_depth || !info.streams.length) {
    renderedFallback(name, idx, 'this dataset records a depth stream.');
    return;
  }
  const probe = document.createElement('video');
  if (!probe.canPlayType('video/mp4; codecs="av01.0.05M.08"')) {
    renderedFallback(name, idx, 'this browser cannot decode AV1.');
    return;
  }
  const tiles = info.streams.map((s, i) =>
    `<figure class="tile"><video id="v${i}" muted preload="metadata" src="${s.url}"></video>
     <figcaption>${s.label}</figcaption></figure>`).join('');
  const motion = motionPanel(info);
  $('#viewer').innerHTML = `
    <div class="tiles">${tiles}</div>
    <div class="bar viewbar">
      <button id="play">▶︎ play</button>
      <input id="scrub" type="range" min="0" max="1000" value="0" class="grow">
      <span class="muted" id="clock">0.00 s</span>
      <a id="dl" class="muted" href="/api/datasets/${name}/episodes/${idx}.mp4"
         download>download mp4</a>
    </div>
    <div class="tables"><table id="joints"></table><table id="ee"></table></div>
    <details id="motion" open>
      <summary>motion over the whole episode — velocity and acceleration</summary>
      ${motion.html}
    </details>
    <p class="muted">episode ${idx} — ${info.streams.length} streams playing from the
    recorded files, joint values beside them.</p>`;
  motion.draw();
  wireMotionMenu(motion);
  sync = startSync(info);
}

function jointRows(info, frame) {
  const cell = (v, dp) => `<td>${v === undefined ? '--' : v.toFixed(dp)}</td>`;
  const rate = (v, dp) =>
    `<td class="rate">${v === undefined ? '--' : v.toFixed(dp)}</td>`;
  const state = info.state[frame] || [], action = info.action[frame] || [];
  const vel = (info.joint_vel || [])[frame] || [];
  const acc = (info.joint_acc || [])[frame] || [];
  let html = '<tr><th></th><th colspan="4">left</th><th colspan="4">right</th></tr>' +
             '<tr><th></th><th>state</th><th>cmd</th><th>vel</th><th>acc</th>' +
             '<th>state</th><th>cmd</th><th>vel</th><th>acc</th></tr>';
  JOINTS.forEach((jn, k) => {
    html += `<tr><th>${jn}</th>` +
      [k, k + 6].map(c =>
        cell(state[c], 1) + cell(action[c], 1) + rate(vel[c], 1) + rate(acc[c], 0)
      ).join('') + '</tr>';
  });
  return html;
}

// The end effector has no recorded stream on any episode collected from the
// leader arms, so these are computed from the joints through the robot's own
// kinematic model; the server says so by sending ee_error instead.
function eeRows(info, frame) {
  if (!info.ee) {
    return `<tr><th>end effector</th><td class="muted">unavailable</td></tr>`;
  }
  const rows = [['|v|', 'v', 'm/s', 3], ['|a|', 'a', 'm/s²', 2],
                ['|ω|', 'w', '°/s', 1], ['|α|', 'alpha', '°/s²', 0]];
  let html = '<tr><th>end effector</th><th>left</th><th>right</th><th></th></tr>';
  for (const [label, key, unit, dp] of rows) {
    const cells = ['left', 'right'].map(side => {
      const v = info.ee[side][key][frame];
      return `<td>${v === undefined ? '--' : v.toFixed(dp)}</td>`;
    }).join('');
    html += `<tr><th>${label}</th>${cells}<td class="muted">${unit}</td></tr>`;
  }
  return html;
}

// ── Motion plots ──────────────────────────────────────────────────────────────
//
// One tile per signal, drawn ONCE when the episode opens. The playhead is a
// positioned element the frame loop slides, not a stroke: paint() already runs
// on requestAnimationFrame and carries the end-of-episode stop, and re-drawing
// sixteen canvases inside it is how that loop starts missing frames.
const PW = 480, PH = 64;

// A canvas is transparent. With nothing painted into it, it shows the page --
// black in a dark browser -- and a translucent line on black is unreadable,
// which is what these plots used to be. So each tile paints its own surface and
// every line is drawn at full opacity, in a palette chosen for the theme in
// force.
function palette() {
  const dark = window.matchMedia
    && matchMedia('(prefers-color-scheme: dark)').matches;
  return dark
    ? {bg: '#151a21', zero: '#4b5563', vel: '#60a5fa', acc: '#fbbf24'}
    : {bg: '#fbfcfe', zero: '#c3c9d4', vel: '#1d4ed8', acc: '#c2410c'};
}

// Signals are grouped into panels, one per joint plus the two end-effector
// pairs, and each panel holds that signal for BOTH arms so they can be compared
// directly. The comparison is only honest if the pair shares a scale: two plots
// side by side on their own scales look alike however differently the arms
// moved, so the panel takes the larger peak of the two and both sides are drawn
// against it.
function motionGroups(info) {
  const accUnit = (u) => u.replace('/s', '/s²');
  const groups = [];
  JOINTS.forEach((jn, k) => {
    groups.push({
      key: jn, title: jn, signed: true,
      vUnit: info.joint_units[k], aUnit: accUnit(info.joint_units[k]),
      vDp: 1, aDp: 0,
      sides: [k, k + 6].map((c, s) => ({
        side: s ? 'right' : 'left',
        vel: info.joint_vel.map(r => r[c]), acc: info.joint_acc.map(r => r[c]),
        vPeak: info.peaks.joint_vel[c], aPeak: info.peaks.joint_acc[c],
      })),
    });
  });
  if (info.ee) {
    const pair = (key, title, vk, ak, vUnit, aUnit, vDp, aDp) => ({
      key: key, title: title, signed: false,
      vUnit: vUnit, aUnit: aUnit, vDp: vDp, aDp: aDp,
      sides: ['left', 'right'].map(side => ({
        side: side, vel: info.ee[side][vk], acc: info.ee[side][ak],
        vPeak: info.peaks.ee[side][vk], aPeak: info.peaks.ee[side][ak],
      })),
    });
    groups.push(pair('ee_linear', 'end effector · linear',
                     'v', 'a', 'm/s', 'm/s²', 3, 2));
    groups.push(pair('ee_angular', 'end effector · angular',
                     'w', 'alpha', '°/s', '°/s²', 1, 0));
  }
  for (const g of groups) {
    g.vPeak = Math.max(...g.sides.map(s => s.vPeak));
    g.aPeak = Math.max(...g.sides.map(s => s.aPeak));
  }
  return groups;
}

// Which panels to show, remembered across episodes and across sessions: a
// reviewer watching one wrist through a session should not have to re-choose it
// at every recording.
const MOTION_PREFS_KEY = 'so101.motion.prefs';
let motionPrefs = {hidden: [], vel: true, acc: true};
try {
  Object.assign(motionPrefs,
                JSON.parse(localStorage.getItem(MOTION_PREFS_KEY) || '{}'));
} catch (e) { /* a browser refusing storage is not a reason to lose the view */ }
const motionShown = (key) => !motionPrefs.hidden.includes(key);
function saveMotionPrefs() {
  try { localStorage.setItem(MOTION_PREFS_KEY, JSON.stringify(motionPrefs)); }
  catch (e) { /* as above */ }
}

// Each pixel column is drawn as the min-to-max span of the samples that land in
// it. A jerk is one or two frames wide, and an episode has more frames than the
// tile has pixels, so sampling every nth value would drop exactly the spikes
// this view exists to show.
function paintSeries(ctx, series, peak, signed, colour) {
  const n = series.length;
  if (!n || !peak) return;
  const base = signed ? PH / 2 : PH - 1, span = signed ? PH / 2 - 1 : PH - 2;
  const y = (v) => base - (v / peak) * span;
  ctx.strokeStyle = colour;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < PW; x++) {
    const a = Math.floor(x * n / PW);
    const b = Math.min(Math.max(Math.floor((x + 1) * n / PW), a + 1), n);
    let lo = Infinity, hi = -Infinity;
    for (let i = a; i < b; i++) {
      const v = series[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (lo === Infinity) continue;
    ctx.moveTo(x + 0.5, y(hi));
    ctx.lineTo(x + 0.5, Math.max(y(lo), y(hi) + 0.6));
  }
  ctx.stroke();
}

function drawTile(canvas, group, side, p) {
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = p.bg;
  ctx.fillRect(0, 0, PW, PH);
  const base = group.signed ? PH / 2 : PH - 1;
  ctx.strokeStyle = p.zero;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, base + 0.5);
  ctx.lineTo(PW, base + 0.5);
  ctx.stroke();
  if (motionPrefs.acc) paintSeries(ctx, side.acc, group.aPeak, group.signed, p.acc);
  if (motionPrefs.vel) paintSeries(ctx, side.vel, group.vPeak, group.signed, p.vel);
}

function drawMotion(groups) {
  const p = palette();
  for (const g of groups) {
    for (const s of g.sides) {
      const canvas = document.querySelector('#' + s.canvasId);
      if (canvas) drawTile(canvas, g, s, p);
    }
  }
}

function motionPanel(info) {
  if (!info.joint_vel) return {groups: [], html: '', draw: () => {}};
  const groups = motionGroups(info);
  const p = palette();
  const bound = (g) => g.signed ? '±' : '≤';
  const chips = groups.map(g =>
    `<button class="chip${motionShown(g.key) ? ' on' : ''}"
             data-sig="${g.key}">${g.title}</button>`).join('') +
    '<span class="grow"></span>' +
    `<button class="chip${motionPrefs.vel ? ' on' : ''}" data-series="vel"
             style="color:${p.vel}">velocity</button>` +
    `<button class="chip${motionPrefs.acc ? ' on' : ''}" data-series="acc"
             style="color:${p.acc}">acceleration</button>`;
  let n = 0;
  const panels = groups.map(g => {
    const plots = g.sides.map(s => {
      s.canvasId = 'c' + (n++);
      return `<figure class="plot">
         <div class="track">
           <canvas id="${s.canvasId}" width="${PW}" height="${PH}"></canvas>
           <div class="head"></div>
         </div>
         <figcaption>${s.side} · v ${bound(g)}${s.vPeak.toFixed(g.vDp)}
         · a ${bound(g)}${s.aPeak.toFixed(g.aDp)}</figcaption>
       </figure>`;
    }).join('');
    return `<section class="panel" data-sig="${g.key}"${
      motionShown(g.key) ? '' : ' hidden'}>
       <h3>${g.title} <span>· v ${bound(g)}${g.vPeak.toFixed(g.vDp)} ${g.vUnit}
       · a ${bound(g)}${g.aPeak.toFixed(g.aDp)} ${g.aUnit}</span></h3>
       <div class="pair">${plots}</div>
     </section>`;
  }).join('');
  const note = info.ee ? '' :
    `<p class="muted">end-effector motion unavailable: ${
      info.ee_error || 'no kinematic model'}</p>`;
  return {
    groups: groups,
    draw: () => drawMotion(groups),
    html: `<div class="bar menu" id="sigmenu">${chips}</div>
           <div class="panels">${panels}</div>${note}`,
  };
}

// Hiding a panel costs nothing but an attribute. Turning a series off means a
// redraw, which happens HERE, on the click -- never in the frame loop, which
// carries the end-of-episode stop and must stay cheap.
function wireMotionMenu(motion) {
  const menu = document.querySelector('#sigmenu');
  if (!menu) return;
  menu.onclick = (ev) => {
    const chip = ev.target.closest('.chip');
    if (!chip) return;
    if (chip.dataset.sig) {
      const key = chip.dataset.sig, hide = motionShown(key);
      motionPrefs.hidden = motionPrefs.hidden.filter(k => k !== key);
      if (hide) motionPrefs.hidden.push(key);
      chip.classList.toggle('on', !hide);
      const panel = document.querySelector(`.panel[data-sig="${key}"]`);
      if (panel) panel.hidden = hide;
    } else {
      const key = chip.dataset.series;
      motionPrefs[key] = !motionPrefs[key];
      chip.classList.toggle('on', motionPrefs[key]);
      motion.draw();
    }
    saveMotionPrefs();
  };
  if (window.matchMedia) {
    matchMedia('(prefers-color-scheme: dark)').onchange = () => motion.draw();
  }
}

// One stream is the clock; the others are told where to be. Each is seeked to
// its own window inside its own file, because the streams are cut at their own
// frame boundaries and so do not share a zero.
function startSync(info) {
  const videos = info.streams.map((s, i) => document.querySelector('#v' + i));
  const span = Math.max(info.streams[0].to - info.streams[0].from, 1e-6);
  const master = videos[0], base = info.streams[0].from;
  const heads = [...document.querySelectorAll('.head')];
  const frames = info.state.length;
  let stopped = false, playing = false;
  const at = () => Math.min(Math.max(master.currentTime - base, 0), span);

  const seek = (t) => info.streams.forEach((s, i) => {
    const want = s.from + Math.min(t, s.to - s.from);
    if (Math.abs(videos[i].currentTime - want) > 0.04) videos[i].currentTime = want;
  });
  // Land exactly on each stream's last frame, ignoring the tolerance the drift
  // correction uses: that tolerance is wider than a frame period, so it would
  // decline to undo an overshoot of the very size we are here to undo.
  const settle = () => info.streams.forEach((s, i) => {
    videos[i].currentTime = s.to;
  });
  const paint = () => {
    if (stopped) return;
    const t = at();
    document.querySelector('#clock').textContent = t.toFixed(2) + ' s';
    document.querySelector('#scrub').value = Math.round((t / span) * 1000);
    const frame = Math.min(Math.round(t * info.fps), frames - 1);
    document.querySelector('#joints').innerHTML = jointRows(info, frame);
    document.querySelector('#ee').innerHTML = eeRows(info, frame);
    // The plots are already drawn; only the marker moves. See motionPanel.
    const pct = frames > 1 ? (frame / (frames - 1)) * 100 : 0;
    for (const head of heads) head.style.left = pct + '%';
    // Stop on this episode's last frame. This has to be driven from the frame
    // loop rather than from the video's own timeupdate event, which browsers
    // throttle to about four times a second: a quarter of a second is seven
    // frames at the dataset rate, so a check driven by it sails well past the
    // end and into the next recording before it fires. Overshoot is then undone
    // rather than merely stopped, so the frame left on screen is this
    // episode's last and not whatever the decoder had reached.
    if (playing && master.currentTime >= info.streams[0].to - 1e-3) {
      pause();
      settle();
    }
    // Drift correction: playback rates differ slightly between streams, so the
    // followers are nudged back whenever they fall more than a frame behind.
    // A follower that has reached its own end is PAUSED rather than nudged: the
    // target is clamped to the end while the video keeps playing past it, so
    // correcting it would drag it back every frame while it ran forward into
    // the next recording in between -- seen as the boundary flickering.
    if (playing) {
      info.streams.forEach((s, i) => {
        if (i === 0) return;
        if (videos[i].currentTime >= s.to) { videos[i].pause(); return; }
        const want = s.from + Math.min(t, s.to - s.from);
        if (Math.abs(videos[i].currentTime - want) > 0.04) videos[i].currentTime = want;
      });
    }
    requestAnimationFrame(paint);
  };
  const play = () => {
    playing = true;
    document.querySelector('#play').textContent = '❚❚ pause';
    videos.forEach(v => v.play().catch(() => {}));
  };
  const pause = () => {
    playing = false;
    document.querySelector('#play').textContent = '▶︎ play';
    videos.forEach(v => v.pause());
  };
  document.querySelector('#play').onclick = () => (playing ? pause() : play());
  document.querySelector('#scrub').oninput = (e) => seek((e.target.value / 1000) * span);
  // Start where the episode starts, not where its file does.
  let ready = 0;
  videos.forEach((v, i) => v.addEventListener('loadedmetadata', () => {
    v.currentTime = info.streams[i].from;
    if (++ready === videos.length) play();
  }, { once: true }));
  requestAnimationFrame(paint);
  return { stop: () => { stopped = true; videos.forEach(v => v.pause()); } };
}

// Marking is cheap, so the rows go immediately and the request follows. No
// confirm dialog: a mark is reversible with Restore until it is compacted.
async function doDelete(indices) {
  const dead = new Set(indices);
  for (const li of document.querySelectorAll('#ep-list li')) {
    if (dead.has(+li.dataset.idx)) li.remove();
  }
  if (dead.has(curEpisode)) {
    curEpisode = null;
    $('#viewer').innerHTML = '<p class="muted">Deleted. Select a recording.</p>';
  }
  updateSelCount();
  try {
    const r = await j(`/api/datasets/${curDataset}/delete`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({episodes: indices}),
    });
    showPending(r.pending);
    loadDatasets();
  } catch (e) {
    alert('delete failed: ' + e.message);
    await loadEpisodes();  // the optimistic removal was wrong; resync
  }
}

// Compaction is the slow half: it rewrites and renumbers the whole dataset, so
// it runs as a job in the dock and every visible index changes when it lands.
async function doCompact() {
  const ok = await confirmDialog({
    title: `Remove the marked episodes from ${curDataset}?`,
    body: 'This rewrites the dataset, renumbers the survivors and cannot be '
      + 'undone. It runs in the background — watch it in the corner of the page.',
    confirmLabel: 'Remove for good',
  });
  if (!ok) return;
  const bar = $('#pending-bar');
  bar.classList.add('busy');
  $('#pending-text').textContent = 'Removing… rewriting the dataset.';
  try {
    await j(`/api/datasets/${curDataset}/compact`, {method: 'POST'});
  } catch (e) {
    alert('compact failed: ' + e.message);
    bar.classList.remove('busy');
    await loadDatasets(); await loadEpisodes();
    return;
  }
  pollJobs();  // the dock reloads the pane when the rewrite lands
}

async function doRestore() {
  try {
    await j(`/api/datasets/${curDataset}/restore`, {method: 'POST'});
  } catch (e) { alert('restore failed: ' + e.message); }
  await loadDatasets(); await loadEpisodes();
}

$('#all').onchange = (e) => {
  document.querySelectorAll('#ep-list .pick').forEach(c => c.checked = e.target.checked);
  updateSelCount();
};
$('#del-sel').onclick = () => doDelete(picked());
$('#compact').onclick = doCompact;
$('#restore').onclick = doRestore;

// The first load waits for roots.js: without a collection directory there is
// nothing to list, and the page opens on the directory dialog instead.
