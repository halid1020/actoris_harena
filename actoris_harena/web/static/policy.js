// The rollout page: what the policy was shown, what it planned, what the arms
// did with it, and whether it arrived in time. One poll drives all of it.

const $ = (s) => document.querySelector(s);
const JOINTS = ['pan', 'lift', 'elbow', 'wrist_flex', 'wrist_roll'];
const ROWS = JOINTS.concat(['gripper']);
const SIDES = ['left', 'right'];
// Orange is the plan and blue is the present, in the twin and in the plan
// panel alike, so the two pictures can be read as one.
const PLAN_INK = '#e0730f';
const NOW_INK = '#4080ff';
let status = {};

async function j(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
}

function fail(e) {
  $('#p-err').textContent = e.message;
  $('#p-err').hidden = false;
}

// -- the throttle -----------------------------------------------------
async function setMode(mode) {
  try {
    await j('/api/mode', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode}),
    });
    $('#p-err').hidden = true;
  } catch (e) {
    fail(e);
  }
  poll();
}

// Ending a trial. Both of these ask for LESS motion, not more — the run holds
// and the arms are RELEASED — but letting go of a rig mid-grasp drops whatever
// it was holding, so each is confirmed the way arming is.
//
// The two differ only in intent, and the dialogs say which: Stop ends the trial
// with no next one in mind, Reset ends it and begins the next.
async function endTrial() {
  if (!confirm(status.dry_run
      ? 'End this trial? The plan is dropped and the run holds.'
      : 'End this trial and release the arms? Torque comes off — anything the '
        + 'grippers are holding will be let go — and the run stays up.')) return;
  await setMode('stop');
}

async function resetRun() {
  const sim = status.resettable === 'sim';
  if (!confirm(sim
      ? 'Put the scene back and begin the next trial? The plan is dropped and '
        + 'nothing moves until you run again.'
      : 'Release the arms and begin the next trial? Torque comes off — anything '
        + 'the grippers are holding will be let go — and you put the scene back '
        + 'yourself.')) return;
  try {
    await j('/api/reset', {method: 'POST'});
    $('#p-err').hidden = true;
  } catch (e) {
    fail(e);
  }
  poll();
}

$('#p-arm').onclick = () => {
  // The same sentence the terminal used to print, in the one place that can
  // still stop it being true.
  if (!confirm('The follower arms will MOVE: they ramp to the policy’s '
               + 'first action, then follow it. Is the workspace clear?')) return;
  setMode('arm');
};
$('#p-hold').onclick = () => setMode('hold');
$('#p-preview').onclick = () => setMode('preview');
$('#p-step').onclick = () => setMode('step');
$('#p-run').onclick = () => setMode('run');
$('#p-stop').onclick = endTrial;
$('#p-reset').onclick = resetRun;

// -- how the attempt went ---------------------------------------------
// The one control on this page that touches nothing: no torque, no mode, no
// scene. So it is never confirmed and never disabled mid-run — the moment an
// operator knows how an attempt went is usually before it has finished playing
// out, and a dialog in front of that verdict would lose it.
let verdictOff = false;

async function judge(outcome) {
  try {
    const said = await j('/api/outcome', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({outcome, notes: $('#p-note').value}),
    });
    $('#p-note').value = '';
    $('#p-err').hidden = true;
    // Shown straight away rather than waiting for the next poll: the click and
    // the confirmation belong to the same moment.
    paintVerdict(said.trial, {[String(said.trial)]: said});
  } catch (e) {
    fail(e);
  }
}

function paintVerdict(trial, verdicts) {
  const n = Number.isFinite(trial) ? trial : 1;
  $('#p-trial').textContent = n;
  const said = (verdicts || {})[String(n)];
  const map = {success: '#p-win', failure: '#p-lose', discard: '#p-void'};
  Object.values(map).forEach((sel) => $(sel).classList.remove('sel'));
  if (said && map[said.outcome]) $(map[said.outcome]).classList.add('sel');
  $('#p-verdict-said').textContent = said
    ? `recorded: ${said.outcome}${said.notes ? ` — ${said.notes}` : ''}`
    : (verdictOff ? 'this run keeps no log (--no-log)' : '');
}

$('#p-win').onclick = () => judge('success');
$('#p-lose').onclick = () => judge('failure');
$('#p-void').onclick = () => judge('discard');

// -- the task ---------------------------------------------------------
// A run may be started with no task at all; it waits for this.
$('#p-task-form').onsubmit = async (ev) => {
  ev.preventDefault();
  try {
    await j('/api/task', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({task: $('#p-task').value}),
    });
    $('#p-err').hidden = true;
  } catch (e) {
    fail(e);
  }
  poll();
};

// -- the twin ---------------------------------------------------------
// The picture is NOT drawn here. The <iframe> holds a viser scene rendered by
// this browser's own GPU, which is what makes it something you can take hold of
// and look at from the side. All this page does is aim it: one request per
// frame naming the plan and the action, and viser moves the two ghosts.
//
// A request that is slow, refused, or answered 409 because the plan moved on
// changes nothing, so the scene simply holds the pose it already has.
const TWIN_PERIOD_MS = 100;
let twinSeq = -1, twinN = 0, twinAt = 0, twinBusy = false, twinPlaying = true;
let twinFramed = false;

function twinFrame(url) {
  // Set once: re-assigning src would reload the scene and throw away whatever
  // angle the operator had turned it to.
  if (twinFramed || !url) return;
  twinFramed = true;
  $('#p-twin').src = url;
}

function twinDraw(i) {
  if (twinBusy || twinSeq < 0) return;
  twinBusy = true;
  fetch(`/twin/at?seq=${twinSeq}&i=${i}`)
    .catch(() => {})
    .finally(() => { twinBusy = false; });
}

function twinBar() {
  $('#p-twin-play').textContent = twinPlaying ? '⏸ pause' : '▶ play';
  $('#p-twin-at').textContent = twinN
    ? `action ${twinAt + 1} of ${twinN}` : '—';
  const scrub = $('#p-twin-scrub');
  scrub.max = Math.max(0, twinN - 1);
  if (document.activeElement !== scrub) scrub.value = twinAt;
}

function twinSeek(i) {
  twinAt = twinN ? ((i % twinN) + twinN) % twinN : 0;
  twinBar();
  twinDraw(twinAt);
}

$('#p-twin-play').onclick = () => { twinPlaying = !twinPlaying; twinBar(); };
$('#p-twin-prev').onclick = () => { twinPlaying = false; twinSeek(twinAt - 1); };
$('#p-twin-next').onclick = () => { twinPlaying = false; twinSeek(twinAt + 1); };
$('#p-twin-scrub').oninput = () => {
  twinPlaying = false;
  twinSeek(parseInt($('#p-twin-scrub').value, 10) || 0);
};

// Paused still re-requests the same frame: the orange ghost holds the action
// being examined while the blue one keeps tracking the real arms.
setInterval(() => {
  if (twinSeq < 0 || !twinN) return;
  if (twinPlaying && !twinBusy) twinAt = (twinAt + 1) % twinN;
  twinBar();
  twinDraw(twinAt);
}, TWIN_PERIOD_MS);

// -- the pictures -----------------------------------------------------
// The live cameras are NOT connected on load. Each MJPEG stream is a
// connection that never closes and a browser allows about six per origin, so
// they are opened only when asked for.
let liveConnected = false;
function buildTiles(cameras) {
  if ($('#p-live').children.length) return;
  const tile = (caption) =>
    `<figure class="tile"><img alt="${caption}">
     <figcaption>${caption}</figcaption></figure>`;
  $('#p-live').innerHTML = cameras.map(tile).join('');
  $('#p-shown').innerHTML = cameras.map(tile).join('');
  // A frame the run has not sent yet answers 404; keep the previous one rather
  // than painting a broken-image icon over the panel.
  $('#p-shown').querySelectorAll('img').forEach((img) => {
    img.onerror = () => img.removeAttribute('src');
  });
}

$('#p-live-connect').onclick = () => {
  const btn = $('#p-live-connect');
  const imgs = $('#p-live').querySelectorAll('img');
  if (liveConnected) {
    imgs.forEach((img) => img.removeAttribute('src'));
    liveConnected = false;
    btn.textContent = 'connect';
    return;
  }
  imgs.forEach((img) => { img.src = `/stream/${img.alt}.mjpg`; });
  liveConnected = true;
  btn.textContent = 'disconnect';
};

// The frames the last request carried are still, so they are fetched rather
// than streamed -- once per chunk, which is when they change.
let shownSeq = -1;
function refreshShown(cameras, seq) {
  if (seq === shownSeq || seq < 0) return;
  shownSeq = seq;
  const imgs = $('#p-shown').querySelectorAll('img');
  cameras.forEach((c, i) => {
    if (imgs[i]) imgs[i].src = `/shown/${c}.jpg?seq=${seq}`;
  });
}

// -- the numbers ------------------------------------------------------
function jointsTable(state, commanded) {
  const rows = [];
  rows.push(`<tr><th></th><th>measured</th><th>commanded</th><th>error</th></tr>`);
  SIDES.forEach((side, s) => {
    JOINTS.forEach((name, k) => {
      const i = s * 6 + k;
      const m = state ? state[i] : null;
      const c = commanded ? commanded[i] : null;
      const e = (m === null || c === null) ? null : c - m;
      const far = e !== null && Math.abs(e) > 5 ? ' class="far"' : '';
      rows.push(`<tr><td>${side} ${name}</td><td>${fmt(m)}</td>
        <td>${fmt(c)}</td><td${far}>${fmt(e)}</td></tr>`);
    });
  });
  $('#p-joints').innerHTML = rows.join('');
}

// The grippers get their own panel because they are where a grasp is won or
// lost, and the whole range they were taught spans a few hundredths.
function grippers(state, commanded) {
  $('#p-grip').innerHTML = SIDES.map((side, s) => {
    const i = s * 6 + 5;
    const m = state ? state[i] : null;
    const c = commanded ? commanded[i] : null;
    const gap = (m === null || c === null) ? null : c - m;
    return `<div class="grip"><span class="muted">${side}</span>
      <b>${fmt(m, 3)}</b>
      <span class="cmd">commanded ${fmt(c, 3)}</span>
      <span class="gap">gap ${fmt(gap, 3)}</span></div>`;
  }).join('');
}

function fmt(v, digits = 2) {
  return (v === null || v === undefined) ? '—' : v.toFixed(digits);
}

// -- the splice -------------------------------------------------------
// What each strategy actually does, in a sentence, because the names and the
// numbers mean nothing on their own.
const HELP = {
  sync: 'Ask, wait, execute the WHOLE chunk, ask again. The arms hold still '
    + 'for every round trip: simple and reproducible, and the slowest.',
  receding: 'Ask, wait, execute only the first fraction of the chunk, throw '
    + 'the rest away and ask again — so the policy looks at the world again '
    + 'long before its plan runs out. The arms still hold for every round trip.',
  append: 'Queue each new chunk BEHIND whatever is still queued. Nothing is '
    + 'discarded, but every action runs later than the moment it was planned '
    + 'for. This is the behaviour the others exist to beat.',
  replace: 'Drop what is queued, drop the rows of the new chunk whose moment '
    + 'has already passed, execute the rest. Every action runs at the time it '
    + 'was planned for; the join can be a visible step.',
  blend: 'Like replace, but the first few actions are mixed out of the old '
    + 'plan into the new one, so the arms do not jump at the join.',
  ensemble: 'Where the old and the new plan overlap, execute a weighted '
    + 'average of the two rather than picking one.',
  rtc: 'Like replace, except the GPU host steers the policy towards the '
    + 'actions already queued while it generates them, so the plan arrives '
    + 'smooth instead of being smoothed here. Flow-matching policies only.',
};

// Changing the splice mid-run drops the queue: what is in there was spliced
// under the old rule. The server does that; here we only send the change.
async function setSplice(body) {
  try {
    paintSplice(await j('/api/strategy', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    }));
    $('#p-err').hidden = true;
  } catch (e) {
    fail(e);
  }
  poll();
}

$('#p-strategy').onchange = () => setSplice({strategy: $('#p-strategy').value});
$('#p-ratio').oninput = () => { $('#p-ratio-out').value = $('#p-ratio').value; };
$('#p-ratio').onchange =
  () => setSplice({execute_ratio: parseFloat($('#p-ratio').value)});
$('#p-window').onchange =
  () => setSplice({blend_window: parseInt($('#p-window').value, 10)});
$('#p-weight').oninput = () => { $('#p-weight-out').value = $('#p-weight').value; };
$('#p-weight').onchange =
  () => setSplice({new_weight: parseFloat($('#p-weight').value)});

// Only the number the current strategy actually uses is on screen, so the row
// never offers a knob that does nothing.
function paintSplice(sp, hz) {
  if (!sp) return;
  if ($('#p-strategy').value !== sp.strategy) $('#p-strategy').value = sp.strategy;
  const show = (wrap, on) => { $(wrap).hidden = !on; };
  show('#p-ratio-wrap', sp.strategy === 'receding');
  show('#p-window-wrap', sp.strategy === 'blend');
  show('#p-weight-wrap', sp.strategy === 'ensemble');
  if (document.activeElement !== $('#p-ratio')) {
    $('#p-ratio').value = sp.execute_ratio;
    $('#p-ratio-out').value = sp.execute_ratio;
  }
  if (document.activeElement !== $('#p-window')) $('#p-window').value = sp.blend_window;
  if (document.activeElement !== $('#p-weight')) {
    $('#p-weight').value = sp.new_weight;
    $('#p-weight-out').value = sp.new_weight;
  }
  // Ticks mean nothing without the rate they are counted at.
  $('#p-window-secs').textContent = hz
    ? `ticks (${(sp.blend_window / hz).toFixed(2)} s)` : 'ticks';
  const blocking = sp.blocking
    ? ' The arms hold still while each request is out.'
    : ' The next plan is fetched while this one is still running.';
  $('#p-splice-help').textContent = (HELP[sp.strategy] || '') + blocking;
}

function buildStrategies(names) {
  const sel = $('#p-strategy');
  if (sel.options.length || !names || !names.length) return;
  sel.innerHTML = names.map((n) => `<option value="${n}">${n}</option>`).join('');
}

// -- the vitals -------------------------------------------------------
function vital(id, value, note, level) {
  const el = $(`#${id}`);
  el.querySelector('b').textContent = value;
  if (note !== undefined) el.querySelector('span').textContent = note;
  el.className = `vital${level ? ' ' + level : ''}`;
}

function vitals(s) {
  const queue = s.queue === undefined ? null : s.queue;
  const need = s.threshold || 0;
  const remote = s.chunked !== false;
  // An empty queue means the arms are holding for want of a plan; a queue at
  // or below the threshold with nothing in flight is about to be.
  const level = queue === 0 ? 'bad' : (queue !== null && queue <= need ? 'warn' : '');
  vital('v-queue', queue === null ? '—' : String(queue),
        remote ? `queue · asks at ${need}` : 'queue · local', level);
  const rtt = (s.round_trip_s || 0) * 1000;
  vital('v-rtt', remote && rtt ? `${rtt.toFixed(0)} ms` : '—', 'round trip');
  const infer = (s.server_infer_s || 0) * 1000;
  vital('v-infer', remote && infer ? `${infer.toFixed(0)} ms` : '—',
        'inference, of that');
  const holds = s.holds || 0;
  const ticks = s.tick || 0;
  vital('v-holds', `${holds}`,
        ticks ? `held · ${((holds / ticks) * 100).toFixed(0)}% of ticks` : 'held',
        holds && ticks && holds / ticks > 0.4 ? 'warn' : '');
  vital('v-chunk',
        s.pending ? `${s.pending.n}` : (s.chunk ? `${s.chunk.n}` : '—'),
        s.pending ? 'queued to execute' : 'plan, actions');
  vital('v-splice', (s.splice && s.splice.strategy) || '—', 'splice');
  vital('v-progress',
        s.ticks_total ? `${ticks}/${s.ticks_total}` : `${(s.t || 0).toFixed(0)} s`,
        s.ticks_total ? `${(s.t || 0).toFixed(0)} s at ${s.hz || 0} Hz`
                      : `running until stopped, ${s.hz || 0} Hz`);
}

function timing(s) {
  const rtt = (s.round_trip_s || 0) * 1000;
  const need = (s.threshold || 0);
  const queue = s.queue === undefined ? null : s.queue;
  const starving = queue !== null && queue === 0;
  const row = (k, v, note = '') =>
    `<tr><td>${k}</td><td>${v}</td><td class="muted">${note}</td></tr>`;
  $('#p-timing').innerHTML = [
    row('round trip', `${rtt.toFixed(0)} ms`,
        `inference ${((s.server_infer_s || 0) * 1000).toFixed(0)} ms of it`),
    row('queue', queue === null ? '—' : `${queue} action(s)`,
        starving ? 'EMPTY — the arms are waiting for a plan'
                 : `next chunk requested at ${need}`),
    row('chunk', s.actions_per_chunk ? `${s.actions_per_chunk} actions`
                                     : '—',
        s.chunk ? `plan #${s.chunk.seq}` : 'none yet'),
    row('holds', `${s.holds || 0} tick(s)`, s.last_error || ''),
    row('progress', `${s.tick || 0} tick(s)`,
        `${(s.t || 0).toFixed(1)} s at ${s.hz || 0} Hz`
        + (s.ticks_total ? ` of ${s.ticks_total}` : ', no limit')),
  ].join('');
}

// -- the plan ---------------------------------------------------------
// Twelve small multiples rather than ten anonymous lines on one axis: each
// joint gets its own scale, because a 90-degree wrist_roll sweep otherwise
// flattens every other channel to a straight line, and the grippers -- where a
// pick-and-place is won or lost -- were not drawn at all.
function fitCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(cv.clientWidth * dpr));
  const h = Math.max(1, Math.round(cv.clientHeight * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cv.clientWidth, cv.clientHeight);
  return [ctx, cv.clientWidth, cv.clientHeight];
}

function drawChunk(chunk, state, hz, at) {
  const cv = $('#p-chunk');
  const [ctx, W, H] = fitCanvas(cv);
  if (!chunk || !chunk.actions.length) {
    $('#p-chunk-note').textContent = 'no plan yet';
    return;
  }
  const a = chunk.actions, n = a.length;
  const ink = getComputedStyle(document.body).color;
  const cw = W / 2, chh = H / 6, LABEL = 92, RIGHT = 66;
  ctx.font = '11px system-ui, sans-serif';
  ctx.textBaseline = 'middle';

  SIDES.forEach((side, s) => ROWS.forEach((name, r) => {
    const ch = s * 6 + r;
    const grip = r === 5;
    const x0 = s * cw, y0 = r * chh;
    const px0 = x0 + LABEL, px1 = x0 + cw - RIGHT;
    const py0 = y0 + 5, py1 = y0 + chh - 5;
    let lo = Infinity, hi = -Infinity;
    for (const row of a) { lo = Math.min(lo, row[ch]); hi = Math.max(hi, row[ch]); }
    const m = state && state.length > ch ? state[ch] : null;
    if (m !== null) { lo = Math.min(lo, m); hi = Math.max(hi, m); }
    const span = (hi - lo) > 1e-6 ? (hi - lo) : 1;
    const X = (i) => px0 + (n < 2 ? 0 : (i / (n - 1)) * (px1 - px0));
    const Y = (v) => py1 - ((v - lo) / span) * (py1 - py0);

    ctx.strokeStyle = 'rgba(128,128,128,.28)';
    ctx.lineWidth = 1;
    ctx.strokeRect(px0, py0, px1 - px0, py1 - py0);

    // Where the joint is RIGHT NOW, so the plan is read as a departure from it.
    if (m !== null) {
      ctx.strokeStyle = NOW_INK;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(px0, Y(m));
      ctx.lineTo(px1, Y(m));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.strokeStyle = PLAN_INK;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    a.forEach((row, i) => (i ? ctx.lineTo(X(i), Y(row[ch]))
                             : ctx.moveTo(X(i), Y(row[ch]))));
    ctx.stroke();

    // The action the twin is showing, so the two panels are one picture.
    if (at !== null && at < n) {
      ctx.strokeStyle = ink;
      ctx.globalAlpha = 0.45;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(X(at), py0);
      ctx.lineTo(X(at), py1);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = PLAN_INK;
      ctx.beginPath();
      ctx.arc(X(at), Y(a[at][ch]), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = ink;
    ctx.globalAlpha = 0.75;
    ctx.textAlign = 'left';
    ctx.fillText(`${side} ${name}`, x0 + 4, (py0 + py1) / 2);
    ctx.textAlign = 'right';
    const unit = grip ? '' : '°';
    ctx.globalAlpha = 0.5;
    ctx.fillText(`${lo.toFixed(grip ? 2 : 0)}…${hi.toFixed(grip ? 2 : 0)}${unit}`,
                 x0 + cw - 4, (py0 + py1) / 2);
    ctx.globalAlpha = 1;
  }));

  const secs = hz ? ` , ${(n / hz).toFixed(2)} s at ${hz} Hz` : '';
  $('#p-chunk-note').textContent =
    `plan #${chunk.seq}: ${n} actions${secs} · orange is the plan, `
    + 'blue is where the joint is now, the upright line is the twin';
}

// -- the loop ---------------------------------------------------------
// The one sentence an operator beside the rig needs, in both of its states.
const TORQUE_ON = 'The arms are under torque. <b>Preview</b> shows the queued '
  + 'plan in the twin without moving anything; <b>Execute one chunk</b> runs '
  + 'exactly that plan and stops; <b>Hold</b> freezes and drops the queue; '
  + '<b>Stop</b> ends the trial and releases the arms.';
const TORQUE_OFF = 'The arms are <b>released</b> and safe to move by hand. '
  + 'Enable them again to begin the next trial; Ctrl+C at the terminal ends '
  + 'the run.';

function render(s) {
  const mode = s.mode || '…';
  $('#p-mode').textContent = mode;
  $('#p-mode').className = `mode ${mode}`;
  // Rendered from the run every poll, not written once and left: between trials
  // the arms are free, and somebody is about to put their hands on them.
  $('#p-torque').hidden = !!s.dry_run;
  $('#p-torque').classList.toggle('free', !s.torque);
  $('#p-torque').innerHTML = s.torque ? TORQUE_ON : TORQUE_OFF;
  ['hold', 'preview', 'step', 'run'].forEach((m) => {
    $(`#p-${m}`).classList.toggle('sel', m === mode);
  });
  $('#p-step').disabled = s.chunked === false;
  $('#p-preview').disabled = s.chunked === false;
  // Only while this run is waiting for consent it delegated to us.
  $('#p-arm').hidden = !(s.arm_from_view && !s.armed && !s.dry_run);
  // Offered only by a run that has a scene it can begin again.
  $('#p-reset').hidden = !s.resettable;
  // The task box shows what the run has, until somebody starts typing a change.
  const task = s.task || '';
  $('#p-task').classList.toggle('unset', !task);
  if (document.activeElement !== $('#p-task')) $('#p-task').value = task;

  buildStrategies(s.strategies);
  paintSplice(s.splice, s.hz);
  $('#p-splice').hidden = s.chunked === false;
  buildTiles(s.cameras || []);
  refreshShown(s.cameras || [], s.chunk ? s.chunk.seq : -1);
  $('#p-shown-note').textContent = s.chunk
    ? `the window of plan #${s.chunk.seq}, ${((Date.now() / 1000) - s.chunk.at).toFixed(1)} s ago`
    : 'nothing sent yet';
  verdictOff = s.logging === false;
  $('#p-verdict').classList.toggle('off', verdictOff);
  ['#p-win', '#p-lose', '#p-void'].forEach((sel) => {
    $(sel).disabled = verdictOff;
  });
  paintVerdict(s.trial, s.verdicts);
  jointsTable(s.state, s.commanded);
  grippers(s.state, s.commanded);
  vitals(s);
  timing(s);

  twinFrame(s.twin_url);
  $('#p-twin-err').textContent = s.twin_url ? '' : (s.twin_error || 'no 3D twin');
  $('#p-twin-err').hidden = !!s.twin_url;

  // A new plan restarts the twin at its first action.
  if (s.chunk && s.chunk.seq !== twinSeq) {
    twinSeq = s.chunk.seq;
    twinN = s.chunk.n;
    twinAt = 0;
    twinBar();
  }
  drawChunk(s.chunk, s.state, s.hz, twinN ? twinAt : null);
}

// Cleared only by the poll recovering, so a refused mode or splice change
// stays on the screen long enough to be read.
let offline = false;
async function poll() {
  try {
    status = await j('/api/status');
    render(status);
    if (offline) { offline = false; $('#p-err').hidden = true; }
  } catch (e) {
    offline = true;
    $('#p-err').textContent = `the rollout is not answering: ${e.message}`;
    $('#p-err').hidden = false;
  }
}

poll();
setInterval(poll, 400);
