// Drawing a curve, and nothing else. No data, no polling, no endpoints.
//
// Hand-rolled 2-D canvas, deliberately: this console vendors no JavaScript and
// has no build step, and it is served on loopback to a machine that may have no
// route out. A charting library from a CDN would work on the laptop and fail on
// the rig, which is the wrong way round.
//
// Split out of training_chart.js when the runs view grew a panel grid. What
// lives here is what a second view could reuse; what knows about runs, machines
// and projects lives in training_jobs.js.

// A canvas is transparent: with nothing painted into it a line is drawn onto
// whatever the page happens to be, which in a dark browser is black on black.
function chartPalette() {
  const dark = window.matchMedia
    && matchMedia('(prefers-color-scheme: dark)').matches;
  return dark
    ? {bg: '#151a21', grid: '#2b323c', axis: '#7c8797', ink: '#d6dbe3',
       ghost: 'rgba(214,219,227,0.22)',
       series: ['#60a5fa', '#fbbf24', '#4ade80', '#f472b6', '#a78bfa', '#22d3ee']}
    : {bg: '#fbfcfe', grid: '#e6e9ef', axis: '#8b93a1', ink: '#31363f',
       ghost: 'rgba(49,54,63,0.18)',
       series: ['#1d4ed8', '#c2410c', '#15803d', '#be185d', '#6d28d9', '#0e7490']};
}

// Device-pixel-ratio aware, so a line is one pixel and not a grey smear.
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

function fmtSeconds(s) {
  if (s === null || s === undefined) return '–';
  if (s < 90) return `${Math.round(s)}s`;
  if (s < 5400) return `${Math.round(s / 60)}m`;
  const h = Math.floor(s / 3600);
  return `${h}h${String(Math.round((s % 3600) / 60)).padStart(2, '0')}`;
}

const fmtInt = (n) => (n === null || n === undefined ? '–' : n.toLocaleString('en-GB'));
const fmtNum = (n, dp = 3) => (n === null || n === undefined ? '–' : n.toFixed(dp));

// A number for a table cell, at a width a person can compare down a column.
function fmtValue(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1000) return fmtInt(Math.round(v));
  if (a >= 1) return v.toFixed(2);
  if (a >= 0.001) return v.toFixed(4);
  return v.toExponential(1);
}

// ── Smoothing ───────────────────────────────────────────────────────────────

// wandb's exponential moving average, and its convention: the slider is the
// WEIGHT given to the running average, so 0 is the raw curve and 0.95 is heavy.
// The raw series stays on the chart behind it -- a smoothed curve alone hides
// how noisy the run actually was, which for a loss curve is half the reading.
function smoothPoints(points, weight) {
  if (!(weight > 0) || points.length < 2) return points;
  const out = [];
  let acc = points[0].y;
  // Debias the first few points, or a heavy weight drags the whole start of
  // the curve towards the first sample and invents a shape that is not there.
  let count = 0;
  for (const q of points) {
    acc = acc * weight + q.y * (1 - weight);
    count += 1;
    out.push({...q, y: acc / (1 - Math.pow(weight, count))});
  }
  return out;
}

// ── The chart ───────────────────────────────────────────────────────────────

// series: [{label, colour, points: [{x, y}]}]. Log-y by default, because this
// rig's runs go 10.1 -> 0.006: on a linear axis everything after the first few
// hundred steps is one flat line along the floor, which is the part worth
// seeing.
//
// Returns the geometry it drew with, so a hover readout can find the point
// under the pointer without recomputing the scales and disagreeing with the
// picture.
function drawCurves(canvas, series, opts = {}) {
  const p = chartPalette();
  const [ctx, W, H] = fitCanvas(canvas);
  const logY = opts.logY !== false;
  const smooth = opts.smooth || 0;
  const padL = 52, padR = 8, padT = 8, padB = 18;

  ctx.fillStyle = p.bg;
  ctx.fillRect(0, 0, W, H);

  const drawn = series.map((s, i) => {
    const raw = (s.points || [])
      .filter((q) => Number.isFinite(q.y) && (!logY || q.y > 0));
    return {
      ...s,
      colour: s.colour || p.series[i % p.series.length],
      raw,
      points: smoothPoints(raw, smooth),
    };
  });

  const all = drawn.flatMap((s) => s.raw);
  if (!all.length) {
    ctx.fillStyle = p.axis;
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText(opts.empty || 'nothing logged yet', padL, H / 2);
    return null;
  }

  const xs = all.map((q) => q.x), ys = all.map((q) => q.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (y1 === y0) { y1 = y0 + Math.abs(y0 || 1) * 0.1; }
  const tf = logY ? Math.log10 : (v) => v;
  const ty0 = tf(y0), ty1 = tf(y1);

  const X = (v) => padL + ((v - x0) / ((x1 - x0) || 1)) * (W - padL - padR);
  const Y = (v) => H - padB - ((tf(v) - ty0) / ((ty1 - ty0) || 1)) * (H - padT - padB);

  ctx.strokeStyle = p.grid;
  ctx.lineWidth = 1;
  ctx.fillStyle = p.axis;
  ctx.font = '10px system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  const ticks = H > 120 ? [0, 0.25, 0.5, 0.75, 1] : [0, 0.5, 1];
  for (const frac of ticks) {
    const value = logY ? 10 ** (ty0 + frac * (ty1 - ty0)) : y0 + frac * (y1 - y0);
    const y = Math.round(Y(value)) + 0.5;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(fmtValue(value), 3, y);
  }
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(opts.xLabel || '', padL, H - 4);
  const right = opts.xMax !== undefined ? opts.xMax : x1;
  const rightText = opts.xUnit === 'h' ? `${right.toFixed(1)} h`
                                       : fmtInt(Math.round(right));
  ctx.fillText(rightText, W - padR - ctx.measureText(rightText).width, H - 4);

  for (const s of drawn) {
    if (!s.points.length) continue;
    if (smooth > 0) {
      // The raw curve, ghosted. Same colour would compete with the smoothed
      // line; the ink colour at low alpha reads as "this is the same data".
      ctx.strokeStyle = p.ghost;
      ctx.lineWidth = 1;
      ctx.beginPath();
      s.raw.forEach((q, k) => (k ? ctx.lineTo(X(q.x), Y(q.y))
                                 : ctx.moveTo(X(q.x), Y(q.y))));
      ctx.stroke();
    }
    ctx.strokeStyle = s.colour;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    s.points.forEach((q, k) => (k ? ctx.lineTo(X(q.x), Y(q.y))
                                  : ctx.moveTo(X(q.x), Y(q.y))));
    ctx.stroke();
    // A single point draws no line at all, which reads as an empty chart.
    if (s.points.length === 1) {
      const q = s.points[0];
      ctx.fillStyle = s.colour;
      ctx.beginPath(); ctx.arc(X(q.x), Y(q.y), 2.5, 0, 7); ctx.fill();
    }
  }
  return {series: drawn, X, Y, x0, x1, padL, padR, W, H};
}

// The crosshair. A vertical rule at the nearest x, and the values written into
// a row UNDER the canvas rather than a floating tooltip: a tooltip near the
// right-hand edge clips, and a row can hold every series at once, which is
// what a comparison wants.
function attachHover(canvas, readout, redraw) {
  if (canvas.dataset.hovered) return;
  canvas.dataset.hovered = '1';
  const clear = () => { readout.textContent = ''; redraw(); };
  canvas.onmouseleave = clear;
  canvas.onmousemove = (event) => {
    const geometry = redraw();
    if (!geometry) return;
    const rect = canvas.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const p = chartPalette();
    const ctx = canvas.getContext('2d');
    const parts = [];
    let ruleX = null;
    for (const s of geometry.series) {
      if (!s.points.length) continue;
      let best = s.points[0], bestD = Infinity;
      for (const q of s.points) {
        const d = Math.abs(geometry.X(q.x) - px);
        if (d < bestD) { bestD = d; best = q; }
      }
      if (ruleX === null) ruleX = geometry.X(best.x);
      parts.push(
        `<span class="muted"><span class="swatch" style="background:${s.colour}"></span>` +
        `${s.label}: <b>${fmtValue(best.y)}</b></span>`);
      ctx.fillStyle = s.colour;
      ctx.beginPath();
      ctx.arc(geometry.X(best.x), geometry.Y(best.y), 3, 0, 7);
      ctx.fill();
    }
    if (ruleX !== null) {
      ctx.strokeStyle = p.axis;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(Math.round(ruleX) + 0.5, 0);
      ctx.lineTo(Math.round(ruleX) + 0.5, geometry.H);
      ctx.stroke();
    }
    readout.innerHTML = parts.join(' ');
  };
}
