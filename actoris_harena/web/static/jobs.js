// The job dock: the slow, whole-dataset work (a merge, a compaction, freeing a
// deleted dataset) runs on the server and is watched here, in the corner of the
// page, instead of holding a dialog open. It polls only while one is running.

let jobs = [];
let jobTimer = null;
let jobsSeen = new Set();
let jobsFirstPoll = true;

function jobLine(job) {
  const verb = {merge: 'merging', delete: 'deleting',
                compact: 'rewriting', repair: 'repairing'}[job.kind] || job.kind;
  return `${verb} ${job.name} — ${job.message}`;
}

function jobAge(job) {
  const end = job.finished || (Date.now() / 1000);
  const s = Math.max(0, Math.round(end - job.started));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
}

function renderJobs() {
  const dock = $('#jobdock');
  dock.hidden = jobs.length === 0;
  if (dock.hidden) return;
  const running = jobs.filter(j => j.state === 'running');
  const head = running.length ? jobLine(running[0])
    : `${jobs.length} finished job(s)`;
  $('#jobdock-line').textContent =
    running.length > 1 ? `${head} (+${running.length - 1} more)` : head;
  $('#jobdock-list').innerHTML = jobs.map(job => `<li>
      <span class="job-${job.state}">${job.state}</span>
      <span class="muted">${jobAge(job)}</span> — ${jobLine(job)}
    </li>`).join('');
}

async function pollJobs() {
  try { jobs = await j('/api/jobs'); }
  catch (e) { jobs = []; }
  for (const job of jobs) {
    // A job that has just finished changed the drive: the dataset list is out
    // of date until it is reloaded, whichever tab started it. On the first poll
    // the finished ones are history — the list is already current.
    if (job.state === 'running') { jobsSeen.delete(job.id); continue; }
    if (jobsSeen.has(job.id)) continue;
    jobsSeen.add(job.id);
    if (!jobsFirstPoll && window.onJobFinished) window.onJobFinished(job);
  }
  jobsFirstPoll = false;
  renderJobs();
  clearTimeout(jobTimer);
  if (jobs.some(job => job.state === 'running')) {
    jobTimer = setTimeout(pollJobs, 2000);
  }
}

$('#jobdock-head').onclick = () => {
  const list = $('#jobdock-list');
  list.hidden = !list.hidden;
  $('#jobdock-caret').textContent = list.hidden ? '▴' : '▾';
};

window.onJobFinished = (job) => {
  loadDatasets();
  // A compaction renumbers the recordings of the dataset on screen, so the
  // episode list beside it is stale in a way reloading the datasets cannot fix.
  // A repair renumbers the recordings just as a compaction does.
  if (job && ['compact', 'repair'].includes(job.kind) && job.name === curDataset) {
    loadEpisodes();
  }
};

// Started from here so a job that outlives the page (or a page opened while one
// runs) is picked up on load, not only when this browser started it.
pollJobs();
