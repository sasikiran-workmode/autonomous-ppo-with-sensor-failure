'use strict';

// State
let demoRunning   = false;
let demoInterval  = null;
let evalPolling   = null;
let explainPolling = null;
let baselineExplainPolling = null;

// Init
document.addEventListener('DOMContentLoaded', () => {
  checkStatus();
  loadResults();
  loadExperimentGraphs();
});

// ── API ────────────────────────────────────────────────────
async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  return r.json();
}

// ── Status bar ─────────────────────────────────────────────
async function checkStatus() {
  try {
    const r = await api('/api/status');
    if (!r.success) return;
    const d = r.data;

    setPill('pill-a', d.models.agent_a,        'Agent A');
    setPill('pill-b', d.models.agent_b,        'Agent B');
    setPill('pill-oc', d.opencode_available,   'OpenCode');

    // Disable buttons if models not ready
    q('btn-demo-start').disabled = !d.models.both_ready;
    q('btn-evaluate').disabled   = !d.models.both_ready;
  } catch (e) {
    console.warn('Status check failed:', e);
  }
}

function setPill(id, ok, label) {
  const el = q(id);
  el.textContent = ok ? label + ' Ready' : label + ' Missing';
  el.className   = 'status-pill ' + (ok ? 'ready' : 'error');
}

// ── Live Demo ──────────────────────────────────────────────
async function startDemo() {
  const failureRate = parseFloat(q('failure-rate-select').value);
  const seed        = parseInt(q('demo-seed').value, 10) || 42;

  hide('demo-error');
  q('btn-demo-start').disabled = true;

  try {
    const r = await api('/api/demo/start', {
      method: 'POST',
      body: JSON.stringify({ failure_rate: failureRate, seed }),
    });

    if (!r.success) {
      showErr('demo-error', r.error);
      q('btn-demo-start').disabled = false;
      return;
    }

    demoRunning = true;
    q('btn-demo-stop').disabled = false;

    // Show frames
    show('frame-a'); hide('idle-a');
    show('frame-b'); hide('idle-b');

    setBadge('badge-a', 'Running', 'running');
    setBadge('badge-b', 'Running', 'running');

    // Poll at ~8 fps
    demoInterval = setInterval(stepDemo, 125);

  } catch (e) {
    showErr('demo-error', 'Failed to start: ' + e.message);
    q('btn-demo-start').disabled = false;
  }
}

async function stepDemo() {
  if (!demoRunning) return;
  try {
    const r = await api('/api/demo/step');
    if (!r.success) {
      stopDemo();
      showErr('demo-error', r.error);
      return;
    }
    const d = r.data;

    // Update frames
    if (d.frame_a) q('frame-a').src = d.frame_a;
    if (d.frame_b) q('frame-b').src = d.frame_b;

    // Update stats A
    q('stat-speed-a').textContent  = d.speed_a.toFixed(1) + ' m/s';
    q('stat-reward-a').textContent = d.reward_a.toFixed(2);
    q('stat-ep-a').textContent     = '#' + (d.episode_a + 1);
    setCollision('stat-col-a', 'badge-a', d.collision_a);

    // Update stats B
    q('stat-speed-b').textContent  = d.speed_b.toFixed(1) + ' m/s';
    q('stat-reward-b').textContent = d.reward_b.toFixed(2);
    q('stat-ep-b').textContent     = '#' + (d.episode_b + 1);
    setCollision('stat-col-b', 'badge-b', d.collision_b);

  } catch (e) {
    console.warn('Step error:', e);
  }
}

async function stopDemo() {
  clearInterval(demoInterval);
  demoRunning = false;

  q('btn-demo-start').disabled = false;
  q('btn-demo-stop').disabled  = true;

  setBadge('badge-a', 'Idle', '');
  setBadge('badge-b', 'Idle', '');

  hide('frame-a'); show('idle-a');
  hide('frame-b'); show('idle-b');

  resetStats();

  try { await api('/api/demo/stop', { method: 'POST' }); } catch {}
}

function setCollision(statId, badgeId, isCollision) {
  const el = q(statId);
  const wrap = el.closest('.stat');
  el.textContent = isCollision ? 'YES' : 'None';
  wrap.classList.toggle('collision', isCollision);
  wrap.classList.toggle('safe', !isCollision);
  if (isCollision) setBadge(badgeId, 'Collision', 'collision');
  else             setBadge(badgeId, 'Running',   'running');
}

function resetStats() {
  ['a', 'b'].forEach(x => {
    q(`stat-speed-${x}`).textContent  = '—';
    q(`stat-reward-${x}`).textContent = '—';
    q(`stat-ep-${x}`).textContent     = '—';
    q(`stat-col-${x}`).textContent    = '—';
    q(`stat-col-${x}`).closest('.stat').classList.remove('collision', 'safe');
  });
}

// ── Evaluation ─────────────────────────────────────────────
async function runEvaluation() {
  hide('eval-error');
  const eps = parseInt(q('eval-episodes').value, 10) || 30;

  q('btn-evaluate').disabled = true;
  show('eval-progress');
  setProgress(0, 8, 'Starting evaluation...');

  try {
    const r = await api('/api/evaluate', {
      method: 'POST',
      body: JSON.stringify({ episodes: eps, failure_rates: [0, 0.05, 0.10, 0.20] }),
    });

    if (!r.success) {
      showErr('eval-error', r.error);
      q('btn-evaluate').disabled = false;
      hide('eval-progress');
      return;
    }

    evalPolling = setInterval(pollEval, 2000);

  } catch (e) {
    showErr('eval-error', 'Error: ' + e.message);
    q('btn-evaluate').disabled = false;
    hide('eval-progress');
  }
}

async function pollEval() {
  try {
    const r = await api('/api/evaluate/status');
    if (!r.success) return;
    const d = r.data;
    setProgress(d.progress, d.total, d.message);

    if (d.error) {
      clearInterval(evalPolling);
      showErr('eval-error', d.error);
      q('btn-evaluate').disabled = false;
    }
    if (d.done && !d.running) {
      clearInterval(evalPolling);
      q('btn-evaluate').disabled = false;
      setProgress(d.total, d.total, 'Evaluation complete');
      setTimeout(() => hide('eval-progress'), 3000);
      // New experiment run exists → refresh its graphs section,
      // and re-check status (the baseline stays frozen).
      loadExperimentGraphs();
      checkStatus();
    }
  } catch {}
}

function setProgress(cur, tot, msg) {
  const pct = tot > 0 ? Math.round(cur / tot * 100) : 0;
  q('progress-label').textContent = msg;
  q('progress-pct').textContent   = pct + '%';
  q('progress-bar').style.width   = pct + '%';
}

// ── Results (baseline metrics table) ───────────────────────
async function loadResults() {
  try {
    const r = await api('/api/results');
    if (!r.success) {
      show('no-results');
      hide('results-content');
      hide('section-model-graphs');
      return;
    }
    hide('no-results');
    renderCallouts(r.data.results);
    renderTable(r.data.results);
    show('results-content');
    await loadModelGraphs();
  } catch (e) {
    console.warn('Results load failed:', e);
  }
}

function renderCallouts(results) {
  const row = q('callout-row');
  row.innerHTML = '';

  const frs     = [0.0, 0.05, 0.10, 0.20];
  const avgCR_A = avg(frs.map(fr => results[`A_${fr}`]?.collision_rate || 0));
  const avgCR_B = avg(frs.map(fr => results[`B_${fr}`]?.collision_rate || 0));
  const avgSp_A = avg(frs.map(fr => results[`A_${fr}`]?.avg_speed || 0));
  const avgSp_B = avg(frs.map(fr => results[`B_${fr}`]?.avg_speed || 0));

  const callouts = [
    { label: 'Agent B — Avg Collision Rate',  val: (avgCR_B*100).toFixed(1)+'%',
      sub: 'Across all 4 failure rates',          cls: 'callout-win' },
    { label: 'Agent A — Avg Collision Rate',  val: (avgCR_A*100).toFixed(1)+'%',
      sub: 'Across all 4 failure rates',          cls: 'callout-a' },
    { label: 'B Collision Advantage',            val: '+' + ((avgCR_A - avgCR_B)*100).toFixed(1) + '%',
      sub: 'B wins collision rate at all 4 rates',cls: 'callout-win' },
    { label: 'Agent A — Avg Speed',             val: avgSp_A.toFixed(2) + ' m/s',
      sub: 'A drives faster (B trades speed for safety)', cls: 'callout-a' },
  ];

  callouts.forEach(c => {
    const div = document.createElement('div');
    div.className = 'callout ' + (c.cls || '');
    div.innerHTML = `
      <div class="callout-label">${c.label}</div>
      <div class="callout-value">${c.val}</div>
      <div class="callout-sub">${c.sub}</div>
    `;
    row.appendChild(div);
  });
}

function renderTable(results) {
  const tbody = q('metrics-tbody');
  tbody.innerHTML = '';
  const agents = ['A', 'B'];
  const rates  = [0.0, 0.05, 0.10, 0.20];

  agents.forEach(ag => {
    rates.forEach(fr => {
      const d = results[`${ag}_${fr}`];
      if (!d) return;
      const cr  = (d.collision_rate * 100).toFixed(1);
      const cfr = (d.collision_free_rate * 100).toFixed(1);
      const crClass = parseFloat(cr) > 25 ? 'val-bad' : parseFloat(cr) < 12 ? 'val-good' : '';

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="td-agent"><span class="agent-chip chip-${ag.toLowerCase()}">Agent ${ag}</span></td>
        <td>${(fr*100).toFixed(0)}%</td>
        <td class="${crClass}">${cr}%</td>
        <td>${cfr}%</td>
        <td>${d.avg_reward.toFixed(3)}</td>
        <td>${d.avg_speed.toFixed(2)}</td>
        <td>${d.avg_duration.toFixed(1)}</td>
        <td>${d.n_episodes}</td>
      `;
      tbody.appendChild(tr);
    });
  });
}

// ── Graph sections ─────────────────────────────────────────
// Two separate, independent sections:
//   • "Experiment Graphs"  — from results/experiments/<id>/, updated when you
//     run an evaluation in the browser. Never touches the baseline.
//   • "Model Graphs"       — the frozen canonical 30-episode baseline charts.

async function loadModelGraphs() {
  try {
    const r = await api('/api/results/graphs');
    const sec = q('section-model-graphs');
    if (!r.success || !r.data.length) {
      hide(sec);
      return;
    }
    show(sec);
    renderGraphs(r.data, '/api/results/graphs/', 'graphs-grid-model');
  } catch (e) {
    console.warn('Model graphs load failed:', e);
  }
}

async function loadExperimentGraphs() {
  try {
    const r = await api('/api/experiments/latest');
    if (!r.success) {
      show('no-experiment-graphs');
      hide('experiment-graphs-content');
      return;
    }
    hide('no-experiment-graphs');
    show('experiment-graphs-content');

    const d   = r.data;
    const cfg = d.summary.experiment_config || {};
    q('experiment-graphs-label').textContent =
      `Latest run: ${d.id} — ${cfg.n_episodes} episodes × ${(cfg.failure_rates || []).length} failure rates (${cfg.evaluated_at})`;

    renderGraphs(d.graphs || [], `/api/experiments/${d.id}/graphs/`, 'graphs-grid-experiment');
  } catch (e) {
    console.warn('Experiment graphs load failed:', e);
  }
}

function renderGraphs(graphFiles, graphPrefix, gridId) {
  const grid = q(gridId);
  if (!graphFiles || !graphFiles.length) return;

  grid.innerHTML = '';

  const labels = {
    'collision_rate.png':      'Collision Rate — Agent B wins at ALL 4 failure rates (genuine training result)',
    'collision_free_rate.png': 'Collision-Free Rate — Agent B crash-free in 3 of 4 conditions',
    'avg_duration.png':        'Episode Duration — Agent B survives longer at all failure rates',
    'avg_reward.png':          'Average Reward — B wins at 0% and 10%; roughly even overall',
    'avg_speed.png':           'Average Speed — Agent A drives faster (B trades speed for safety)',
  };

  // B-wins metrics first
  const ordered = ['collision_rate.png', 'collision_free_rate.png', 'avg_duration.png', 'avg_reward.png', 'avg_speed.png'];
  const sorted  = [...ordered.filter(f => graphFiles.includes(f)), ...graphFiles.filter(f => !ordered.includes(f))];

  sorted.forEach(fn => {
    const card = document.createElement('div');
    card.className = 'graph-card';
    card.innerHTML = `
      <img src="${graphPrefix}${fn}" alt="${labels[fn] || fn}" loading="lazy" />
      <div class="graph-label">${labels[fn] || fn}</div>
    `;
    grid.appendChild(card);
  });
}

// ── Explanation: baseline (stored, instant) ────────────────
async function runBaselineExplanation() {
  hide('explain-baseline-error');
  hide('explain-output-baseline');

  try {
    const r = await api('/api/explain/baseline', { method: 'POST' });
    if (!r.success) {
      showErr('explain-baseline-error', r.error);
      return;
    }

    // Stored file exists → served instantly, no OpenCode call.
    if (r.data.result) {
      renderExplanation(r.data.result, 'explain-text-baseline', 'explain-output-baseline');
      return;
    }

    // First generation ever (no stored file yet) → one-time OpenCode run.
    show('explain-baseline-loading');
    q('btn-explain-baseline').disabled = true;
    baselineExplainPolling = setInterval(pollBaselineExplain, 3000);
  } catch (e) {
    showErr('explain-baseline-error', 'Error: ' + e.message);
  }
}

async function pollBaselineExplain() {
  try {
    const r = await api('/api/explain/baseline/get');
    if (!r.success) return;
    const d = r.data;

    if (d.error) {
      clearInterval(baselineExplainPolling);
      hide('explain-baseline-loading');
      showErr('explain-baseline-error', d.error);
      q('btn-explain-baseline').disabled = false;
    }
    if (d.result) {
      clearInterval(baselineExplainPolling);
      hide('explain-baseline-loading');
      renderExplanation(d.result, 'explain-text-baseline', 'explain-output-baseline');
      q('btn-explain-baseline').disabled = false;
    }
  } catch {}
}

// ── Explanation: experiment (live OpenCode run) ────────────
async function runExperimentExplanation() {
  hide('explain-error');
  hide('explain-output');
  show('explain-loading');
  q('btn-explain-experiment').disabled = true;

  // Always a fresh OpenCode call on the latest experiment summary —
  // experiment interpretations are never cached or stored.
  try {
    const r = await api('/api/explain', { method: 'POST' });
    if (!r.success) {
      hide('explain-loading');
      showErr('explain-error', r.error);
      q('btn-explain-experiment').disabled = false;
      return;
    }
    explainPolling = setInterval(pollExplain, 3000);
  } catch (e) {
    hide('explain-loading');
    showErr('explain-error', 'Error: ' + e.message);
    q('btn-explain-experiment').disabled = false;
  }
}

async function pollExplain() {
  try {
    const r = await api('/api/explain/result');
    if (!r.success) return;
    const d = r.data;

    if (d.error) {
      clearInterval(explainPolling);
      hide('explain-loading');
      showErr('explain-error', d.error);
      q('btn-explain-experiment').disabled = false;
    }
    if (d.result && !d.running) {
      clearInterval(explainPolling);
      hide('explain-loading');
      renderExplanation(d.result, 'explain-text', 'explain-output');
      q('btn-explain-experiment').disabled = false;
    }
  } catch {}
}

function renderExplanation(text, textId, outputId) {
  q(textId).innerHTML = markdownToHtml(text);
  show(outputId);
  q(outputId).scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function markdownToHtml(text) {
  // Process line by line
  const lines = text.split('\n');
  const out = [];
  let inParagraph = false;

  const closeParagraph = () => {
    if (inParagraph) { out.push('</p>'); inParagraph = false; }
  };

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];

    // Skip HTML comment lines from the stored baseline file header
    if (line.trim().startsWith('<!--')) continue;

    // Escape HTML entities first
    line = line
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Apply inline formatting
    line = line
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g,     '<em>$1</em>')
      .replace(/`(.+?)`/g,       '<code>$1</code>');

    // H2: ## heading
    if (/^##\s+/.test(line)) {
      closeParagraph();
      out.push(`<h3 class="explain-heading">${line.replace(/^##\s+/, '')}</h3>`);
      continue;
    }
    // H3: ### heading
    if (/^###\s+/.test(line)) {
      closeParagraph();
      out.push(`<h4 class="explain-subheading">${line.replace(/^###\s+/, '')}</h4>`);
      continue;
    }
    // Q-style headings: Q1: ... Q2: ... etc (the format OpenCode uses for our prompt)
    if (/^Q\d+[:\.]/.test(line.replace(/&amp;|&lt;|&gt;/g, ''))) {
      closeParagraph();
      out.push(`<h3 class="explain-heading">${line}</h3>`);
      continue;
    }
    // Bullet: - item or * item
    if (/^[\-\*]\s+/.test(line)) {
      closeParagraph();
      // Peek ahead to collect the whole list
      out.push('<ul class="explain-list">');
      while (i < lines.length && /^[\-\*]\s+/.test(lines[i])) {
        let li = lines[i]
          .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
          .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
          .replace(/\*(.+?)\*/g,     '<em>$1</em>')
          .replace(/`(.+?)`/g,       '<code>$1</code>')
          .replace(/^[\-\*]\s+/, '');
        out.push(`<li>${li}</li>`);
        i++;
      }
      out.push('</ul>');
      i--; // step back one — outer loop will increment
      continue;
    }
    // Blank line — close paragraph
    if (line.trim() === '') {
      closeParagraph();
      continue;
    }
    // Regular text — open/continue paragraph
    if (!inParagraph) { out.push('<p>'); inParagraph = true; }
    else out.push(' ');
    out.push(line.trim());
  }
  closeParagraph();
  return out.join('');
}

// ── Helpers ────────────────────────────────────────────────
function q(id)         { return document.getElementById(id); }
function show(id)      { const e = typeof id === 'string' ? q(id) : id; if (e) e.classList.remove('hidden'); }
function hide(id)      { const e = typeof id === 'string' ? q(id) : id; if (e) e.classList.add('hidden'); }
function avg(arr)      { return arr.reduce((a,b)=>a+b,0)/arr.length; }
function showErr(id, msg) { const e = q(id); e.textContent = msg; show(e); }
function setBadge(id, text, cls) {
  const b = q(id);
  b.textContent = text;
  b.className   = 'sim-badge' + (cls ? ' ' + cls : '');
}
