// lab.js - Hero Lab controller.
//
// One clock drives every viewport. Prototypes render as a pure function of
// their cycle time t, so pause, scrub, speed and replay are exact and any frame
// can be reproduced from the URL (?p=A&t=9.5&pause=1&ui=0 for capture).

import { PROTOTYPES } from './prototypes/registry.js';
import { REVIEW } from './prototypes/review.js';
import { loadFonts, computeLayout } from './core/text.js';
import { createReveal, revealTiming, revealPhases } from './core/reveal.js';

const $ = (s, r = document) => r.querySelector(s);
const params = new URLSearchParams(location.search);
const reduceMotion = !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
const STORE_KEY = 'heroLab.v1';
const IDS = PROTOTYPES.map((p) => p.id);
const CRITERIA = [
  ['visual', 'Visual quality'],
  ['smooth', 'Transition smoothness'],
  ['read', 'Readability'],
  ['perf', 'Performance'],
  ['story', 'Communicates interests'],
];

const store = (() => {
  try { return JSON.parse(localStorage.getItem(STORE_KEY)) || {}; } catch { return {}; }
})();
const save = () => { try { localStorage.setItem(STORE_KEY, JSON.stringify(store)); } catch {} };
store.scores ??= {};

const pick = (v, ok, d) => (ok.includes(v) ? v : d);
const state = {
  id: pick(params.get('p') || store.last, IDS, 'A'),
  compareWith: null,
  mode: pick(params.get('mode'), ['single', 'compare', 'grid'], 'single'),
  frame: pick(params.get('frame'), ['fill', 'mobile'], 'fill'),
  quality: pick(params.get('q'), ['high', 'low'], 'high'),
  speed: Number(params.get('speed')) || 1,
  elapsed: Number(params.get('t')) || 0,
  playing: !(params.has('pause') || params.has('t') || reduceMotion),
  boardSrc: 'you',
};
state.compareWith = IDS.find((i) => i !== state.id);
const bare = params.get('ui') === '0';
if (bare) document.body.classList.add('bare');

/* ---------------- prototype loading ---------------- */

const defs = {};
async function loadDef(id) {
  if (!defs[id]) {
    const entry = PROTOTYPES.find((p) => p.id === id);
    defs[id] = entry.load().then((m) => ({ ...m.default, id, name: m.default.name || entry.name }));
  }
  return defs[id];
}

/* ---------------- viewport ---------------- */

const perf = {}; // id -> { cost, fps, n }

class Viewport {
  constructor(parent, id, { compareSlot = false } = {}) {
    this.id = id;
    this.el = document.createElement('div');
    this.el.className = 'vp';
    this.host = document.createElement('div');
    this.host.className = 'vp-host';
    this.tag = document.createElement('div');
    this.tag.className = 'vp-tag';
    this.el.append(this.host, this.tag);
    parent.appendChild(this.el);
    this.compareSlot = compareSlot;
    this.cost = 0;
    this.ro = new ResizeObserver(() => this.scheduleResize());
    this.ro.observe(this.host);
    this.el.addEventListener('click', (e) => {
      if (state.mode === 'grid' && e.target.tagName !== 'SELECT') { setMode('single'); select(this.id); }
    });
  }

  async mount(id = this.id) {
    this.unmount();
    this.id = id;
    const token = (this.token = {});
    this.host.innerHTML = '<div class="vp-loading">Loading</div>';
    let def;
    try {
      [def] = await Promise.all([loadDef(id), loadFonts()]);
    } catch (err) {
      return this.fail(err);
    }
    if (token !== this.token) return;
    this.host.innerHTML = '';
    this.def = def;
    this.timing = revealTiming(def.nameAt, def.duration);
    const { w, h } = this.size();
    this.ctx = {
      container: this.host,
      width: w,
      height: h,
      quality: state.mode === 'grid' ? 'low' : state.quality,
      reducedMotion: reduceMotion,
      layout: computeLayout(w, h, { weight: def.nameWeight ?? 700 }),
      timing: this.timing,
    };
    try {
      this.inst = await def.create(this.ctx);
      if (token !== this.token) { this.inst?.dispose?.(); return; }
      this.reveal = createReveal(this.host, def.reveal || {});
      this.reveal.setLayout(this.ctx.layout, this.timing);
    } catch (err) {
      return this.fail(err);
    }
    this.renderTag();
    dirty = true;
  }

  fail(err) {
    console.error(err);
    this.inst = null;
    this.host.innerHTML = `<div class="vp-error">Prototype ${this.id} failed to load\n\n${String(err?.message || err)}</div>`;
  }

  size() {
    return { w: Math.max(2, this.host.clientWidth), h: Math.max(2, this.host.clientHeight) };
  }

  scheduleResize() {
    clearTimeout(this.rz);
    this.rz = setTimeout(() => this.resize(), 120);
  }

  resize() {
    if (!this.inst || !this.ctx) return;
    const { w, h } = this.size();
    if (w === this.ctx.width && h === this.ctx.height) return;
    this.ctx.width = w; this.ctx.height = h;
    this.ctx.layout = computeLayout(w, h, { weight: this.def.nameWeight ?? 700 });
    try { this.inst.resize?.(this.ctx); } catch (err) { console.error(err); }
    this.reveal?.setLayout(this.ctx.layout, this.timing);
    dirty = true;
  }

  frame(elapsed, dt) {
    if (!this.inst) return;
    const t = ((elapsed % this.def.duration) + this.def.duration) % this.def.duration;
    this.t = t;
    const t0 = performance.now();
    try { this.inst.update(t, dt); } catch (err) { this.fail(err); return; }
    this.reveal.update(t);
    this.cost = this.cost * 0.9 + (performance.now() - t0) * 0.1;
    if (state.mode !== 'single') this.updateTag();
  }

  renderTag() {
    const opts = this.compareSlot
      ? `<select aria-label="Compare with">${PROTOTYPES.filter((p) => p.id !== state.id).map((p) => `<option value="${p.id}"${p.id === this.id ? ' selected' : ''}>${p.id} · ${p.name}</option>`).join('')}</select>`
      : `<b>${this.id}</b> · ${this.def?.name ?? ''}`;
    this.tag.innerHTML = `<span>${opts}</span><span class="ph"></span><i class="bar"></i>`;
    const sel = this.tag.querySelector('select');
    if (sel) sel.addEventListener('change', () => { state.compareWith = sel.value; this.mount(sel.value); });
  }

  updateTag() {
    const ph = this.tag.querySelector('.ph');
    const bar = this.tag.querySelector('.bar');
    if (!ph || this.t == null) return;
    const p = phaseAt(allPhases(this.def, this.timing), this.t);
    ph.textContent = p ? p.label : '';
    bar.style.width = `${(this.t / this.def.duration) * 100}%`;
  }

  unmount() {
    this.token = null;
    try { this.inst?.dispose?.(); } catch (err) { console.error(err); }
    this.reveal?.dispose();
    this.inst = null; this.reveal = null; this.def = null; this.ctx = null;
    this.host.innerHTML = '';
  }

  destroy() {
    this.unmount();
    this.ro.disconnect();
    this.el.remove();
  }
}

/* ---------------- phases ---------------- */

function allPhases(def, timing) {
  const own = (def.phases || []).map((p) => ({ ...p, reveal: false }));
  return own.concat(revealPhases(timing).map((p) => ({ ...p, reveal: true })));
}
const phaseAt = (phases, t) => phases.find((p) => t >= p.start && t < p.end) || phases[phases.length - 1];

/* ---------------- stage ---------------- */

const stage = $('#stage');
let viewports = [];
let dirty = true;

function buildStage() {
  viewports.forEach((v) => v.destroy());
  viewports = [];
  stage.dataset.mode = state.mode;
  stage.dataset.frame = state.frame;
  stage.style.setProperty('--cols', state.mode === 'single' ? 1 : state.mode === 'compare' ? 2 : 3);
  const ids = state.mode === 'single' ? [state.id]
    : state.mode === 'compare' ? [state.id, state.compareWith]
    : IDS;
  ids.forEach((id, i) => {
    const vp = new Viewport(stage, id, { compareSlot: state.mode === 'compare' && i === 1 });
    viewports.push(vp);
  });
  return Promise.all(viewports.map((v) => v.mount()));
}

let ready = null;
function rebuild() {
  ready = buildStage().then(() => {
    refreshPanel();
    buildTimeline();
    if (reduceMotion && !state.playing && !params.has('t')) {
      const v = viewports[0];
      if (v?.def) state.elapsed = v.def.posterTime ?? (v.timing.read[0] + 0.8);
      $('#rm-note').hidden = false;
    }
    dirty = true;
  });
  return ready;
}

/* ---------------- selection & modes ---------------- */

function select(id) {
  if (!IDS.includes(id)) return;
  state.id = id;
  store.last = id;
  save();
  if (state.compareWith === id) state.compareWith = IDS.find((i) => i !== id);
  renderTabs();
  if (state.mode === 'grid') { refreshPanel(); return; }
  state.elapsed = 0;
  rebuild();
}

function setMode(m) {
  if (m === state.mode) return;
  state.mode = m;
  pressSeg('#mode', 'mode', m);
  rebuild();
}

function setFrame(f) {
  state.frame = f;
  pressSeg('#frame', 'frame', f);
  stage.dataset.frame = f;
}

function pressSeg(sel, key, val) {
  document.querySelectorAll(`${sel} button`).forEach((b) => b.setAttribute('aria-pressed', String(b.dataset[key] === val)));
}

/* ---------------- tabs & panel ---------------- */

function renderTabs() {
  $('#tabs').innerHTML = PROTOTYPES.map((p) => {
    const fav = store.scores[p.id]?.star ? '<span class="fav" aria-label="shortlisted">★</span>' : '';
    return `<button class="tab" role="tab" data-id="${p.id}" aria-selected="${p.id === state.id}"><b>${p.id}</b>${p.short || p.name}${fav}</button>`;
  }).join('');
}

async function refreshPanel() {
  const def = await loadDef(state.id).catch(() => null);
  const entry = PROTOTYPES.find((p) => p.id === state.id);
  $('#about-id').textContent = `Prototype ${state.id}`;
  $('#about-name').textContent = def?.name || entry.name;
  $('#about-blurb').textContent = def?.blurb || '';
  const m = perf[state.id];
  const specs = [
    ['Cycle', def ? `${def.duration.toFixed(1)}s · name at ${def.nameAt.toFixed(1)}s` : '-'],
    ['Tech', def?.tech || '-'],
    ['Measured', m ? `${Math.round(m.fps)} fps · ${m.cost.toFixed(1)} ms/frame` : 'play in Single view'],
  ];
  $('#about-specs').innerHTML = specs.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');

  const s = (store.scores[state.id] ??= {});
  $('#criteria').innerHTML = CRITERIA.map(([k, label]) => `
    <div class="crit"><span>${label}</span><span class="dots" data-k="${k}">
      ${[1, 2, 3, 4, 5].map((n) => `<button data-n="${n}" aria-label="${label} ${n} of 5" aria-pressed="${s[k] === n}" class="${s[k] >= n ? 'on' : ''}">${n}</button>`).join('')}
    </span></div>`).join('');
  $('#star').setAttribute('aria-pressed', String(!!s.star));
  $('#star').textContent = s.star ? '★ Shortlisted' : '☆ Shortlist';
  $('#notes').value = s.notes || '';
  renderBoard();
}

function average(scores) {
  if (!scores) return 0;
  const vals = CRITERIA.map(([k]) => scores[k]).filter((v) => typeof v === 'number');
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
}

function renderBoard() {
  const src = state.boardSrc;
  const rows = PROTOTYPES.map((p) => {
    const sc = src === 'you' ? store.scores[p.id] : REVIEW.scores?.[p.id];
    return { ...p, avg: average(sc), star: src === 'you' && !!store.scores[p.id]?.star };
  }).sort((a, b) => b.avg - a.avg || Number(b.star) - Number(a.star));
  const best = rows[0]?.avg > 0 ? rows[0].id : null;
  $('#board').innerHTML = rows.map((r) => `
    <li data-id="${r.id}" class="${r.id === best ? 'top' : ''}"><b>${r.id}</b>
      <span>${r.name}${r.star ? ' <span style="color:#E0B04A">★</span>' : ''}${r.id === best ? '<span class="badge">Most promising</span>' : ''}<div class="meter"><i style="width:${(r.avg / 5) * 100}%"></i></div></span>
      <span class="val">${r.avg ? r.avg.toFixed(1) : '-'}</span></li>`).join('');
  $('#review').textContent = src === 'claude'
    ? [REVIEW.notes?.[state.id] ? `${state.id}: ${REVIEW.notes[state.id]}` : '', REVIEW.summary || ''].filter(Boolean).join('\n\n')
    : (rows.every((r) => !r.avg) ? 'Rate each prototype 1–5 on the five criteria; the leader is marked most promising. Scores stay in this browser.' : '');
}

/* ---------------- timeline ---------------- */

const tlEl = $('#timeline');
const phasesEl = $('#phases');
const playhead = $('#playhead');

function primary() { return viewports.find((v) => v.id === state.id && v.def) || viewports.find((v) => v.def); }

function buildTimeline() {
  const v = primary();
  if (!v) { phasesEl.innerHTML = ''; return; }
  const D = v.def.duration;
  tlEl.setAttribute('aria-valuemax', D.toFixed(1));
  phasesEl.innerHTML = allPhases(v.def, v.timing).map((p, i) =>
    `<div class="phase${p.reveal ? ' reveal' : ''}" data-i="${i}" style="width:${((p.end - p.start) / D) * 100}%" title="${p.label} · ${p.start.toFixed(1)}–${p.end.toFixed(1)}s">${p.label}</div>`).join('');
}

function updateTransport() {
  const v = primary();
  if (!v || v.t == null) return;
  const D = v.def.duration;
  playhead.style.left = `${(v.t / D) * 100}%`;
  $('#clock').textContent = `${v.t.toFixed(1).padStart(4, '0')} / ${D.toFixed(1)}s`;
  tlEl.setAttribute('aria-valuenow', v.t.toFixed(1));
  const phases = allPhases(v.def, v.timing);
  const now = phaseAt(phases, v.t);
  $('#phase-now').textContent = now?.label || '';
  const idx = phases.indexOf(now);
  phasesEl.querySelectorAll('.phase').forEach((el, i) => el.classList.toggle('now', i === idx));
}

function seekFromPointer(e) {
  const v = primary();
  if (!v) return;
  const r = tlEl.getBoundingClientRect();
  const f = Math.min(0.9999, Math.max(0, (e.clientX - r.left) / r.width));
  seek(f * v.def.duration);
}

function seek(t) {
  const v = primary();
  // keep the other viewports' relative phase when seeking: elapsed is shared
  state.elapsed = v ? Math.floor(state.elapsed / v.def.duration) * v.def.duration + t : t;
  if (state.elapsed < 0) state.elapsed = t;
  dirty = true;
}

/* ---------------- transport ---------------- */

function setPlaying(p) {
  state.playing = p;
  const b = $('#play');
  b.textContent = p ? '❚❚' : '▶';
  b.setAttribute('aria-label', p ? 'Pause' : 'Play');
  if (p) $('#rm-note').hidden = true;
}

function replay() { state.elapsed = 0; setPlaying(true); dirty = true; }

function setSpeed(s) {
  state.speed = s;
  $('#speed').value = String(s);
}

function setQuality(q) {
  state.quality = q;
  pressSeg('#quality', 'q', q);
  rebuild();
}

/* ---------------- loop & fps ---------------- */

let last = performance.now();
let fpsFrames = 0, fpsT0 = last, fps = 0;

function tick(now) {
  const real = Math.min(0.1, (now - last) / 1000);
  last = now;
  const dt = state.playing ? real * state.speed : 0;
  if (state.playing) state.elapsed += dt;
  if (state.playing || dirty) {
    dirty = false;
    for (const v of viewports) v.frame(state.elapsed, dt);
    updateTransport();
  }
  fpsFrames++;
  if (now - fpsT0 >= 500) {
    fps = (fpsFrames * 1000) / (now - fpsT0);
    fpsFrames = 0; fpsT0 = now;
    const cost = viewports.reduce((a, v) => a + v.cost, 0);
    const el = $('#fps');
    el.textContent = `${Math.round(fps)} fps · ${cost.toFixed(1)} ms`;
    el.classList.toggle('warn', state.playing && fps < 50);
    el.classList.toggle('bad', state.playing && fps < 30);
    if (state.playing && state.mode === 'single' && viewports[0]?.def) {
      const m = (perf[state.id] ??= { fps, cost, n: 0 });
      m.fps = m.fps * 0.8 + fps * 0.2; m.cost = m.cost * 0.8 + cost * 0.2; m.n++;
    }
  }
  requestAnimationFrame(tick);
}

/* ---------------- wiring ---------------- */

renderTabs();
pressSeg('#mode', 'mode', state.mode);
pressSeg('#frame', 'frame', state.frame);
pressSeg('#quality', 'q', state.quality);
setPlaying(state.playing);
setSpeed(state.speed);

$('#tabs').addEventListener('click', (e) => { const b = e.target.closest('.tab'); if (b) select(b.dataset.id); });
$('#mode').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) setMode(b.dataset.mode); });
$('#frame').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) setFrame(b.dataset.frame); });
$('#quality').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) setQuality(b.dataset.q); });
$('#play').addEventListener('click', () => setPlaying(!state.playing));
$('#replay').addEventListener('click', replay);
$('#speed').addEventListener('change', (e) => setSpeed(Number(e.target.value)));
$('#board-src').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  state.boardSrc = b.dataset.src; pressSeg('#board-src', 'src', b.dataset.src); renderBoard();
});
$('#board').addEventListener('click', (e) => { const li = e.target.closest('li'); if (li) select(li.dataset.id); });

$('#criteria').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  const k = b.parentElement.dataset.k, n = Number(b.dataset.n);
  const s = (store.scores[state.id] ??= {});
  s[k] = s[k] === n ? undefined : n;
  save(); refreshPanel();
});
$('#star').addEventListener('click', () => {
  const s = (store.scores[state.id] ??= {});
  s.star = !s.star; save(); refreshPanel(); renderTabs();
});
$('#notes').addEventListener('input', (e) => { (store.scores[state.id] ??= {}).notes = e.target.value; save(); });
$('#export').addEventListener('click', async () => {
  const out = JSON.stringify({ you: store.scores, measured: perf }, null, 2);
  try { await navigator.clipboard.writeText(out); $('#export').textContent = 'Copied'; } catch { console.log(out); $('#export').textContent = 'Logged to console'; }
  setTimeout(() => { $('#export').textContent = 'Copy scores as JSON'; }, 1600);
});

let scrubWasPlaying = false;
tlEl.addEventListener('pointerdown', (e) => {
  scrubWasPlaying = state.playing; setPlaying(false);
  tlEl.setPointerCapture(e.pointerId); seekFromPointer(e);
});
tlEl.addEventListener('pointermove', (e) => { if (tlEl.hasPointerCapture(e.pointerId)) seekFromPointer(e); });
tlEl.addEventListener('pointerup', () => { if (scrubWasPlaying) setPlaying(true); });

document.addEventListener('keydown', (e) => {
  if (e.target.closest('textarea, select, input')) return;
  const v = primary();
  if (e.key === ' ') { e.preventDefault(); setPlaying(!state.playing); }
  else if (e.key === 'r' || e.key === 'R') replay();
  else if (e.key >= '1' && e.key <= '6') select(IDS[Number(e.key) - 1]);
  else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
    e.preventDefault();
    const step = (e.shiftKey ? 0.05 : 0.5) * (e.key === 'ArrowRight' ? 1 : -1);
    if (v) seek(Math.min(v.def.duration - 0.001, Math.max(0, (v.t ?? 0) + step)));
  } else if (e.key === '[' || e.key === ']') {
    const speeds = [0.25, 0.5, 1, 1.5, 2];
    const i = speeds.indexOf(state.speed);
    setSpeed(speeds[Math.min(speeds.length - 1, Math.max(0, (i < 0 ? 2 : i) + (e.key === ']' ? 1 : -1)))]);
  } else if (e.key === 'g' || e.key === 'G') setMode(state.mode === 'grid' ? 'single' : 'grid');
  else if (e.key === 'c' || e.key === 'C') setMode(state.mode === 'compare' ? 'single' : 'compare');
  else if (e.key === 'm' || e.key === 'M') setFrame(state.frame === 'mobile' ? 'fill' : 'mobile');
});

rebuild();
requestAnimationFrame(tick);

// Capture / test hooks (used by tools/gauntlet.cjs).
window.heroLab = {
  get ready() { return ready; },
  state,
  viewports: () => viewports,
  select: (id) => { select(id); return ready; },
  async renderAt(t) {
    await ready;
    setPlaying(false);
    state.elapsed = t;
    for (const v of viewports) v.frame(t, 0);
    updateTransport();
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  },
};
