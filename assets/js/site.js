// site.js - reveals, scroll progress bar and journey timeline (vanilla ES module).

// Flag JS as running so CSS can scope the hidden reveal state to html.js .reveal.
const root = document.documentElement;
root.classList.add('js');

const reduceMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// Cache element references once.
const els = {
  progress: document.getElementById('progress'),
  route: document.getElementById('route'),
  chapters: [0, 1, 2].map((i) => document.getElementById(`jch${i}`)),
  stops: [],
};
if (els.route) {
  els.route.querySelectorAll('[data-stop]').forEach((stop) => {
    const i = parseInt(stop.dataset.stop, 10);
    if (i >= 0 && i <= 2) els.stops[i] = stop;
  });
}

/* ---------- reveals: fire once, never un-reveal ---------- */

function revealEl(el) {
  const delay = parseInt(el.dataset.revealDelay || '0', 10) || 0;
  if (delay > 0) {
    el.style.transitionDelay = `${delay}ms`;
    // Drop the delay once the reveal has started so later hover transitions stay snappy.
    setTimeout(() => { el.style.transitionDelay = ''; }, delay + 1200);
  }
  el.classList.add('is-in');
}

function revealAll(list) {
  list.forEach((el) => el.classList.add('is-in'));
}

function registerReveals() {
  const list = Array.from(document.querySelectorAll('.reveal, [data-reveal]'));
  if (!list.length) return;
  if (reduceMotion || !('IntersectionObserver' in window)) {
    revealAll(list);
    return;
  }
  const io = new IntersectionObserver((entries, obs) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      obs.unobserve(entry.target);
      revealEl(entry.target);
    });
  }, { threshold: 0.12 });
  list.forEach((el) => io.observe(el));
}

/* ---------- journey timeline: one-way fill ---------- */

const routeState = { done: [false, false, false], progress: 0, complete: false };

function markStop(i) {
  if (routeState.done[i]) return;
  routeState.done[i] = true;
  if (els.stops[i]) els.stops[i].classList.add('is-done');
}

function setRouteProgress(v) {
  if (!els.route || v <= routeState.progress) return;
  routeState.progress = v;
  els.route.style.setProperty('--route-progress', v.toFixed(4));
}

function fillRoute() {
  [0, 1, 2].forEach(markStop);
  setRouteProgress(1);
  routeState.complete = true;
}

// A stop completes once its chapter's center passes the viewport middle;
// the line fills continuously between consecutive chapter centers.
function updateRoute() {
  if (routeState.complete || !els.route) return;
  if (els.chapters.some((c) => !c)) return;
  const vmid = window.innerHeight / 2;
  const centers = els.chapters.map((c) => {
    const r = c.getBoundingClientRect();
    return r.top + r.height / 2;
  });
  centers.forEach((c, i) => { if (c <= vmid) markStop(i); });
  const part = (i) => clamp((vmid - centers[i]) / ((centers[i + 1] - centers[i]) || 1), 0, 1);
  setRouteProgress(0.5 * part(0) + 0.5 * part(1));
  if (routeState.progress >= 1 && routeState.done.every(Boolean)) routeState.complete = true;
}

/* ---------- scroll progress bar ---------- */

function updateProgress() {
  if (!els.progress) return;
  const max = root.scrollHeight - window.innerHeight;
  const v = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
  els.progress.style.transform = `scaleX(${v.toFixed(4)})`;
}

/* ---------- rAF-throttled scroll work ---------- */

let ticking = false;

function onFrame() {
  ticking = false;
  updateProgress();
  updateRoute();
}

function requestUpdate() {
  if (ticking) return;
  ticking = true;
  requestAnimationFrame(onFrame);
}

/* ---------- boot ---------- */

function init() {
  registerReveals();
  if (reduceMotion) {
    fillRoute();
    return;
  }
  if (els.route) els.route.style.setProperty('--route-progress', '0');
  window.addEventListener('scroll', requestUpdate, { passive: true });
  window.addEventListener('resize', requestUpdate, { passive: true });
  window.addEventListener('load', requestUpdate, { once: true });
  requestUpdate();
}

init();
