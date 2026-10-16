// site.js - scroll behaviours for the page (vanilla ES module).
// Ported from the original page script: scroll-in reveals (which also un-reveal),
// pointer tilt on the journey photos, scroll progress bar, the sticky journey
// route timeline, and smooth wheel scrolling. The molecule intro lives in molecule.js.

const root = document.documentElement;
root.classList.add('js');

const reduceMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
const motionOn = !reduceMotion;

const EASE = 'cubic-bezier(0.22,1,0.36,1)';

/* ---------- reveals ---------- */

// Hidden pose for each reveal kind (shared by initial state and un-reveal).
function hiddenTransform(kind) {
  if (kind === 'up') return 'translateY(36px)';
  if (kind === 'left') return 'translateX(40px)';
  if (kind === 'right') return 'translateX(-40px)';
  if (kind === 'scale') return 'scale(0.94)';
  return 'none';
}

function revealEl(el) {
  el.dataset.revealed = '1';
  const delay = parseInt(el.dataset.revealDelay || '0', 10);
  el.style.transition = `opacity 0.9s ${EASE} ${delay}ms, transform 0.9s ${EASE} ${delay}ms`;
  requestAnimationFrame(() => {
    el.style.opacity = '1';
    el.style.transform = 'none';
  });
}

function unrevealEl(el) {
  delete el.dataset.revealed;
  el.style.transition = `opacity 0.55s ${EASE}, transform 0.55s ${EASE}`;
  el.style.opacity = '0';
  el.style.transform = hiddenTransform(el.dataset.reveal);
}

// Hero elements keep the class-driven one-shot reveal (CSS holds their hidden state).
function revealClassEl(el) {
  const delay = parseInt(el.dataset.revealDelay || '0', 10) || 0;
  if (delay > 0) {
    el.style.transitionDelay = `${delay}ms`;
    setTimeout(() => { el.style.transitionDelay = ''; }, delay + 1200);
  }
  el.classList.add('is-in');
}

function registerReveals() {
  const all = Array.from(document.querySelectorAll('[data-reveal]'));
  const classed = all.filter((el) => el.classList.contains('reveal'));
  const inline = all.filter((el) => !el.classList.contains('reveal'));

  if (reduceMotion || !('IntersectionObserver' in window)) {
    classed.forEach((el) => el.classList.add('is-in'));
    inline.forEach((el) => {
      el.style.transition = 'none';
      el.style.opacity = '1';
      el.style.transform = 'none';
    });
    return;
  }

  if (classed.length) {
    const once = new IntersectionObserver((entries, obs) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        obs.unobserve(entry.target);
        revealClassEl(entry.target);
      });
    }, { threshold: 0.12 });
    classed.forEach((el) => once.observe(el));
  }

  if (inline.length) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) revealEl(entry.target);
        else if (entry.target.dataset.revealed) unrevealEl(entry.target);
      });
    }, { threshold: 0.12 });
    inline.forEach((el) => {
      if (el.dataset.revealInit) return;
      el.dataset.revealInit = '1';
      el.style.opacity = '0';
      const t = hiddenTransform(el.dataset.reveal);
      if (t !== 'none') el.style.transform = t;
      el.style.transition = 'none';
      io.observe(el);
    });
  }
}

/* ---------- pointer tilt on journey photos ---------- */

function registerTilts() {
  if (!motionOn) return;
  document.querySelectorAll('[data-jimg]').forEach((el) => {
    if (el.dataset.tiltInit) return;
    el.dataset.tiltInit = '1';
    const frame = el.firstElementChild;
    if (!frame) return;
    frame.style.transition = `transform 0.3s ${EASE}, box-shadow 0.3s`;
    frame.style.willChange = 'transform';
    el.addEventListener('pointermove', (e) => {
      if (e.pointerType && e.pointerType !== 'mouse') return; // no tilt-jitter while touch-scrolling
      const r = el.getBoundingClientRect();
      const nx = (e.clientX - r.left) / r.width - 0.5;
      const ny = (e.clientY - r.top) / r.height - 0.5;
      frame.style.transform = `perspective(900px) rotateY(${(nx * 14).toFixed(2)}deg) rotateX(${(-ny * 12).toFixed(2)}deg) scale(1.05)`;
      frame.style.boxShadow = '0 60px 130px rgba(0,0,0,0.8)';
    });
    el.addEventListener('pointerleave', () => {
      frame.style.transform = 'perspective(900px) rotateY(0deg) rotateX(0deg) scale(1)';
      frame.style.boxShadow = '0 40px 90px rgba(0,0,0,0.6)';
    });
  });
}

/* ---------- journey route timeline ---------- */

const routeDone = new Set();

// A stage completes the moment its chapter sits centered in view (fully readable)
// and un-completes when the user scrolls back above that spot.
function updateRoute() {
  const vmid = window.innerHeight / 2 + 30;
  const centers = ['jch0', 'jch1', 'jch2'].map((id) => {
    const el = document.getElementById(id);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return r.top + r.height / 2;
  });
  if (centers.some((c) => c === null)) return;

  // dots pop exactly when their chapter is centered in view
  centers.forEach((c, i) => {
    if (c <= vmid) activateRouteStep(i);
    else deactivateRouteStep(i);
  });

  // lines draw continuously while travelling between chapters
  const route = document.getElementById('jroute');
  if (route) {
    const lines = route.querySelectorAll('[data-jline]');
    [0, 1].forEach((i) => {
      if (!lines[i]) return;
      const dist = centers[i + 1] - centers[i] || 1;
      const p = Math.min(1, Math.max(0, (vmid - centers[i]) / dist));
      lines[i].style.transform = 'scaleX(' + p.toFixed(4) + ')';
    });
  }

  // the whole timeline strip only appears once the KL stage completes
  const wrap = document.getElementById('jroute-wrap');
  if (wrap) {
    const on = routeDone.has(0);
    wrap.style.opacity = on ? '1' : '0';
    wrap.style.transform = on ? 'none' : 'translateY(16px)';
  }
}

function activateRouteStep(i) {
  if (routeDone.has(i)) return;
  routeDone.add(i);
  const route = document.getElementById('jroute');
  if (!route) return;
  const rootEl = document.getElementById('dt-root');
  const accent = (rootEl ? getComputedStyle(rootEl).getPropertyValue('--accent').trim() : '') || '#A31F34';
  const dot = route.querySelectorAll('[data-jdot]')[i];
  const label = route.querySelectorAll('[data-jlabel]')[i];
  if (dot) {
    dot.style.background = accent;
    dot.style.boxShadow = '0 0 0 5px color-mix(in oklab, ' + accent + ' 20%, transparent)';
    dot.style.transform = 'scale(1.15)';
  }
  if (label) label.style.opacity = '1';
}

function deactivateRouteStep(i) {
  if (!routeDone.has(i)) return;
  routeDone.delete(i);
  const route = document.getElementById('jroute');
  if (!route) return;
  const dot = route.querySelectorAll('[data-jdot]')[i];
  const label = route.querySelectorAll('[data-jlabel]')[i];
  if (dot) {
    dot.style.background = 'rgba(244,244,242,0.22)';
    dot.style.boxShadow = 'none';
    dot.style.transform = 'scale(1)';
  }
  if (label) label.style.opacity = '0.38';
}

/* ---------- scroll: progress bar, route, parallax ---------- */

let ticking = false;

function onScroll() {
  if (ticking) return;
  ticking = true;
  requestAnimationFrame(() => {
    ticking = false;
    const y = window.scrollY;
    const bar = document.getElementById('progress');
    if (bar) {
      const max = root.scrollHeight - window.innerHeight;
      bar.style.transform = `scaleX(${max > 0 ? Math.min(1, y / max) : 0})`;
    }
    updateRoute();
    if (!motionOn) return;
    const vh = window.innerHeight;
    document.querySelectorAll('[data-parallax]').forEach((el) => {
      const rect = el.parentElement.getBoundingClientRect();
      const center = rect.top + rect.height / 2 - vh / 2;
      el.style.transform = `translateY(${(-center * parseFloat(el.dataset.parallax)).toFixed(1)}px)`;
    });
  });
}

/* ---------- smooth wheel scrolling (Lenis-style) ---------- */

function initSmoothScroll() {
  if (!motionOn) return;
  if (!window.matchMedia || !window.matchMedia('(pointer: fine)').matches) return; // touch stays native
  let animating = false;
  let targetY = 0;
  let lastT = 0;
  let lastSet = -1;
  const step = (now) => {
    if (!animating) return;
    const cur = window.scrollY;
    // Something else moved the page (anchor jump, keyboard, scrollbar drag): yield to it.
    if (lastSet >= 0 && Math.abs(cur - lastSet) > 2) { animating = false; return; }
    const diff = targetY - cur;
    // < 1px: scroll offsets can round to whole pixels, so a sub-pixel threshold never settles.
    if (Math.abs(diff) < 1) {
      window.scrollTo({ top: targetY, behavior: 'instant' });
      animating = false;
      return;
    }
    const dtm = Math.min(0.05, (now - (lastT || now)) / 1000) || 0.016;
    lastT = now;
    lastSet = cur + diff * (1 - Math.exp(-dtm * 7));
    window.scrollTo({ top: lastSet, behavior: 'instant' });
    requestAnimationFrame(step);
  };
  window.addEventListener('wheel', (e) => {
    if (e.ctrlKey || e.metaKey || e.defaultPrevented) return;
    e.preventDefault();
    let d = e.deltaY;
    if (e.deltaMode === 1) d *= 16;
    else if (e.deltaMode === 2) d *= window.innerHeight;
    const max = root.scrollHeight - window.innerHeight;
    if (!animating) targetY = window.scrollY;
    targetY = Math.max(0, Math.min(max, targetY + d));
    if (!animating) {
      animating = true;
      lastSet = -1;
      lastT = performance.now();
      requestAnimationFrame(step);
    }
  }, { passive: false });
}

/* ---------- boot ---------- */

function init() {
  registerReveals();
  registerTilts();
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  window.addEventListener('load', onScroll, { once: true });
  onScroll();
  initSmoothScroll();
}

init();
