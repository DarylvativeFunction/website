// text.js - the single source of truth for where the name sits.
//
// Every prototype forms "Daryl Tan" out of particles / lines / voxels and then
// hands off to real HTML typography (reveal.js). For that crossfade to be
// invisible, the canvas raster the particles are sampled from and the HTML
// spans must agree to the pixel, so both are positioned from the same glyph
// layout computed here. Glyphs are placed one by one (prefix measuring keeps
// kerning) so tight tracking works without relying on ctx.letterSpacing.

export const NAME = { first: 'Daryl', middle: 'Yi Jin', last: 'Tan' };
export const FULL_NAME = 'Daryl Yi Jin Tan';
export const SUBTITLE_ITEMS = ['Petroleum Engineering', 'Economics', 'Machine Learning', 'Data Analytics'];
export const SUBTITLE = SUBTITLE_ITEMS.join(' · ');

export const FONT_FAMILY = "'Archivo', 'Helvetica Neue', Helvetica, Arial, sans-serif";
export const MONO_FAMILY = "'IBM Plex Mono', ui-monospace, Menlo, Consolas, monospace";
const TRACKING_EM = -0.028;

let fontsReady = null;
export function loadFonts() {
  if (!fontsReady) {
    fontsReady = Promise.all([
      document.fonts.load(`400 64px Archivo`),
      document.fonts.load(`600 64px Archivo`),
      document.fonts.load(`700 64px Archivo`),
      document.fonts.load(`800 64px Archivo`),
      document.fonts.load(`400 14px "IBM Plex Mono"`),
    ]).then(() => document.fonts.ready).catch(() => {});
  }
  return fontsReady;
}

const mctx = document.createElement('canvas').getContext('2d');

function glyphRun(word, fs, ls) {
  // x offset of every glyph from the word origin, kerning preserved.
  const xs = [];
  for (let i = 0; i < word.length; i++) {
    const withChar = mctx.measureText(word.slice(0, i + 1)).width;
    const charW = mctx.measureText(word[i]).width;
    xs.push(withChar - charW + i * ls);
  }
  const width = mctx.measureText(word).width + (word.length - 1) * ls;
  return { xs, width };
}

/**
 * Layout for a viewport of w x h CSS pixels.
 * Returns pixel positions (container space, origin top-left) for:
 *   words.daryl / words.tan  - x0: position in "Daryl Tan", x1: position in the full name
 *   words.yijin              - x1 only (it appears in place)
 *   baseline, capHeight, fontSize, subtitle block position.
 */
export function computeLayout(w, h, { weight = 700, scale = 1, centerY = 0.47 } = {}) {
  const probe = 100;
  mctx.font = `${weight} ${probe}px ${FONT_FAMILY}`;
  const lsP = TRACKING_EM * probe;
  const em = (s) => glyphRun(s, probe, lsP).width / probe;
  const space = mctx.measureText(' ').width / probe + TRACKING_EM;
  const eD = em(NAME.first), eY = em(NAME.middle), eT = em(NAME.last);
  const fullEm = eD + space + eY + space + eT;

  const narrow = w < 700;
  const maxW = w * (narrow ? 0.9 : 0.74);
  const fontSize = Math.max(18, Math.min(maxW / fullEm, h * 0.19, 176) * scale);
  const fs = fontSize;
  const ls = TRACKING_EM * fs;

  mctx.font = `${weight} ${fs}px ${FONT_FAMILY}`;
  const runD = glyphRun(NAME.first, fs, ls);
  const runY = glyphRun(NAME.middle, fs, ls);
  const runT = glyphRun(NAME.last, fs, ls);
  const sp = space * fs;

  const capM = mctx.measureText('DTY');
  const capHeight = capM.actualBoundingBoxAscent;
  const descM = mctx.measureText('y');
  const descent = descM.actualBoundingBoxDescent;

  const shortW = runD.width + sp + runT.width;
  const fullW = runD.width + sp + runY.width + sp + runT.width;
  const sx = (w - shortW) / 2;
  const fx = (w - fullW) / 2;

  const cy = h * centerY;
  const baseline = Math.round(cy + capHeight / 2);

  const subSize = Math.max(10, Math.min(narrow ? 11 : 13, fs * 0.13));
  const subTop = baseline + descent + Math.max(18, fs * 0.32);

  return {
    w, h, weight, fontSize: fs, tracking: ls, trackingEm: TRACKING_EM,
    font: `${weight} ${fs}px ${FONT_FAMILY}`,
    baseline, capHeight, descent,
    centerX: w / 2, centerY: baseline - capHeight / 2,
    words: {
      daryl: { text: NAME.first, xs: runD.xs, width: runD.width, x0: sx, x1: fx },
      yijin: { text: NAME.middle, xs: runY.xs, width: runY.width, x1: fx + runD.width + sp },
      tan: { text: NAME.last, xs: runT.xs, width: runT.width, x0: sx + runD.width + sp, x1: fx + runD.width + sp + runY.width + sp },
    },
    // bounding box of "Daryl Tan" in container px (cap top to descender)
    shortBox: { x: sx, y: baseline - capHeight, w: shortW, h: capHeight + descent },
    fullBox: { x: fx, y: baseline - capHeight, w: fullW, h: capHeight + descent },
    subtitle: { top: subTop, fontSize: subSize, narrow },
  };
}

function drawWord(ctx, word, x, baseline, k = 1) {
  for (let i = 0; i < word.text.length; i++) ctx.fillText(word.text[i], (x + word.xs[i]) * k, baseline * k);
}

/**
 * Rasterize the name into an alpha grid.
 *   which: 'short' ("Daryl Tan" at its formation position) | 'full' (final settled name)
 *   res:   raster pixels per CSS pixel (0.5 = half resolution)
 * Returns { data: Uint8ClampedArray alpha (0..255), cols, rows, res, wordOf(x,y) }.
 * Grid cell (i, j) covers container px [i/res, (i+1)/res).
 */
export function rasterizeName(layout, { which = 'short', res = 0.5, pad = 0 } = {}) {
  const cols = Math.max(1, Math.ceil(layout.w * res));
  const rows = Math.max(1, Math.ceil(layout.h * res));
  const c = document.createElement('canvas');
  c.width = cols; c.height = rows;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.font = `${layout.weight} ${layout.fontSize * res}px ${FONT_FAMILY}`;
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = '#fff';
  if (pad > 0) { ctx.strokeStyle = '#fff'; ctx.lineWidth = pad * 2 * res; ctx.lineJoin = 'round'; }
  const W = layout.words;
  const parts = which === 'short'
    ? [[W.daryl, W.daryl.x0], [W.tan, W.tan.x0]]
    : [[W.daryl, W.daryl.x1], [W.yijin, W.yijin.x1], [W.tan, W.tan.x1]];
  for (const [word, x] of parts) {
    drawWord(ctx, word, x, layout.baseline, res);
    if (pad > 0) for (let i = 0; i < word.text.length; i++) ctx.strokeText(word.text[i], (x + word.xs[i]) * res, layout.baseline * res);
  }
  const img = ctx.getImageData(0, 0, cols, rows).data;
  const data = new Uint8ClampedArray(cols * rows);
  for (let i = 0; i < data.length; i++) data[i] = img[i * 4 + 3];
  // word index by x (short layout: 0 = Daryl, 1 = Tan; full adds 2 = Yi Jin)
  const splitX = which === 'short' ? W.tan.x0 - 1 : W.yijin.x1 - 1;
  const splitX2 = W.tan.x1 - 1;
  const wordOf = (x) => which === 'short' ? (x < splitX ? 0 : 1) : (x < splitX ? 0 : x < splitX2 ? 2 : 1);
  return { data, cols, rows, res, wordOf };
}

/** Deterministic PRNG (mulberry32). */
export function rng(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Sample exactly `count` points inside the glyphs, in container px.
 *   mode: 'fill' (uniform over glyph area) | 'edge' (biased to outlines) | 'mix'
 * Returns { positions: Float32Array(count*2), word: Uint8Array(count), edge: Float32Array(count) (0 inside .. 1 on edge) }.
 */
export function sampleName(layout, count, { which = 'short', mode = 'fill', seed = 7, res } = {}) {
  const r = res ?? Math.min(1, Math.max(0.35, 70 / layout.fontSize));
  const ras = rasterizeName(layout, { which, res: r });
  const { data, cols, rows } = ras;
  const inside = [], edges = [];
  for (let y = 1; y < rows - 1; y++) {
    for (let x = 1; x < cols - 1; x++) {
      const i = y * cols + x;
      if (data[i] < 128) continue;
      const isEdge = data[i - 1] < 128 || data[i + 1] < 128 || data[i - cols] < 128 || data[i + cols] < 128;
      (isEdge ? edges : inside).push(i);
    }
  }
  const rand = rng(seed);
  const pool = mode === 'edge' ? edges : mode === 'fill' ? inside.concat(edges) : null;
  const positions = new Float32Array(count * 2);
  const word = new Uint8Array(count);
  const edge = new Float32Array(count);
  for (let k = 0; k < count; k++) {
    let src, isE;
    if (pool) { src = pool[Math.floor(rand() * pool.length)]; isE = mode === 'edge'; }
    else { isE = rand() < 0.45; const p = isE ? edges : inside; src = p[Math.floor(rand() * p.length)]; }
    if (src === undefined) src = 0;
    const gx = src % cols, gy = Math.floor(src / cols);
    const px = (gx + rand()) / r, py = (gy + rand()) / r;
    positions[k * 2] = px; positions[k * 2 + 1] = py;
    word[k] = ras.wordOf(px);
    edge[k] = isE ? 1 : 0;
  }
  return { positions, word, edge };
}

/**
 * Marching-squares iso-contours of a scalar grid (e.g. a raster from rasterizeName,
 * or a distance field you build from it). Returns polylines in grid units:
 * [[x0,y0,x1,y1,...], ...]. Divide by raster.res for container px.
 */
export function traceContours(field, cols, rows, iso) {
  const segs = [];
  const v = (x, y) => field[y * cols + x];
  const lerpE = (a, b) => (iso - a) / (b - a || 1e-9);
  for (let y = 0; y < rows - 1; y++) {
    for (let x = 0; x < cols - 1; x++) {
      const a = v(x, y), b = v(x + 1, y), c = v(x + 1, y + 1), d = v(x, y + 1);
      const idx = (a > iso ? 8 : 0) | (b > iso ? 4 : 0) | (c > iso ? 2 : 0) | (d > iso ? 1 : 0);
      if (idx === 0 || idx === 15) continue;
      const T = [x + lerpE(a, b), y], R = [x + 1, y + lerpE(b, c)];
      const B = [x + lerpE(d, c), y + 1], L = [x, y + lerpE(a, d)];
      switch (idx) {
        case 1: case 14: segs.push([L, B]); break;
        case 2: case 13: segs.push([B, R]); break;
        case 3: case 12: segs.push([L, R]); break;
        case 4: case 11: segs.push([T, R]); break;
        case 6: case 9: segs.push([T, B]); break;
        case 7: case 8: segs.push([L, T]); break;
        case 5: segs.push([L, T]); segs.push([B, R]); break;
        case 10: segs.push([T, R]); segs.push([L, B]); break;
      }
    }
  }
  // stitch segments into polylines
  const key = (p) => `${Math.round(p[0] * 1000)},${Math.round(p[1] * 1000)}`;
  const ends = new Map();
  segs.forEach((s, i) => {
    for (const p of s) { const k = key(p); if (!ends.has(k)) ends.set(k, []); ends.get(k).push(i); }
  });
  const used = new Uint8Array(segs.length);
  const lines = [];
  for (let i = 0; i < segs.length; i++) {
    if (used[i]) continue;
    used[i] = 1;
    const pts = [segs[i][0], segs[i][1]];
    for (const dir of [1, 0]) {
      for (;;) {
        const tip = dir ? pts[pts.length - 1] : pts[0];
        const cand = (ends.get(key(tip)) || []).find((j) => !used[j]);
        if (cand === undefined) break;
        used[cand] = 1;
        const s = segs[cand];
        const next = key(s[0]) === key(tip) ? s[1] : s[0];
        if (dir) pts.push(next); else pts.unshift(next);
      }
    }
    const flat = new Array(pts.length * 2);
    pts.forEach((p, k) => { flat[k * 2] = p[0]; flat[k * 2 + 1] = p[1]; });
    lines.push(flat);
  }
  return lines;
}

/**
 * Signed-ish distance field (in raster px) from a raster: positive inside the glyphs,
 * negative outside. Two-pass chamfer approximation, fast enough for ~1M cells.
 */
export function distanceField(raster, maxDist = 64) {
  const { data, cols, rows } = raster;
  const n = cols * rows;
  const INF = 1e9;
  const dIn = new Float32Array(n), dOut = new Float32Array(n);
  for (let i = 0; i < n; i++) { const ins = data[i] >= 128; dIn[i] = ins ? INF : 0; dOut[i] = ins ? 0 : INF; }
  const pass = (d) => {
    const A = 1, B = Math.SQRT2;
    for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
      const i = y * cols + x; let m = d[i];
      if (x > 0) m = Math.min(m, d[i - 1] + A);
      if (y > 0) { m = Math.min(m, d[i - cols] + A); if (x > 0) m = Math.min(m, d[i - cols - 1] + B); if (x < cols - 1) m = Math.min(m, d[i - cols + 1] + B); }
      d[i] = m;
    }
    for (let y = rows - 1; y >= 0; y--) for (let x = cols - 1; x >= 0; x--) {
      const i = y * cols + x; let m = d[i];
      if (x < cols - 1) m = Math.min(m, d[i + 1] + A);
      if (y < rows - 1) { m = Math.min(m, d[i + cols] + A); if (x < cols - 1) m = Math.min(m, d[i + cols + 1] + B); if (x > 0) m = Math.min(m, d[i + cols - 1] + B); }
      d[i] = m;
    }
  };
  pass(dIn); pass(dOut);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.max(-maxDist, Math.min(maxDist, dIn[i] > 0 ? dIn[i] : -dOut[i]));
  return out;
}
