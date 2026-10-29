// d-flowpath.js - Prototype D, "Flow-Path Reveal".
// A potential-flow reservoir (injectors, producers, regional drift) is traced once
// into long streamlines. Luminous comet trails ride them; the trails lengthen,
// converge and cross, and their polylines morph (on the GPU, from two path
// textures) onto the glyph outlines plus a fine interior hatch of "Daryl Tan".
// After the handoff they release back into faint streamlines.

import * as THREE from 'three';
import { EffectComposer, RenderPass, UnrealBloomPass, OutputPass } from 'three/addons';
import { createRenderer, pixelOrthoCamera, ramp, easeInOutCubic, easeInOutSine, clamp, rng } from '../core/stage.js';
import { rasterizeName, traceContours } from '../core/text.js';

const K = 40;            // vertices per comet trail
const KF = 96;           // vertices per static field line
const PS = 128;          // samples per streamline in the path texture
const PG = 64;           // samples per glyph path in the path texture

/* ------------------------------------------------------------------ streamlines */

function makeWells(w, h) {
  const portrait = h > w * 1.1;
  // (u along the flow axis, v across), 0..1
  const inj = [[0.04, 0.22], [0.04, 0.78], [0.3, 0.05], [0.3, 0.95], [0.06, 0.5]];
  const prod = [[0.96, 0.22], [0.96, 0.78], [0.7, 0.05], [0.7, 0.95], [0.94, 0.5]];
  const map = ([u, v]) => (portrait ? [v * w, u * h] : [u * w, v * h]);
  return { inj: inj.map(map), prod: prod.map(map), portrait };
}

function buildStreams(w, h, count, rand) {
  const { inj, prod, portrait } = makeWells(w, h);
  const wells = [...inj.map((p) => [p[0], p[1], 1]), ...prod.map((p) => [p[0], p[1], -1.15])];
  wells[4][2] = 0.22; wells[9][2] = -0.27;
  const Lref = Math.max(w, h);
  const U = portrait ? [0, 1.1 / Lref] : [1.1 / Lref, 0];
  const vel = (x, y, out) => {
    let vx = U[0], vy = U[1];
    for (let i = 0; i < wells.length; i++) {
      const dx = x - wells[i][0], dy = y - wells[i][1];
      const d2 = dx * dx + dy * dy + 30;
      vx += (wells[i][2] * dx) / d2; vy += (wells[i][2] * dy) / d2;
    }
    out[0] = vx; out[1] = vy;
  };
  const mx = w * 0.08, my = h * 0.08;
  const inDom = (x, y) => x > -mx && x < w + mx && y > -my && y < h + my;
  const near = (x, y) => {
    for (let i = 0; i < wells.length; i++) { const dx = x - wells[i][0], dy = y - wells[i][1]; if (dx * dx + dy * dy < 100) return true; }
    return false;
  };
  const ds = Math.max(4, Lref / 260);
  const v = [0, 0];
  const integrate = (x, y, dir) => {
    const pts = [];
    for (let n = 0; n < 1600; n++) {
      vel(x, y, v); let l = Math.hypot(v[0], v[1]) || 1;
      const mxp = x + (v[0] / l) * ds * 0.5 * dir, myp = y + (v[1] / l) * ds * 0.5 * dir;
      vel(mxp, myp, v); l = Math.hypot(v[0], v[1]) || 1;
      x += (v[0] / l) * ds * dir; y += (v[1] / l) * ds * dir;
      pts.push(x, y);
      if (!inDom(x, y) || near(x, y)) break;
    }
    return pts;
  };

  const lines = [];
  const minLen = Lref * 0.34;
  const tries = count * 5;
  // jittered grid of seeds
  const cols = Math.ceil(Math.sqrt(tries * (w / h))), rows = Math.ceil(tries / cols);
  const seeds = [];
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) seeds.push([((i + rand()) / cols) * (w * 1.1) - w * 0.05, ((j + rand()) / rows) * (h * 1.1) - h * 0.05]);
  for (let i = seeds.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [seeds[i], seeds[j]] = [seeds[j], seeds[i]]; }
  for (const [sx, sy] of seeds) {
    if (lines.length >= count) break;
    if (near(sx, sy)) continue;
    const back = integrate(sx, sy, -1), fwd = integrate(sx, sy, 1);
    const pts = [];
    for (let i = back.length - 2; i >= 0; i -= 2) pts.push(back[i], back[i + 1]);
    pts.push(sx, sy);
    for (let i = 0; i < fwd.length; i += 2) pts.push(fwd[i], fwd[i + 1]);
    // length
    let len = 0;
    for (let i = 2; i < pts.length; i += 2) len += Math.hypot(pts[i] - pts[i - 2], pts[i + 1] - pts[i - 1]);
    if (len < minLen) continue;
    lines.push({ pts, len });
  }
  return { lines, wells: { inj, prod } };
}

/** Resample a flat polyline to n points, evenly by arc length. */
function resample(pts, n, out, off) {
  const m = pts.length / 2;
  const cum = new Float32Array(m);
  for (let i = 1; i < m; i++) cum[i] = cum[i - 1] + Math.hypot(pts[i * 2] - pts[i * 2 - 2], pts[i * 2 + 1] - pts[i * 2 - 1]);
  const total = cum[m - 1] || 1e-6;
  let j = 0;
  for (let k = 0; k < n; k++) {
    const d = (k / (n - 1)) * total;
    while (j < m - 2 && cum[j + 1] < d) j++;
    const seg = cum[j + 1] - cum[j] || 1e-6;
    const f = clamp((d - cum[j]) / seg);
    out[off + k * 4] = pts[j * 2] + (pts[j * 2 + 2] - pts[j * 2]) * f;
    out[off + k * 4 + 1] = pts[j * 2 + 1] + (pts[j * 2 + 3] - pts[j * 2 + 1]) * f;
    out[off + k * 4 + 2] = 0; out[off + k * 4 + 3] = 1;
  }
  return total;
}

/* ------------------------------------------------------------------ glyph paths */

function smoothClosed(pts, passes) {
  const n = pts.length / 2;
  const closed = Math.hypot(pts[0] - pts[n * 2 - 2], pts[1] - pts[n * 2 - 1]) < 1e-3;
  let a = pts.slice();
  for (let p = 0; p < passes; p++) {
    const b = a.slice();
    for (let i = 0; i < n; i++) {
      const ip = i === 0 ? (closed ? n - 2 : 0) : i - 1;
      const inx = i === n - 1 ? (closed ? 1 : n - 1) : i + 1;
      b[i * 2] = (a[ip * 2] + 2 * a[i * 2] + a[inx * 2]) / 4;
      b[i * 2 + 1] = (a[ip * 2 + 1] + 2 * a[i * 2 + 1] + a[inx * 2 + 1]) / 4;
    }
    if (closed) { b[(n - 1) * 2] = b[0]; b[(n - 1) * 2 + 1] = b[1]; }
    a = b;
  }
  return a;
}

/** Sub-polyline between arc positions d0..d1 -> flat array of n points (appended to dst as a path). */
function subPath(pts, cum, d0, d1, n) {
  const out = new Array(n * 2);
  let j = 0;
  for (let k = 0; k < n; k++) {
    const d = d0 + ((d1 - d0) * k) / (n - 1);
    while (j < cum.length - 2 && cum[j + 1] < d) j++;
    const seg = cum[j + 1] - cum[j] || 1e-6;
    const f = clamp((d - cum[j]) / seg);
    out[k * 2] = pts[j * 2] + (pts[j * 2 + 2] - pts[j * 2]) * f;
    out[k * 2 + 1] = pts[j * 2 + 1] + (pts[j * 2 + 3] - pts[j * 2 + 1]) * f;
  }
  return out;
}

function buildGlyphPaths(layout, quality, rand) {
  const fs = layout.fontSize;
  const res = clamp(110 / fs, 1, 2.5);
  const ras = rasterizeName(layout, { which: 'short', res });
  const { data, cols, rows } = ras;
  const segLen = clamp(fs * 0.7, 34, 96);
  const paths = []; // { pts:flat(PG), kind:0 outline | 1 hatch }

  // outlines
  const contours = traceContours(data, cols, rows, 127.5);
  for (const c of contours) {
    if (c.length < 8) continue;
    const px = new Array(c.length);
    for (let i = 0; i < c.length; i += 2) { px[i] = (c[i] + 0.5) / res; px[i + 1] = (c[i + 1] + 0.5) / res; }
    const pts = smoothClosed(px, 2);
    const n = pts.length / 2;
    const cum = new Float32Array(n);
    for (let i = 1; i < n; i++) cum[i] = cum[i - 1] + Math.hypot(pts[i * 2] - pts[i * 2 - 2], pts[i * 2 + 1] - pts[i * 2 - 1]);
    const L = cum[n - 1];
    if (L < 10) continue;
    const segs = Math.max(1, Math.round(L / segLen));
    for (let s = 0; s < segs; s++) {
      const d0 = (s / segs) * L - 0.6, d1 = ((s + 1) / segs) * L + 0.6;
      paths.push({ pts: subPath(pts, cum, Math.max(0, d0), Math.min(L, d1), PG), kind: 0 });
    }
  }
  const nOutline = paths.length;

  // interior hatch
  const sp = clamp(fs * 0.021, 2.3, 3.4) * (quality === 'low' ? 1.9 : 1);
  const ang = (62 * Math.PI) / 180;
  const dx = Math.cos(ang), dy = Math.sin(ang), nx = -dy, ny = dx;
  const box = layout.shortBox;
  const corners = [[box.x - 3, box.y - 3], [box.x + box.w + 3, box.y - 3], [box.x - 3, box.y + box.h + 3], [box.x + box.w + 3, box.y + box.h + 3]];
  let cmin = 1e9, cmax = -1e9, tmin = 1e9, tmax = -1e9;
  for (const [x, y] of corners) {
    const c = x * nx + y * ny, t = x * dx + y * dy;
    cmin = Math.min(cmin, c); cmax = Math.max(cmax, c); tmin = Math.min(tmin, t); tmax = Math.max(tmax, t);
  }
  const inside = (x, y) => {
    const ix = Math.round(x * res - 0.5), iy = Math.round(y * res - 0.5);
    return ix >= 0 && iy >= 0 && ix < cols && iy < rows && data[iy * cols + ix] >= 128;
  };
  const step = 0.6, inset = 1.1;
  const maxHatch = segLen * 0.6;
  for (let c = cmin + rand() * sp; c < cmax; c += sp) {
    let runStart = -1;
    for (let t = tmin; t <= tmax + step; t += step) {
      const x = nx * c + dx * t, y = ny * c + dy * t;
      const ins = t <= tmax && inside(x, y);
      if (ins && runStart < 0) runStart = t;
      if (!ins && runStart >= 0) {
        const a = runStart + inset, b = t - step - inset;
        runStart = -1;
        if (b - a < 2.5) continue;
        const pieces = Math.ceil((b - a) / maxHatch);
        for (let p = 0; p < pieces; p++) {
          const t0 = a + ((b - a) * p) / pieces, t1 = a + ((b - a) * (p + 1)) / pieces;
          const pts = new Array(PG * 2);
          for (let k = 0; k < PG; k++) {
            const tt = t0 + ((t1 - t0) * k) / (PG - 1);
            pts[k * 2] = nx * c + dx * tt; pts[k * 2 + 1] = ny * c + dy * tt;
          }
          paths.push({ pts, kind: 1 });
        }
      }
    }
  }
  return { paths, nOutline, res };
}

/* ------------------------------------------------------------------ shaders */

const VERT = /* glsl */ `
  uniform sampler2D uS; uniform sampler2D uG;
  uniform float uPS, uPG, uPh, uMup, uMdn, uLen, uStreamA, uGlyphA, uPulse, uGlob, uW, uH, uBandY, uBandH, uBandAmt;
  attribute vec4 aA;   // u, rowS, rowG, seed
  attribute vec4 aB;   // offset, lenFrac, amp, cycles
  attribute vec4 aC;   // tint, win0, winSpan, streamAmp
  varying vec4 vCol;
  vec2 fetchP(sampler2D tex, float P, float row, float s) {
    float f = clamp(s, 0.0, 1.0) * (P - 1.0);
    float i0 = floor(f); float fr = f - i0;
    float i1 = min(i0 + 1.0, P - 1.0);
    vec2 a = texelFetch(tex, ivec2(int(i0), int(row)), 0).xy;
    vec2 b = texelFetch(tex, ivec2(int(i1), int(row)), 0).xy;
    return mix(a, b, fr);
  }
  float ease(float x) { x = clamp(x, 0.0, 1.0); return x * x * (3.0 - 2.0 * x); }
  void main() {
    float u = aA.x, seed = aA.w;
    float up = seed < 0.0 ? 0.0 : ease((uMup - seed * 0.62) / 0.38);
    float dn = seed < 0.0 ? 0.0 : ease((uMdn - (1.0 - seed) * 0.45) / 0.55);
    float m = up * (1.0 - dn);
    float settle = smoothstep(0.35, 0.95, m);
    float L0 = seed < 0.0 ? max(aB.y * 2.6, 0.4) : max(aB.y * 1.6, 0.25);
    float L = mix(mix(aB.y, L0, uLen), 1.0, m);
    float ph = fract(aB.w * uPh + aB.x);
    float head = mix(ph * (1.0 + L), 1.0, settle);
    float s = head - L * u;
    float sc = clamp(s, 0.0, 1.0);
    float mq = clamp(m * 1.7, 0.0, 1.0);
    float mv = smoothstep(0.0, 1.0, mq * 1.55 - 0.55 * (1.0 - sc));
    float sS = mix(s, aC.y + sc * aC.z, mv);
    vec2 Sp = fetchP(uS, uPS, aA.y, sS);
    vec2 Gp = fetchP(uG, uPG, aA.z, s);
    // sweep along a long arc that crosses the centre: control point offset towards the opposite side
    float side = sign(Gp.x - uW * 0.5 + 0.001);
    vec2 C = 0.5 * (Sp + Gp) + vec2(-side * uW * 0.2, (seed - 0.5) * uH * 0.55) * (seed < 0.0 ? 0.0 : 1.0);
    float q = 1.0 - mv;
    vec2 p = q * q * Sp + 2.0 * q * mv * C + mv * mv * Gp;
    float fall = mix(pow(1.0 - u, 1.65), 1.0, settle);
    float edge = smoothstep(0.0, 0.025, s) * (1.0 - smoothstep(0.975, 1.0, s));
    edge = mix(edge, 1.0, settle);
    float hd = smoothstep(0.1, 0.0, u) * (1.0 - smoothstep(0.85, 1.0, m));
    float amb = seed < 0.0 ? (1.0 - 0.95 * smoothstep(0.0, 0.8, uGlob)) : 1.0;
    float mA = smoothstep(0.0, 0.3, m);
    float A = mix(uStreamA * aC.w, uGlyphA * aB.z * uPulse * (1.0 + 0.5 * sin(3.14159 * clamp(m, 0.0, 1.0))), mA) * fall * edge * (1.0 + 1.6 * hd) * amb;
    A *= 1.0 - uBandAmt * (1.0 - uGlob) * (1.0 - smoothstep(uBandH, uBandH * 1.7, abs(p.y - uBandY)));
    vec3 cool = vec3(0.56, 0.84, 1.0), amber = vec3(1.0, 0.6, 0.22);
    vec3 col = mix(cool, amber, aC.x);
    col = mix(col, vec3(1.0), 0.55 * hd + 0.3 * settle * (1.0 - aC.x));
    vCol = vec4(col, A);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 0.0, 1.0);
  }`;

const FRAG = /* glsl */ `
  varying vec4 vCol;
  void main() { gl_FragColor = vCol; }`;

const FIELD_VERT = /* glsl */ `
  uniform sampler2D uS; uniform float uPS, uPh, uFieldA, uBandY, uBandH, uBandAmt, uGlob;
  attribute vec2 aF;   // u, rowS
  varying vec4 vCol;
  void main() {
    float u = aF.x;
    float f = u * (uPS - 1.0);
    float i0 = floor(f); float fr = f - i0; float i1 = min(i0 + 1.0, uPS - 1.0);
    vec2 a = texelFetch(uS, ivec2(int(i0), int(aF.y)), 0).xy;
    vec2 b = texelFetch(uS, ivec2(int(i1), int(aF.y)), 0).xy;
    vec2 p = mix(a, b, fr);
    float env = smoothstep(0.0, 0.08, u) * (1.0 - smoothstep(0.92, 1.0, u));
    float sh = 0.55 + 0.45 * sin(u * 31.0 + aF.y * 1.7 - 6.2831853 * 2.0 * uPh);
    vCol = vec4(0.5, 0.78, 1.0, uFieldA * env * sh * (1.0 - uBandAmt * (1.0 - smoothstep(uBandH, uBandH * 1.7, abs(p.y - uBandY)))));
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 0.0, 1.0);
  }`;

const RING_VERT = /* glsl */ `
  attribute vec3 color; uniform float uA; varying vec4 vCol;
  void main() { vCol = vec4(color, uA); gl_Position = projectionMatrix * modelViewMatrix * vec4(position.xy, 0.0, 1.0); }`;

const BG_FRAG = /* glsl */ `
  uniform vec2 uRes; varying vec2 vUv;
  void main() {
    vec2 q = (vUv - 0.5) * vec2(uRes.x / uRes.y, 1.0);
    float r = length(q);
    vec3 c = mix(vec3(0.0011, 0.0034, 0.0065), vec3(0.0001, 0.0003, 0.0010), smoothstep(0.0, 0.85, r));
    gl_FragColor = vec4(c, 1.0);
  }`;

/* ------------------------------------------------------------------ prototype */

export default {
  name: 'Flow-Path Reveal',
  blurb: 'Reservoir streamlines lengthen into luminous trails that converge, intersect and inscribe the name.',
  tech: 'Three.js · GPU path morph · potential flow · bloom',
  duration: 18,
  nameAt: 9.0,
  nameWeight: 700,
  phases: [
    { label: 'Streamlines', start: 0, end: 2.8 },
    { label: 'Lengthen', start: 2.8, end: 5.0 },
    { label: 'Converge', start: 5.0, end: 7.8 },
    { label: 'Inscribe', start: 7.8, end: 9.0 },
  ],
  reveal: { colors: { name: '#F3F8FB', yijin: 'rgba(214,232,244,0.66)', sub: 'rgba(210,228,240,0.72)', accent: '#E3A15B', rule: 'rgba(170,215,240,0.3)', glow: 'rgba(120,205,255,0.5)' } },
  posterTime: 9.0,

  create(ctx) {
    const low = ctx.quality === 'low';
    const renderer = createRenderer(ctx, { clearColor: 0x01030a });
    const scene = new THREE.Scene();
    let camera = pixelOrthoCamera(ctx.width, ctx.height);
    const T = ctx.timing;
    const dur = T.duration;

    let composer = null;
    if (!low) {
      const rt = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
      composer = new EffectComposer(renderer, rt);
      composer.addPass(new RenderPass(scene, camera));
      composer.addPass(new UnrealBloomPass(new THREE.Vector2(ctx.width, ctx.height), 0.32, 0.4, 0.08));
      composer.addPass(new OutputPass());
      composer.setPixelRatio(renderer.getPixelRatio());
      composer.setSize(ctx.width, ctx.height);
    }

    const uniforms = {
      uS: { value: null }, uG: { value: null }, uPS: { value: PS }, uPG: { value: PG },
      uPh: { value: 0 }, uMup: { value: 0 }, uMdn: { value: 0 }, uLen: { value: 0 },
      uStreamA: { value: 1 }, uGlyphA: { value: 1 }, uPulse: { value: 1 },
      uFieldA: { value: 0.1 }, uGlob: { value: 0 }, uW: { value: 1 }, uH: { value: 1 }, uBandY: { value: 0 }, uBandH: { value: 1 }, uBandAmt: { value: 0 },
    };
    const ringU = { uA: { value: 0.4 } };

    const trailMat = new THREE.ShaderMaterial({ uniforms, vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending });
    const fieldMat = new THREE.ShaderMaterial({ uniforms, vertexShader: FIELD_VERT, fragmentShader: FRAG, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending });
    const ringMat = new THREE.ShaderMaterial({ uniforms: ringU, vertexShader: RING_VERT, fragmentShader: FRAG, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending });
    const bgMat = new THREE.ShaderMaterial({
      uniforms: { uRes: { value: new THREE.Vector2(ctx.width, ctx.height) } },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
      fragmentShader: BG_FRAG, depthWrite: false, depthTest: false, side: THREE.DoubleSide,
    });
    const bg = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), bgMat);
    bg.renderOrder = -10; bg.frustumCulled = false;
    scene.add(bg);

    let texS = null, texG = null, trailGeo = null, fieldGeo = null, ringGeo = null;
    let trails = null, field = null, rings = null;
    let count = 0, nField = 0;

    function build() {
      for (const o of [trails, field, rings]) if (o) scene.remove(o);
      for (const o of [texS, texG, trailGeo, fieldGeo, ringGeo]) if (o) o.dispose();
      const w = ctx.width, h = ctx.height;
      const rand = rng(11);
      bg.scale.set(w, h, 1); bg.position.set(w / 2, h / 2, -500);
      bgMat.uniforms.uRes.value.set(w, h);

      // streamlines
      const dens = clamp((w * h) / (1440 * 900) * 1.7, 0.4, 1);
      const S = Math.round((low ? 56 : 150) * dens);
      const { lines, wells } = buildStreams(w, h, S, rand);
      const nS = lines.length;
      const sData = new Float32Array(nS * PS * 4);
      const sLen = new Float32Array(nS);
      for (let i = 0; i < nS; i++) sLen[i] = resample(lines[i].pts, PS, sData, i * PS * 4);
      texS = new THREE.DataTexture(sData, PS, nS, THREE.RGBAFormat, THREE.FloatType);
      texS.minFilter = texS.magFilter = THREE.NearestFilter; texS.needsUpdate = true;

      // glyph paths
      const glyph = buildGlyphPaths(ctx.layout, ctx.quality, rand);
      const gp = glyph.paths;
      count = gp.length;
      const gData = new Float32Array(count * PG * 4);
      const load = new Uint16Array(nS);
      uniforms.uW.value = w; uniforms.uH.value = h;
      { const L = ctx.layout; const top = L.shortBox.y - L.fontSize * 0.25, bot = L.subtitle.top + L.fontSize * 0.7; uniforms.uBandY.value = (top + bot) / 2; uniforms.uBandH.value = (bot - top) / 2; }
      const gcx = new Float32Array(count);
      const asg = new Int32Array(count), asgJ = new Int32Array(count), flip = new Uint8Array(count);
      for (let g = 0; g < count; g++) {
        const pts = gp[g].pts;
        let cx = 0, cy = 0;
        for (let k = 0; k < PG; k++) { cx += pts[k * 2]; cy += pts[k * 2 + 1]; }
        cx /= PG; cy /= PG;
        gcx[g] = clamp((cx - ctx.layout.shortBox.x) / ctx.layout.shortBox.w);
        let best = 1e18, bs = 0, bj = 0;
        for (let si = 0; si < nS; si++) {
          let d2 = 1e18, jj = 0;
          for (let j = 0; j < PS; j += 2) {
            const dx = sData[(si * PS + j) * 4] - cx, dy = sData[(si * PS + j) * 4 + 1] - cy;
            const dd = dx * dx + dy * dy;
            if (dd < d2) { d2 = dd; jj = j; }
          }
          const cost = Math.sqrt(d2) + load[si] * 14;
          if (cost < best) { best = cost; bs = si; bj = jj; }
        }
        load[bs]++; asg[g] = bs; asgJ[g] = bj;
        // orient glyph path with the local flow
        const j0 = Math.max(0, bj - 2), j1 = Math.min(PS - 1, bj + 2);
        const tx = sData[(bs * PS + j1) * 4] - sData[(bs * PS + j0) * 4], ty = sData[(bs * PS + j1) * 4 + 1] - sData[(bs * PS + j0) * 4 + 1];
        const gx = pts[(PG - 1) * 2] - pts[0], gy = pts[(PG - 1) * 2 + 1] - pts[1];
        flip[g] = gx * tx + gy * ty < 0 ? 1 : 0;
        for (let k = 0; k < PG; k++) {
          const kk = flip[g] ? PG - 1 - k : k;
          gData[(g * PG + k) * 4] = pts[kk * 2]; gData[(g * PG + k) * 4 + 1] = pts[kk * 2 + 1];
          gData[(g * PG + k) * 4 + 2] = 0; gData[(g * PG + k) * 4 + 3] = 1;
        }
      }
      texG = new THREE.DataTexture(gData, PG, count, THREE.RGBAFormat, THREE.FloatType);
      texG.minFilter = texG.magFilter = THREE.NearestFilter; texG.needsUpdate = true;
      uniforms.uS.value = texS; uniforms.uG.value = texG;

      // trail geometry
      const kv = low ? 28 : K;
      const nAmb = Math.round(nS * (low ? 2 : dens < 0.7 ? 2 : 3));
      const total = count + nAmb;
      const nv = total * kv;
      const aA = new Float32Array(nv * 4), aB = new Float32Array(nv * 4), aC = new Float32Array(nv * 4);
      const pos = new Float32Array(nv * 3);
      const idx = new Uint32Array(total * (kv - 1) * 2);
      const inStream = new Uint16Array(nS);
      for (let g = 0; g < total; g++) {
        const isAmb = g >= count;
        const sr = isAmb ? (g - count) % nS : asg[g];
        const outline = isAmb ? false : gp[g].kind === 0;
        const seed = isAmb ? -1 : clamp(gcx[g] * 0.8 + rand() * 0.2, 0, 1);
        const len = sLen[sr];
        const lenPx = 150 + rand() * 300;
        const lenFrac = Math.min(0.5, (isAmb ? lenPx * 1.3 : lenPx) / len);
        const cycles = clamp(Math.round((150 * dur) / (len * 1.3)), 1, 4);
        const off = (inStream[sr]++ * 0.37 + rand() * 0.25) % 1;
        const amp = outline ? 1.0 : 0.62;
        const tint = rand() < (outline ? 0.14 : 0.1) ? 1 : 0;
        const sAmp = isAmb ? 0.18 + Math.pow(rand(), 2.2) * 0.95 : 0.08 + rand() * 0.16;
        const span = 0.2;
        const win0 = isAmb ? 0 : clamp(asgJ[g] / (PS - 1) - span / 2, 0, 1 - span);
        for (let k = 0; k < kv; k++) {
          const v = g * kv + k;
          aA.set([k / (kv - 1), sr, isAmb ? 0 : g, seed], v * 4);
          aB.set([off, lenFrac, amp, cycles], v * 4);
          aC.set([tint, win0, span, sAmp], v * 4);
          if (k < kv - 1) { idx[(g * (kv - 1) + k) * 2] = v; idx[(g * (kv - 1) + k) * 2 + 1] = v + 1; }
        }
      }
      trailGeo = new THREE.BufferGeometry();
      trailGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      trailGeo.setAttribute('aA', new THREE.BufferAttribute(aA, 4));
      trailGeo.setAttribute('aB', new THREE.BufferAttribute(aB, 4));
      trailGeo.setAttribute('aC', new THREE.BufferAttribute(aC, 4));
      trailGeo.setIndex(new THREE.BufferAttribute(idx, 1));
      trails = new THREE.LineSegments(trailGeo, trailMat);
      trails.frustumCulled = false; trails.renderOrder = 2;
      scene.add(trails);

      // faint static field lines
      nField = nS;
      const kf = low ? 56 : KF;
      const fA = new Float32Array(nS * kf * 2), fPos = new Float32Array(nS * kf * 3);
      const fIdx = new Uint32Array(nS * (kf - 1) * 2);
      for (let si = 0; si < nS; si++) for (let k = 0; k < kf; k++) {
        const v = si * kf + k;
        fA[v * 2] = k / (kf - 1); fA[v * 2 + 1] = si;
        if (k < kf - 1) { fIdx[(si * (kf - 1) + k) * 2] = v; fIdx[(si * (kf - 1) + k) * 2 + 1] = v + 1; }
      }
      fieldGeo = new THREE.BufferGeometry();
      fieldGeo.setAttribute('position', new THREE.BufferAttribute(fPos, 3));
      fieldGeo.setAttribute('aF', new THREE.BufferAttribute(fA, 2));
      fieldGeo.setIndex(new THREE.BufferAttribute(fIdx, 1));
      field = new THREE.LineSegments(fieldGeo, fieldMat);
      field.frustumCulled = false; field.renderOrder = 1;
      scene.add(field);

      // wells: tiny rings with a centre tick
      const rp = [], rc = [];
      const seg = 28;
      const ringAt = (x, y, r, col) => {
        for (let i = 0; i < seg; i++) {
          const a0 = (i / seg) * Math.PI * 2, a1 = ((i + 1) / seg) * Math.PI * 2;
          rp.push(x + Math.cos(a0) * r, y + Math.sin(a0) * r, 0, x + Math.cos(a1) * r, y + Math.sin(a1) * r, 0);
          for (let q = 0; q < 2; q++) rc.push(col[0], col[1], col[2]);
        }
        for (const [ax, ay] of [[1, 0], [0, 1]]) {
          rp.push(x - ax * (r + 4), y - ay * (r + 4), 0, x - ax * (r + 1.5), y - ay * (r + 1.5), 0, x + ax * (r + 1.5), y + ay * (r + 1.5), 0, x + ax * (r + 4), y + ay * (r + 4), 0);
          for (let q = 0; q < 4; q++) rc.push(col[0], col[1], col[2]);
        }
      };
      const rr = clamp(Math.min(w, h) * 0.011, 4, 7);
      for (const p of wells.inj) ringAt(p[0], p[1], rr, [0.45, 0.78, 1.0]);
      for (const p of wells.prod) ringAt(p[0], p[1], rr * 0.8, [1.0, 0.62, 0.26]);
      ringGeo = new THREE.BufferGeometry();
      ringGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(rp), 3));
      ringGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(rc), 3));
      rings = new THREE.LineSegments(ringGeo, ringMat);
      rings.frustumCulled = false; rings.renderOrder = 3;
      scene.add(rings);
    }
    build();

    const N = T.nameAt;
    return {
      get stats() { return `${count} trails · ${count * (low ? 28 : K)} verts · ${nField} field lines`; },
      update(t) {
        const e = easeInOutCubic;
        const out0 = T.out[0];
        uniforms.uPh.value = t / dur;
        const lenAmt = e(ramp(t, 2.7, 5.2)) * (1 - e(ramp(t, T.out[1] - 0.3, dur - 0.1)));
        uniforms.uLen.value = lenAmt;
        uniforms.uMup.value = (() => { const x = ramp(t, 5.4, 8.75); return x * 0.8 + 0.2 * easeInOutSine(x); })();
        uniforms.uMdn.value = e(ramp(t, N + 0.5, N + 2.5));
        // glyph material fades while the HTML name takes over
        const glyphFade = 1 - e(ramp(t, N + 0.55, N + 1.2));
        uniforms.uGlyphA.value = (0.9 + 0.1 * ramp(t, 8.3, 9.0)) * glyphFade;
        uniforms.uPulse.value = 1 + 0.06 * Math.sin((t / dur) * Math.PI * 2 * 9);
        // stream brightness: full in the opening, dim for the read phase
        const dim = e(ramp(t, N + 0.7, N + 2.2));
        const back = e(ramp(t, T.out[1] - 0.3, dur - 0.15));
        const rd = 1 - 0.7 * dim * (1 - back);
        uniforms.uBandAmt.value = 0.75 * dim * (1 - back);
        const streamA = 0.44 * rd;
        uniforms.uStreamA.value = streamA;
        const glob = uniforms.uMup.value * (1 - uniforms.uMdn.value);
        uniforms.uGlob.value = glob;
        uniforms.uFieldA.value = 0.06 * (1 - 0.92 * glob) * rd;
        ringU.uA.value = 0.45 * (1 - 0.95 * glob) * rd;
        if (composer) composer.render(); else renderer.render(scene, camera);
      },
      resize(c) {
        ctx.width = c.width; ctx.height = c.height; ctx.layout = c.layout;
        renderer.setSize(c.width, c.height, false);
        camera = pixelOrthoCamera(c.width, c.height);
        if (composer) { composer.passes[0].camera = camera; composer.setSize(c.width, c.height); }
        build();
      },
      dispose() {
        for (const o of [texS, texG, trailGeo, fieldGeo, ringGeo, bg.geometry, trailMat, fieldMat, ringMat, bgMat]) if (o) o.dispose();
        if (composer) composer.dispose();
        renderer.dispose(); renderer.domElement.remove();
      },
    };
  },
};
