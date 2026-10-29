// f-graph.js - Prototype F, "Graph Learning".
// A reservoir cross-section drawn as a research-paper scatter (measurement points
// coloured by porosity) lifts into a latent-space embedding, grows a k-nearest-
// neighbour graph, passes messages along its edges, then re-wires itself so the
// nodes and short edges sit exactly on the glyphs of "Daryl Tan".
// Everything is a pure function of t: node/edge morphs happen in vertex shaders,
// the figure furniture (ticks, labels, HUD) is a 2D canvas redrawn from t.

import * as THREE from 'three';
import { createRenderer, pixelOrthoCamera, ramp, clamp, easeInOutCubic, dprFor, rng } from '../core/stage.js';
import { sampleName, MONO_FAMILY } from '../core/text.js';

const TAU = Math.PI * 2;
const K = 4;
const NCL = 7;
const ACCENT = [1.0, 0.33, 0.40];
const CAT = [
  [0.42, 0.62, 0.88], [0.34, 0.76, 0.72], [0.84, 0.72, 0.42], [0.66, 0.80, 0.94],
  [0.64, 0.56, 0.86], [0.56, 0.78, 0.48], [0.78, 0.80, 0.84],
];
const CLUSTER_C = [[-0.78, 0.30], [-0.50, -0.42], [-0.20, 0.38], [0.08, -0.30], [0.32, 0.42], [0.58, -0.22], [0.82, 0.28]];
const RAMP = [[0.22, 0.28, 0.48], [0.18, 0.48, 0.60], [0.52, 0.76, 0.68], [0.96, 0.90, 0.68]];

// ---------- geometry helpers -------------------------------------------------

function knnGrid(px, py, N, k) {
  let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
  for (let i = 0; i < N; i++) { minX = Math.min(minX, px[i]); maxX = Math.max(maxX, px[i]); minY = Math.min(minY, py[i]); maxY = Math.max(maxY, py[i]); }
  const cell = Math.max(1, Math.sqrt(((maxX - minX) * (maxY - minY)) / N) * 1.4);
  const cols = Math.ceil((maxX - minX) / cell) + 1, rows = Math.ceil((maxY - minY) / cell) + 1;
  const head = new Int32Array(cols * rows).fill(-1), next = new Int32Array(N);
  const cx = new Int32Array(N), cy = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    cx[i] = Math.floor((px[i] - minX) / cell); cy[i] = Math.floor((py[i] - minY) / cell);
    const c = cy[i] * cols + cx[i]; next[i] = head[c]; head[c] = i;
  }
  const nb = new Int32Array(N * k).fill(-1), nd = new Float32Array(N * k);
  const bd = new Float32Array(k), bi = new Int32Array(k);
  for (let i = 0; i < N; i++) {
    bd.fill(1e18); bi.fill(-1);
    for (let r = 0; r < 40; r++) {
      for (let y = cy[i] - r; y <= cy[i] + r; y++) {
        if (y < 0 || y >= rows) continue;
        const edgeRow = y === cy[i] - r || y === cy[i] + r;
        for (let x = cx[i] - r; x <= cx[i] + r; x += edgeRow ? 1 : Math.max(1, 2 * r)) {
          if (x < 0 || x >= cols) continue;
          for (let j = head[y * cols + x]; j >= 0; j = next[j]) {
            if (j === i) continue;
            const d = (px[j] - px[i]) ** 2 + (py[j] - py[i]) ** 2;
            if (d < bd[k - 1]) {
              let s = k - 1;
              while (s > 0 && bd[s - 1] > d) { bd[s] = bd[s - 1]; bi[s] = bi[s - 1]; s--; }
              bd[s] = d; bi[s] = j;
            }
          }
        }
      }
      if (bi[k - 1] >= 0 && Math.sqrt(bd[k - 1]) <= r * cell) break;
    }
    for (let s = 0; s < k; s++) { nb[i * k + s] = bi[s]; nd[i * k + s] = Math.sqrt(bd[s]); }
  }
  return { nb, nd };
}

// Undirected, de-duplicated edge list with a length cut, plus BFS depth from `seeds`.
function buildGraph(px, py, N, seeds, rand, lenMul) {
  const { nb, nd } = knnGrid(px, py, N, K);
  const kth = [];
  for (let i = 0; i < N; i++) kth.push(nd[i * K + 1]);
  kth.sort((a, b) => a - b);
  const maxLen = kth[Math.floor(kth.length * 0.5)] * lenMul;
  const seen = new Set(); const ea = [], eb = [];
  for (let i = 0; i < N; i++) for (let s = 0; s < K; s++) {
    const j = nb[i * K + s]; if (j < 0 || nd[i * K + s] > maxLen) continue;
    const a = Math.min(i, j), b = Math.max(i, j), key = a * N + b;
    if (seen.has(key)) continue; seen.add(key); ea.push(a); eb.push(b);
  }
  const E = ea.length, deg = new Int32Array(N + 1);
  for (let e = 0; e < E; e++) { deg[ea[e] + 1]++; deg[eb[e] + 1]++; }
  for (let i = 0; i < N; i++) deg[i + 1] += deg[i];
  const fill = deg.slice(0, N), adj = new Int32Array(E * 2);
  for (let e = 0; e < E; e++) { adj[fill[ea[e]]++] = eb[e]; adj[fill[eb[e]]++] = ea[e]; }
  const depth = new Float32Array(N).fill(-1); const q = new Int32Array(N); let qh = 0, qt = 0;
  for (const s of seeds) { depth[s] = 0; q[qt++] = s; }
  let maxD = 0;
  while (qh < qt) {
    const u = q[qh++];
    for (let p = deg[u]; p < deg[u + 1]; p++) { const v = adj[p]; if (depth[v] < 0) { depth[v] = depth[u] + 1; maxD = Math.max(maxD, depth[v]); q[qt++] = v; } }
  }
  for (let i = 0; i < N; i++) if (depth[i] < 0) depth[i] = Math.floor(rand() * (maxD + 1));
  return { ea, eb, E, depth, maxD, maxLen };
}

// ---------- shaders ------------------------------------------------------------

const COMMON = /* glsl */`
uniform float uSec, uConv, uCyc, uYaw; uniform vec2 uCenter;
const float TAU = 6.28318530718;
vec3 nodePos(vec2 s, vec2 tg, vec3 c, float seed, out float m1, out float m2) {
  float n = floor(1.0 + seed * 3.0);
  float ph = TAU * (n * uCyc + seed * 7.0);
  vec2 br = vec2(sin(ph), cos(ph + seed * 5.0));
  m1 = clamp(uSec * 1.45 - seed * 0.45, 0.0, 1.0); m1 = m1 * m1 * (3.0 - 2.0 * m1);
  m2 = clamp(uConv * 1.5 - fract(seed * 13.7) * 0.5, 0.0, 1.0); m2 = m2 * m2 * (3.0 - 2.0 * m2);
  float cy = cos(uYaw), sy = sin(uYaw);
  vec2 rel = c.xy - uCenter;
  vec3 r = vec3(rel.x * cy + c.z * sy, rel.y, -rel.x * sy + c.z * cy);
  float f = 1.0 + r.z * 0.0007;
  vec2 cp = uCenter + r.xy * f + br * 3.2;
  vec2 sp = s + br * 1.1;
  vec2 d = cp - sp; vec2 nrm = normalize(d + 1e-4);
  vec2 p1 = mix(sp, cp, m1) + vec2(-nrm.y, nrm.x) * sin(m1 * 3.14159) * (seed - 0.5) * 70.0;
  vec2 d2 = tg - p1; vec2 n2 = normalize(d2 + 1e-4);
  vec2 p = mix(p1, tg, m2) + vec2(-n2.y, n2.x) * sin(m2 * 3.14159) * (fract(seed * 5.3) - 0.5) * 50.0;
  return vec3(p, f);
}
uniform vec3 uCat[${NCL}];
vec3 rampCol(float v) {
  vec3 a = vec3(${RAMP[0].join(',')}), b = vec3(${RAMP[1].join(',')}), c = vec3(${RAMP[2].join(',')}), d = vec3(${RAMP[3].join(',')});
  float x = clamp(v, 0.0, 1.0) * 3.0;
  return x < 1.0 ? mix(a, b, x) : x < 2.0 ? mix(b, c, x - 1.0) : mix(c, d, x - 2.0);
}
`;

const NODE_VS = /* glsl */`
attribute vec2 aS; attribute vec2 aTg; attribute vec3 aC; attribute vec4 aMeta; attribute vec2 aW;
uniform float uSize, uDpr, uAlpha, uDensity, uTint, uWhite, uScanY, uScanW, uScanOn;
uniform float uPulseC, uPulseT, uGainC, uGainT, uWc, uWt;
varying vec3 vCol; varying float vA;
${COMMON}
void main() {
  float m1, m2;
  vec3 np = nodePos(aS, aTg, aC, aMeta.x, m1, m2);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(np.xy, 0.0, 1.0);
  float live = m1 * (1.0 - m2);
  float fc = exp(-pow((uPulseC - aW.x) / uWc, 2.0)) * uGainC;
  float ft = exp(-pow((uPulseT - aW.y) / uWt, 2.0)) * uGainT * m2;
  float fl = clamp(fc + ft, 0.0, 1.0);
  float sc = exp(-pow((np.y - uScanY) / uScanW, 2.0)) * uScanOn * (1.0 - m1);
  float thr = uDensity * 1.05;
  float vis = 1.0 - smoothstep(thr - 0.05, thr, aMeta.x);
  vec3 col = mix(rampCol(aMeta.y), uCat[int(aMeta.z + 0.5)], uTint * 0.9);
  col = mix(col, vec3(0.93, 0.95, 0.97), uWhite);
  col = mix(col, vec3(1.0), sc * 0.7);
  col = mix(col, vec3(${ACCENT.join(',')}), fl * 0.85);
  vCol = col;
  float dep = clamp(1.0 + (np.z - 1.0) * 7.0, 0.5, 1.3);
  vA = uAlpha * vis * mix(1.0, dep, live) * (1.0 + 0.5 * sc);
  gl_PointSize = uSize * uDpr * mix(1.0, np.z, live) * (1.0 + 0.9 * fl + 0.7 * sc) * (1.0 - 0.1 * m2);
}`;

const NODE_FS = /* glsl */`
varying vec3 vCol; varying float vA;
void main() {
  float d = length(gl_PointCoord - 0.5) * 2.0;
  float a = (1.0 - smoothstep(0.55, 1.0, d)) * vA;
  gl_FragColor = vec4(vCol, a);
}`;

const EDGE_VS = /* glsl */`
attribute vec2 aIdx; attribute float aEnd; attribute vec4 aE;
uniform sampler2D uNodes;
uniform float uGrow, uAlpha, uDensity, uLmax, uTint, uWhite;
varying vec3 vCol; varying float vG, vU, vD;
${COMMON}
vec3 endPos(int i, out vec3 col) {
  vec4 r0 = texelFetch(uNodes, ivec2(i, 0), 0);
  vec4 r1 = texelFetch(uNodes, ivec2(i, 1), 0);
  vec4 r2 = texelFetch(uNodes, ivec2(i, 2), 0);
  float m1, m2;
  vec3 p = nodePos(r0.xy, r0.zw, r1.xyz, r1.w, m1, m2);
  vec3 base = vec3(0.60, 0.70, 0.80);
  col = mix(base, uCat[int(r2.y + 0.5)], uTint * 0.6);
  col = mix(col, vec3(0.80, 0.86, 0.92), uWhite);
  return p;
}
void main() {
  vec3 ca, cb;
  vec3 pa = endPos(int(aIdx.x + 0.5), ca);
  vec3 pb = endPos(int(aIdx.y + 0.5), cb);
  float g = clamp((uGrow - aE.x * 0.62) / 0.38, 0.0, 1.0);
  g = g * g * (3.0 - 2.0 * g);
  vec2 pe = aEnd < 0.5 ? pa.xy : mix(pa.xy, pb.xy, g);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(pe, 0.0, 1.0);
  float len = length(pb.xy - pa.xy);
  float thr = uDensity * 1.05;
  float vis = 1.0 - smoothstep(thr - 0.05, thr, aE.z);
  vG = vis * (1.0 - smoothstep(uLmax, uLmax * 2.6, len)) * step(0.001, g);
  vCol = aEnd < 0.5 ? ca : cb;
  vU = aEnd * g; vD = aE.y;
}`;

const EDGE_FS = /* glsl */`
uniform float uAlpha, uPulse, uGain, uW, uPulseVis;
varying vec3 vCol; varying float vG, vU, vD;
void main() {
  float pulse = exp(-pow((uPulse - vD - vU) / uW, 2.0)) * uGain * vG * uPulseVis;
  float a0 = uAlpha * vG;
  float a = a0 + pulse;
  vec3 acc = vec3(${ACCENT.join(',')});
  gl_FragColor = vec4((vCol * a0 + acc * pulse) / max(a, 1e-4), a);
}`;

// ---------- figure furniture (2D canvas) ------------------------------------------

const lerp3 = (a, b, t) => a + (b - a) * t;
function rampJS(v) {
  const x = clamp(v) * 3, i = Math.min(2, Math.floor(x)), f = x - i;
  return RAMP[i].map((c, k) => lerp3(c, RAMP[i + 1][k], f));
}

export default {
  name: 'Graph Learning',
  blurb: 'A porosity scatter lifts into an embedding, grows a k-NN graph, passes messages, and re-wires into the name.',
  tech: 'Three.js · node + edge shaders · 3k nodes · 16k edges · kNN grid',
  duration: 17.2,
  nameAt: 8.2,
  nameWeight: 700,
  posterTime: 12.6,
  phases: [
    { label: 'Section', start: 0, end: 2.3 },
    { label: 'Embedding', start: 2.3, end: 4.3 },
    { label: 'k-NN graph', start: 4.3, end: 5.8 },
    { label: 'Messages', start: 5.8, end: 6.8 },
    { label: 'Converge', start: 6.8, end: 8.2 },
  ],
  reveal: { colors: { name: '#F2F3F4', yijin: 'rgba(210,222,235,0.6)', sub: 'rgba(210,222,235,0.68)', accent: '#C8364C', rule: 'rgba(200,215,230,0.26)', glow: 'rgba(150,190,230,0.4)' } },

  create(ctx) {
    const T = ctx.timing;
    const low = ctx.quality === 'low';
    const renderer = createRenderer(ctx, { clearColor: 0x090c11 });
    const scene = new THREE.Scene();
    let camera = pixelOrthoCamera(ctx.width, ctx.height);
    let N = 0;
    const dpr = renderer.getPixelRatio();

    // overlay DOM: vignette + 2D canvas
    const wrap = document.createElement('div');
    Object.assign(wrap.style, { position: 'absolute', inset: '0', zIndex: '1', pointerEvents: 'none',
      background: 'radial-gradient(ellipse 75% 70% at 50% 46%, rgba(0,0,0,0) 40%, rgba(2,3,6,0.62) 100%)' });
    const ov = document.createElement('canvas');
    Object.assign(ov.style, { position: 'absolute', inset: '0', width: '100%', height: '100%' });
    ov.setAttribute('aria-hidden', 'true');
    wrap.appendChild(ov);
    ctx.container.appendChild(wrap);
    const o2 = ov.getContext('2d');
    const odpr = Math.min(window.devicePixelRatio || 1, 2);

    const shared = {
      uSec: { value: 0 }, uConv: { value: 0 }, uCyc: { value: 0 }, uYaw: { value: 0 },
      uCenter: { value: new THREE.Vector2() },
      uCat: { value: CAT.map((c) => new THREE.Vector3(...c)) },
      uTint: { value: 0 }, uWhite: { value: 0 }, uDensity: { value: 1 },
    };

    let geo = null, nodeMat = null, nodePts = null, edgeSets = [], nodeTex = null;
    let G = null; // geometry of the figure
    let maxDC = 1;

    function build() {
      for (const e of edgeSets) { e.geo.dispose(); e.mat.dispose(); scene.remove(e.mesh); }
      edgeSets = [];
      if (geo) { geo.dispose(); nodeMat.dispose(); scene.remove(nodePts); }
      nodeTex?.dispose();

      const w = ctx.width, h = ctx.height, narrow = w < 700;
      N = low ? 1100 : narrow ? 1900 : 3000;
      const cy0 = ctx.layout.centerY;
      const sec = { x0: w * (narrow ? 0.07 : 0.1), x1: w * (narrow ? 0.93 : 0.9) };
      const sH = narrow ? h * 0.30 : h * 0.40;
      sec.y0 = cy0 - sH / 2; sec.y1 = cy0 + sH / 2;
      const emb = { cx: w / 2, cy: cy0, sx: w * (narrow ? 0.38 : 0.35), sy: h * (narrow ? 0.19 : 0.26) };
      G = { w, h, narrow, sec, emb };

      const r = rng(11);
      const gauss = () => { let s = 0; for (let i = 0; i < 4; i++) s += r(); return (s - 2) * 1.73; };
      // layered porosity model
      const L = 16, base = [];
      for (let l = 0; l < L; l++) base.push(clamp(0.5 + 0.22 * Math.sin(l * 0.85 + 1.0) + (r() - 0.5) * 0.4 + ((l === 5 || l === 11) ? 0.22 : 0), 0.04, 1));
      const S = new Float32Array(N * 2), C = new Float32Array(N * 3), Tg = new Float32Array(N * 2), Meta = new Float32Array(N * 4), W = new Float32Array(N * 2);
      const prop = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        let u, d, f, tries = 0;
        do { u = r(); d = r(); f = (d * L) % 1; tries++; } while ((f < 0.07 || f > 0.93) && tries < 8);
        const fold = 0.07 * Math.sin(u * 5.5 + d * 3.0) + 0.06 * (u - 0.5);
        const yy = sec.y0 + (d + fold * 0.9) * (sec.y1 - sec.y0);
        S[i * 2] = sec.x0 + u * (sec.x1 - sec.x0);
        S[i * 2 + 1] = yy;
        const l = Math.min(L - 1, Math.floor(d * L));
        prop[i] = clamp(base[l] + 0.18 * (u - 0.5) * (1 - d) + gauss() * 0.045, 0, 1);
        Meta[i * 4] = r();
        Meta[i * 4 + 1] = prop[i];
      }
      // clusters = equal-count porosity classes (latent space groups similar rock)
      const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => prop[a] - prop[b]);
      const cl = new Int32Array(N);
      order.forEach((idx, rank) => { cl[idx] = Math.min(NCL - 1, Math.floor((rank / N) * NCL)); Meta[idx * 4 + 2] = cl[idx]; });
      const ang = [0.4, -0.6, 0.9, -0.2, 0.7, -0.9, 0.1];
      for (let i = 0; i < N; i++) {
        const k = cl[i], wide = r() < 0.06 ? 2.4 : 1;
        const a = gauss() * 0.13 * wide, b = gauss() * 0.07 * wide;
        const ca = Math.cos(ang[k]), sa = Math.sin(ang[k]);
        const nx = CLUSTER_C[k][0] + a * ca - b * sa, ny = CLUSTER_C[k][1] + a * sa + b * ca;
        C[i * 3] = emb.cx + nx * emb.sx; C[i * 3 + 1] = emb.cy + ny * emb.sy; C[i * 3 + 2] = gauss() * 120 + (k - 3) * 28;
      }
      // targets: x-sorted slabs matched to clusters in x-order of their centres
      const tgt = sampleName(ctx.layout, N, { mode: 'mix', seed: 7 }).positions;
      const tix = Array.from({ length: N }, (_, i) => i).sort((a, b) => tgt[a * 2] - tgt[b * 2]);
      const corder = Array.from({ length: NCL }, (_, k) => k).sort((a, b) => CLUSTER_C[a][0] - CLUSTER_C[b][0]);
      const members = Array.from({ length: NCL }, () => []);
      for (let i = 0; i < N; i++) members[cl[i]].push(i);
      let cursor = 0;
      for (const k of corder) {
        const slab = tix.slice(cursor, cursor + members[k].length); cursor += members[k].length;
        for (let s = slab.length - 1; s > 0; s--) { const j = Math.floor(r() * (s + 1)); [slab[s], slab[j]] = [slab[j], slab[s]]; }
        members[k].forEach((node, s) => { Tg[node * 2] = tgt[slab[s] * 2]; Tg[node * 2 + 1] = tgt[slab[s] * 2 + 1]; });
      }

      // graphs
      const cx = new Float32Array(N), cyv = new Float32Array(N), tx = new Float32Array(N), ty = new Float32Array(N);
      for (let i = 0; i < N; i++) { cx[i] = C[i * 3]; cyv[i] = C[i * 3 + 1]; tx[i] = Tg[i * 2]; ty[i] = Tg[i * 2 + 1]; }
      // seeds: node closest to each cluster centre; leftmost nodes for the name wave
      const seedsC = [];
      for (let k = 0; k < NCL; k++) {
        let best = -1, bd = 1e18;
        for (const i of members[k]) { const d = (cx[i] - (emb.cx + CLUSTER_C[k][0] * emb.sx)) ** 2 + (cyv[i] - (emb.cy + CLUSTER_C[k][1] * emb.sy)) ** 2; if (d < bd) { bd = d; best = i; } }
        seedsC.push(best);
      }
      const minTx = Math.min(...tx);
      const seedsT = []; for (let i = 0; i < N; i++) if (tx[i] < minTx + 6) seedsT.push(i);
      const gC = buildGraph(cx, cyv, N, seedsC, r, 2.4);
      const gT = buildGraph(tx, ty, N, seedsT, r, 2.3);
      maxDC = gC.maxD; G.maxDT = gT.maxD; G.gC = gC; G.gT = gT;
      for (let i = 0; i < N; i++) { W[i * 2] = gC.depth[i]; W[i * 2 + 1] = gT.depth[i]; }

      // node texture for edges
      const td = new Float32Array(N * 12);
      for (let i = 0; i < N; i++) {
        td.set([S[i * 2], S[i * 2 + 1], Tg[i * 2], Tg[i * 2 + 1]], i * 4);
        td.set([C[i * 3], C[i * 3 + 1], C[i * 3 + 2], Meta[i * 4]], N * 4 + i * 4);
        td.set([Meta[i * 4 + 1], Meta[i * 4 + 2], 0, 0], N * 8 + i * 4);
      }
      nodeTex = new THREE.DataTexture(td, N, 3, THREE.RGBAFormat, THREE.FloatType);
      nodeTex.minFilter = nodeTex.magFilter = THREE.NearestFilter; nodeTex.needsUpdate = true;

      // nodes
      geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
      geo.setAttribute('aS', new THREE.BufferAttribute(S, 2));
      geo.setAttribute('aTg', new THREE.BufferAttribute(Tg, 2));
      geo.setAttribute('aC', new THREE.BufferAttribute(C, 3));
      geo.setAttribute('aMeta', new THREE.BufferAttribute(Meta, 4));
      geo.setAttribute('aW', new THREE.BufferAttribute(W, 2));
      geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);
      nodeMat = new THREE.ShaderMaterial({
        transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
        uniforms: {
          ...shared,
          uSize: { value: narrow ? 1.7 : 2.1 }, uDpr: { value: dpr }, uAlpha: { value: 1 },
          uScanY: { value: 0 }, uScanW: { value: sH * 0.02 + 3 }, uScanOn: { value: 0 },
          uPulseC: { value: -99 }, uPulseT: { value: -99 }, uGainC: { value: 0 }, uGainT: { value: 0 },
          uWc: { value: 0.7 }, uWt: { value: 14 },
        },
        vertexShader: NODE_VS, fragmentShader: NODE_FS,
      });
      nodePts = new THREE.Points(geo, nodeMat);
      nodePts.frustumCulled = false; nodePts.renderOrder = 2;
      scene.add(nodePts);

      // edges
      const mkEdges = (g, depthSel, lmax, pulseW, order) => {
        const E = g.E, idx = new Float32Array(E * 4), end = new Float32Array(E * 2), ee = new Float32Array(E * 8);
        const pos = new Float32Array(E * 2 * 3);
        for (let e = 0; e < E; e++) {
          let a = g.ea[e], b = g.eb[e];
          if (g.depth[b] < g.depth[a]) { const t = a; a = b; b = t; }
          const del = r(), vs = Math.max(Meta[a * 4], Meta[b * 4]);
          for (let v = 0; v < 2; v++) {
            idx[e * 4 + v * 2] = a; idx[e * 4 + v * 2 + 1] = b; end[e * 2 + v] = v;
            ee.set([del, g.depth[a], vs, 0], e * 8 + v * 4);
          }
        }
        const eg = new THREE.BufferGeometry();
        eg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        eg.setAttribute('aIdx', new THREE.BufferAttribute(idx, 2));
        eg.setAttribute('aEnd', new THREE.BufferAttribute(end, 1));
        eg.setAttribute('aE', new THREE.BufferAttribute(ee, 4));
        eg.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);
        const mat = new THREE.ShaderMaterial({
          transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
          uniforms: {
            ...shared, uNodes: { value: nodeTex },
            uGrow: { value: 0 }, uAlpha: { value: 0 }, uLmax: { value: lmax }, uPulse: { value: -99 }, uGain: { value: 0 }, uW: { value: pulseW }, uPulseVis: { value: 1 },
          },
          vertexShader: EDGE_VS, fragmentShader: EDGE_FS,
        });
        const mesh = new THREE.LineSegments(eg, mat);
        mesh.frustumCulled = false; mesh.renderOrder = order;
        scene.add(mesh);
        return { geo: eg, mat, mesh, E };
      };
      edgeSets = [mkEdges(gC, 0, gC.maxLen * 1.2, 0.5, 0), mkEdges(gT, 1, gT.maxLen * 1.3, 12, 1)];
      shared.uCenter.value.set(emb.cx, emb.cy);

      ov.width = Math.round(w * odpr); ov.height = Math.round(h * odpr);
    }
    build();

    // ---------- 2D figure furniture -------------------------------------------
    const A = (a) => Math.max(0, Math.min(1, a));
    function drawFigure(t, ovl, secP, convP) {
      const { w, h, narrow, sec, emb } = G;
      o2.setTransform(odpr, 0, 0, odpr, 0, 0);
      o2.clearRect(0, 0, w, h);
      if (ovl <= 0.002) return;
      const fs = narrow ? 8.5 : 9.5;
      o2.font = `${fs}px ${MONO_FAMILY}`;
      o2.textBaseline = 'alphabetic';
      const ink = (a) => `rgba(190,206,222,${A(a) * ovl})`;
      o2.lineWidth = 1;

      const inSec = 1 - smooth(secP, 0.0, 0.6);              // section furniture
      const inEmb = smooth(secP, 0.45, 1.0) * (1 - smooth(convP, 0.0, 0.35)); // embedding furniture

      if (inSec > 0.01) {
        const x0 = sec.x0 - 10, x1 = sec.x1 + 10, y0 = sec.y0 - 8, y1 = sec.y1 + 8;
        o2.strokeStyle = ink(0.30 * inSec); o2.fillStyle = ink(0.42 * inSec);
        o2.beginPath();
        o2.moveTo(x0, y0); o2.lineTo(x0, y1); o2.lineTo(x1, y1);
        o2.stroke();
        o2.beginPath();
        const ny = narrow ? 4 : 6;
        for (let i = 0; i <= ny; i++) { const y = y0 + (y1 - y0) * i / ny; o2.moveTo(x0, y); o2.lineTo(x0 - 4, y); }
        const nx = narrow ? 4 : 12;
        for (let i = 0; i <= nx; i++) { const x = x0 + (x1 - x0) * i / nx; o2.moveTo(x, y1); o2.lineTo(x, y1 + 4); }
        o2.stroke();
        o2.textAlign = 'right';
        for (let i = 0; i <= ny; i += narrow ? 2 : 1) { const y = y0 + (y1 - y0) * i / ny; o2.fillText(String(2140 + i * (narrow ? 20 : 15)), x0 - 8, y + 3); }
        o2.textAlign = 'center';
        for (let i = 0; i <= nx; i += narrow ? 2 : 2) { const x = x0 + (x1 - x0) * i / nx; o2.fillText(String(i * (narrow ? 300 : 100)), x, y1 + 6 + fs + 1); }
        o2.textAlign = 'left';
        o2.fillText('depth (m)', x0 - (narrow ? 4 : 2), y0 - 8 - (narrow ? 14 : 0));
        o2.textAlign = 'right';
        o2.fillText('x (m)', x1, y1 + 6 + fs * 2 + 6);
        // colour key for porosity
        const bw = narrow ? 70 : 96, bx = narrow ? x0 + 30 : x1 - bw, by = narrow ? y1 + 6 + fs * 2 + 22 : y0 - 18;
        for (let i = 0; i < bw; i += 2) { const c = rampJS(i / bw); o2.fillStyle = `rgba(${c[0] * 255 | 0},${c[1] * 255 | 0},${c[2] * 255 | 0},${0.8 * inSec * ovl})`; o2.fillRect(bx + i, by, 2, 3); }
        o2.fillStyle = ink(0.42 * inSec); o2.textAlign = 'right'; o2.fillText('φ 0.05', bx - 6, by + 4);
        o2.textAlign = 'left'; o2.fillText('0.30', bx + bw + 6, by + 4);
      }
      if (inEmb > 0.01) {
        const x0 = emb.cx - emb.sx * 1.12, x1 = emb.cx + emb.sx * 1.12, y0 = emb.cy - emb.sy * 1.2, y1 = emb.cy + emb.sy * 1.2;
        o2.strokeStyle = ink(0.26 * inEmb); o2.fillStyle = ink(0.4 * inEmb);
        o2.beginPath(); o2.moveTo(x0, y0); o2.lineTo(x0, y1); o2.lineTo(x1, y1); o2.stroke();
        o2.beginPath();
        for (let i = 0; i <= 8; i++) { const x = x0 + (x1 - x0) * i / 8; o2.moveTo(x, y1); o2.lineTo(x, y1 + (i % 2 ? 2 : 4)); }
        for (let i = 0; i <= 6; i++) { const y = y0 + (y1 - y0) * i / 6; o2.moveTo(x0, y); o2.lineTo(x0 - (i % 2 ? 2 : 4), y); }
        o2.stroke();
        o2.textAlign = 'right'; o2.fillText('z₂', x0 - 8, y0 + 8);
        o2.fillText('z₁', x1, y1 + 6 + fs + 1);
        o2.textAlign = 'left';
        o2.fillText(narrow ? '' : 'n = ' + N.toLocaleString('en-US') + ' · 7 clusters', x0, y1 + 6 + fs + 1);
      }
      // figure caption
      const cap = (s, a) => { if (a > 0.01) { o2.fillStyle = ink(0.5 * a); o2.textAlign = 'left'; o2.fillText(s, sec.x0 - 10, (narrow ? G.sec.y0 - 36 : sec.y0 - 18)); } };
      const capY = narrow ? Math.max(18, sec.y0 - 36) : Math.max(22, emb.cy - emb.sy * 1.2 - 20);
      const capX = narrow ? sec.x0 - 10 : emb.cx - emb.sx * 1.12;
      const caption = (s, a) => { if (a > 0.01) { o2.fillStyle = ink(0.5 * a); o2.textAlign = 'left'; o2.fillText(s, capX, capY); } };
      void cap;
      const c1 = 1 - smooth(t, 2.4, 3.2), c2 = smooth(t, 2.8, 3.6) * (1 - smooth(t, 4.6, 5.2));
      const c3 = smooth(t, 4.8, 5.6);
      caption(narrow ? 'fig. 1  porosity φ, section A–A′' : 'fig. 1  well-log porosity φ, section A–A′', c1);
      caption(narrow ? 'fig. 2  latent embedding' : 'fig. 2  latent embedding of log features, 2-d projection', c2);
      caption(narrow ? 'fig. 3  k-NN graph, k = 4' : 'fig. 3  k-NN graph, k = 4  ·  message passing', c3);

      // HUD
      const hu = smooth(t, 4.4, 5.2) * (1 - smooth(t, T.nameAt - 1.4, T.nameAt - 0.5));
      if (hu > 0.01) {
        const ep = Math.floor(48 * ramp(t, 4.4, 7.4)), loss = (e) => 0.021 + 0.68 * Math.exp(-e / 10);
        const hx = narrow ? 16 : 28, hy = h - (narrow ? 26 : 30);
        o2.fillStyle = ink(0.42 * hu); o2.textAlign = 'left';
        o2.fillText(`k = ${K} · epoch ${String(ep).padStart(3, '0')} · loss ${loss(ep).toFixed(3)}`, hx, hy);
        const sw = 64, sh = 12, sx0 = hx + (narrow ? 175 : 205), sy0 = hy - 9;
        o2.strokeStyle = ink(0.34 * hu); o2.beginPath();
        for (let e = 0; e <= ep; e++) { const x = sx0 + (e / 48) * sw, y = sy0 + sh - (loss(e) / 0.7) * sh; e ? o2.lineTo(x, y) : o2.moveTo(x, y); }
        o2.stroke();
        o2.fillStyle = `rgba(255,84,102,${0.8 * hu * ovl})`;
        o2.fillRect(sx0 + (ep / 48) * sw - 1, sy0 + sh - (loss(ep) / 0.7) * sh - 1, 2.4, 2.4);
      }
    }
    function smooth(x, a, b) { const k = clamp((x - a) / (b - a)); return k * k * (3 - 2 * k); }

    // ---------- timeline ---------------------------------------------------------
    const na = T.nameAt, D = T.duration, OUT = T.out[0];
    const stats = `${N.toLocaleString()} nodes · ${(edgeSets[0].E + edgeSets[1].E).toLocaleString()} edges`;

    return {
      stats,
      update(t) {
        const nu = nodeMat.uniforms;
        const secUp = easeInOutCubic(ramp(t, 2.3, 4.5)), secDn = easeInOutCubic(ramp(t, OUT + 0.15, D - 0.3));
        const secP = secUp * (1 - secDn);
        const convUp = easeInOutCubic(ramp(t, 6.4, na)), convDn = easeInOutCubic(ramp(t, na + 0.6, na + 2.7));
        const convP = convUp * (1 - convDn);
        const readEnv = smooth(t, na + 0.4, na + 1.4) * (1 - smooth(t, OUT + 0.2, D - 0.3));
        const tint = smooth(t, 4.6, 6.0) * (1 - smooth(t, OUT, OUT + 1.2));

        shared.uSec.value = secP; shared.uConv.value = convP;
        shared.uCyc.value = t / D;
        shared.uYaw.value = 0.5 * Math.sin(TAU * t / D) * (1 - 0.0);
        shared.uTint.value = tint;
        shared.uWhite.value = smooth(convP, 0.55, 1.0);
        shared.uDensity.value = 1 - 0.62 * readEnv;

        nu.uAlpha.value = 0.95 - 0.62 * readEnv;
        // scan cursor through the section at the start
        const sc = ramp(t, 0.35, 2.15);
        nu.uScanOn.value = Math.sin(Math.PI * sc) * (sc > 0 && sc < 1 ? 1 : 0) * 0.9;
        nu.uScanY.value = G.sec.y0 + (G.sec.y1 - G.sec.y0) * easeInOutCubic(sc);
        // message passing: cloud waves (period in hops)
        const rateC = (maxDC + 1) / 1.6, perC = maxDC + 5;
        const tpC = t - 4.9;
        nu.uPulseC.value = tpC > 0 ? (tpC * rateC) % perC : -99;
        const gC = smooth(t, 4.9, 5.4) * (1 - smooth(convP, 0.0, 0.5)) + 0.35 * readEnv * (1 - smooth(t, OUT, OUT + 0.8));
        nu.uGainC.value = gC;
        // final cascade across the letters
        const rateT = (G.maxDT + 1) / 1.5;
        const tpT = t - (na - 1.0);
        nu.uPulseT.value = tpT > 0 && tpT < 2.2 ? tpT * rateT - 10 : -99;
        nu.uGainT.value = smooth(t, na - 1.0, na - 0.6) * (1 - smooth(t, na + 0.5, na + 1.1)) * 0.9;

        const [ec, et] = edgeSets;
        ec.mat.uniforms.uGrow.value = ramp(t, 3.9, 5.6);
        ec.mat.uniforms.uAlpha.value = (G.narrow ? 0.2 : 0.17) * smooth(t, 3.9, 4.4) * (1 - smooth(convP, 0.0, 0.4)) * (1 - 0.45 * readEnv) * (1 - smooth(t, OUT, OUT + 1.1));
        ec.mat.uniforms.uPulse.value = nu.uPulseC.value;
        ec.mat.uniforms.uGain.value = gC;
        ec.mat.uniforms.uPulseVis.value = 4.5;
        ec.mesh.visible = ec.mat.uniforms.uAlpha.value > 0.001 || gC > 0.001;

        et.mat.uniforms.uGrow.value = ramp(t, 6.7, na - 0.2);
        const typeFade = 1 - smooth(t, na + 0.35, na + 1.0);
        et.mat.uniforms.uAlpha.value = 0.42 * smooth(convP, 0.5, 0.95) * typeFade;
        et.mat.uniforms.uPulse.value = nu.uPulseT.value;
        et.mat.uniforms.uGain.value = nu.uGainT.value;
        et.mat.uniforms.uPulseVis.value = 1.4;
        et.mesh.visible = et.mat.uniforms.uAlpha.value > 0.001 || nu.uGainT.value > 0.001;
        // edge sets share cluster tint/white uniforms through `shared`

        drawFigure(t, t < 12 ? 1 - smooth(t, na - 1.3, na - 0.4) : smooth(t, OUT + 0.5, D - 0.25), secP, convP);
        renderer.render(scene, camera);
      },
      resize(c) {
        renderer.setSize(c.width, c.height, false);
        camera = pixelOrthoCamera(c.width, c.height);
        build();
      },
      dispose() {
        for (const e of edgeSets) { e.geo.dispose(); e.mat.dispose(); }
        geo.dispose(); nodeMat.dispose(); nodeTex.dispose(); renderer.dispose();
        renderer.domElement.remove(); wrap.remove();
      },
    };
  },
};
