// c-contour.js - Contour-Line Reveal.
// A structure-contour map of a reservoir (top-of-reservoir depth, closed anticline,
// one fault scarp) is drawn as glowing iso-lines by a single full-screen fragment
// shader. The scalar field is blended over time: reservoir -> flat strata -> bent
// strata -> distance field of "Daryl Tan" (contours become concentric rings and the
// zero contour is the glyph outline) -> calm strata -> reservoir again.
// Everything is a pure function of t; the drift uses integer harmonics of the cycle.

import * as THREE from 'three';
import { createRenderer, ramp, easeInOutCubic, easeInOutSine, window4 } from '../core/stage.js';
import { rasterizeName, MONO_FAMILY } from '../core/text.js';

const TAU = Math.PI * 2;

// exact Euclidean distance transform (Felzenszwalb-Huttenlocher), squared distance to nearest 'on' cell
function edt(on, cols, rows) {
  const INF = 1e12, n = cols * rows, g = new Float32Array(n);
  const m = Math.max(cols, rows), f = new Float64Array(m), d = new Float64Array(m), v = new Int32Array(m), z = new Float64Array(m + 1);
  const run = (len) => {
    let k = 0; v[0] = 0; z[0] = -INF; z[1] = INF;
    for (let q = 1; q < len; q++) {
      let s;
      for (;;) { s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); if (s <= z[k] && k > 0) k--; else break; }
      k++; v[k] = q; z[k] = s; z[k + 1] = INF;
    }
    k = 0;
    for (let q = 0; q < len; q++) { while (z[k + 1] < q) k++; d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]; }
  };
  for (let x = 0; x < cols; x++) { for (let y = 0; y < rows; y++) f[y] = on[y * cols + x] ? 0 : INF; run(rows); for (let y = 0; y < rows; y++) g[y * cols + x] = d[y]; }
  for (let y = 0; y < rows; y++) { for (let x = 0; x < cols; x++) f[x] = g[y * cols + x]; run(cols); for (let x = 0; x < cols; x++) g[y * cols + x] = d[x]; }
  return g;
}

function boxBlur(a, cols, rows, rad, tmp) {
  const w = 2 * rad + 1;
  for (let y = 0; y < rows; y++) {
    let acc = 0; const o = y * cols;
    for (let x = -rad; x <= rad; x++) acc += a[o + Math.min(cols - 1, Math.max(0, x))];
    for (let x = 0; x < cols; x++) {
      tmp[o + x] = acc / w;
      acc += a[o + Math.min(cols - 1, x + rad + 1)] - a[o + Math.max(0, x - rad)];
    }
  }
  for (let x = 0; x < cols; x++) {
    let acc = 0;
    for (let y = -rad; y <= rad; y++) acc += tmp[Math.min(rows - 1, Math.max(0, y)) * cols + x];
    for (let y = 0; y < rows; y++) {
      a[y * cols + x] = acc / w;
      acc += tmp[Math.min(rows - 1, y + rad + 1) * cols + x] - tmp[Math.max(0, y - rad) * cols + x];
    }
  }
}

function signedDistance(ras, maxD, blurRad) {
  const { data, cols, rows } = ras, n = cols * rows;
  const ins = new Uint8Array(n), outs = new Uint8Array(n);
  for (let i = 0; i < n; i++) { const a = data[i] >= 128; ins[i] = a ? 1 : 0; outs[i] = a ? 0 : 1; }
  const dOutside = edt(ins, cols, rows);   // for outside cells: distance to glyph
  const dInside = edt(outs, cols, rows);   // for inside cells: distance to background
  const df = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = data[i];
    let d = a >= 128 ? Math.sqrt(dInside[i]) - 0.5 : -(Math.sqrt(dOutside[i]) - 0.5);
    if (a > 0 && a < 255) d = (a / 255 - 0.5) * 1.1;
    df[i] = Math.max(-maxD, Math.min(maxD, d));
  }
  // round the far-field corners: blend toward a blurred copy away from the outline
  const bl = Float32Array.from(df), tmp = new Float32Array(n);
  for (let k = 0; k < 3; k++) boxBlur(bl, cols, rows, blurRad, tmp);
  for (let i = 0; i < n; i++) {
    const ad = Math.abs(df[i]), k = Math.min(1, Math.max(0, (ad - 3 * blurRad * 0.25) / (blurRad * 2)));
    df[i] = df[i] + (bl[i] - df[i]) * k * k * (3 - 2 * k);
  }
  return df;
}

const VERT = `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const FRAG = `
precision highp float;
uniform vec2 uSize, uTex;
uniform float uDpr, uRes, uPh, uS, uB, uWarp, uGain, uRead, uSpT, uFs, uSpS, uFill, uBend, uQ, uSharp, uRet, uDef;
uniform sampler2D uDist, uSoft;
varying vec2 vUv;

const vec3 cLow = vec3(0.035, 0.27, 0.35);
const vec3 cMid = vec3(0.13, 0.55, 0.66);
const vec3 cHi  = vec3(0.74, 0.95, 0.98);
const vec3 amber = vec3(1.0, 0.70, 0.30);

vec3 ramp3(float c) { return c < 0.5 ? mix(cLow, cMid, c * 2.0) : mix(cMid, cHi, (c - 0.5) * 2.0); }

float asp() { return clamp(uSize.x / uSize.y, 0.46, 1.6); }
float scaleL() { return max(uSize.y, uSize.x * 0.5) * 0.5; }
vec2 origin() { return vec2(uSize.x * 0.5, uSize.y * 0.47); }
vec2 crest1(float A) { return vec2(0.22 * A, -0.08) + 0.025 * vec2(sin(uPh), cos(uPh)); }

float resField(vec2 u, float A) {
  vec2 w = u + 0.035 * vec2(sin(u.y * 3.3 + uPh), cos(u.x * 2.9 + 2.0 * uPh + 1.0))
             + 0.018 * vec2(sin(u.y * 7.0 - 2.0 * uPh), sin(u.x * 6.1 + uPh));
  vec2 c1 = crest1(A);
  vec2 c2 = vec2(-0.36 * A, 0.30) + 0.02 * vec2(cos(uPh), sin(2.0 * uPh));
  vec2 c3 = vec2(-0.02 * A, -0.52) + 0.02 * vec2(sin(2.0 * uPh), cos(uPh));
  vec2 d1 = (w - c1) / vec2(0.62, 0.40);
  vec2 d2 = (w - c2) / vec2(0.40, 0.30);
  vec2 d3 = (w - c3) / vec2(0.30, 0.20);
  float f = 12.5 + 6.5 * w.y + 1.4 * w.x;
  f -= 13.5 * exp(-dot(d1, d1)) + 8.0 * exp(-dot(d2, d2)) + 4.5 * exp(-dot(d3, d3));
  f += 0.9 * sin(w.x * 4.1 + w.y * 2.3 + 0.5 * sin(uPh));
  return f * (1.0 + 0.03 * sin(uPh));
}

void main() {
  vec2 p = vec2(gl_FragCoord.x, uSize.y * uDpr - gl_FragCoord.y) / uDpr;
  float L = scaleL(), A = asp();
  vec2 u = (p - origin()) / L;

  // ---- fields (in "level" units: one contour per integer) ----
  float fR = resField(u, A);
  float fault = 1.9 * smoothstep(-0.014, 0.014, dot(u, normalize(vec2(0.8, 0.6))) + 0.08 * sin(u.y * 2.2 + 1.0) - 0.30);
  float fS = (p.y - uSize.y * 0.47) / uSpS
           + uBend * (sin(p.x * 0.0042 + uPh) + 0.6 * sin(p.x * 0.0097 - 1.0 * uPh + p.y * 0.004))
           + 0.3 * sin(p.x * 0.003 + 2.0 * uPh);
  vec2 pw = p + uWarp * vec2(sin(p.y * 0.012 + uPh) + 0.5 * sin(p.y * 0.031 - uPh), cos(p.x * 0.010 + 2.0 * uPh) + 0.5 * sin(p.x * 0.027 + uPh));
  vec2 tuv = pw * uRes / uTex;
  float dcss = mix(texture2D(uSoft, tuv).r, texture2D(uDist, tuv).r, uSharp) / uRes;   // + inside glyph, - outside (css px)
  float fT = -dcss / uSpT;
  float r = max(-dcss, 0.0);

  vec2 eq = (p / uSize - vec2(0.5, 0.47)) * vec2(1.0, uSize.y / uSize.x * 1.6);
  float edgeN = clamp(length(eq) * 1.5, 0.0, 1.0);
  float rk = clamp((uRet - (1.0 - edgeN) * 0.82) / 0.18, 0.0, 1.0);
  float Sx = uS * (1.0 - rk * rk * (3.0 - 2.0 * rk));
  fS -= uDef * 15.0 * exp(-r / (uFs * 0.9));   // strata pile up into an anticline over the name
  float fBase = mix(fR, fS, Sx);
  float fNo = mix(fBase, fT, uB);
  float resW = (1.0 - Sx) * (1.0 - uB);
  float f = fNo + fault * resW;

  // ---- iso-line evaluation with screen-space AA ----
  float fw = max(fwidth(fNo), 1e-4);
  float dpx = abs(fract(f + 0.5) - 0.5) / fw;                 // device px to nearest contour
  float idx = floor(f + 0.5);
  float isIdx = 1.0 - step(0.5, mod(idx, 5.0));
  float hw = uDpr * mix(0.55, 0.95, isIdx);
  float core = 1.0 - smoothstep(hw - 0.6, hw + 0.6, dpx);
  float glow = exp(-dpx / (uDpr * 3.4)) * mix(0.16, 0.30, isIdx) * uQ;
  float dens = (1.0 - smoothstep(0.16, 0.38, fw)) * smoothstep(0.004, 0.014, fw);
  float I = mix(0.50, 1.0, isIdx) * dens;

  float inside = smoothstep(-0.8, 0.8, dcss);
  float ringI = mix(0.20, 1.0, exp(-r / (uFs * 0.85)));
  ringI *= mix(1.0, 0.45, inside) * (1.0 - smoothstep(uFs * 2.2, uFs * 3.6, r));
  I *= mix(1.0, ringI, uB);

  // ---- colour ----
  float cR = 1.0 - clamp((fR + 3.0) / 26.0, 0.0, 1.0);
  float cS = 0.50 - 0.28 * (p.y / uSize.y - 0.5);
  float cT = mix(0.12, 1.0, exp(-r / (uFs * 0.5)));
  float c = mix(mix(cR, cS, Sx), cT, uB);
  vec3 lineCol = mix(ramp3(c), cHi, isIdx * 0.40);
  float amW = resW * smoothstep(4.5, 1.0, fR);
  lineCol = mix(lineCol, amber, amW * 0.85);

  // ---- ground ----
  vec2 q = (p / uSize - vec2(0.5, 0.47));
  float vig = exp(-dot(q * vec2(1.1, 1.5), q * vec2(1.1, 1.5)) * 1.6);
  vec3 bg = mix(vec3(0.008, 0.016, 0.030), vec3(0.018, 0.048, 0.075), vig);
  bg += amber * 0.040 * smoothstep(3.0, -5.0, fR) * resW;
  bg += vec3(0.07, 0.26, 0.34) * inside * uFill * (0.30 + 1.0 * exp(-max(dcss, 0.0) / 9.0));
  bg += cHi * 0.035 * exp(-abs(dcss) / 6.0) * uB;

  // map graticule: faint registration crosses
  float G = 96.0;
  vec2 gp = mod(p - origin() + 0.5 * G, G) - 0.5 * G;
  float cross = (1.0 - smoothstep(0.4, 1.0, min(abs(gp.x), abs(gp.y)))) * (1.0 - smoothstep(3.0, 4.5, max(abs(gp.x), abs(gp.y))));
  bg += cMid * 0.30 * cross * (1.0 - uB) * (1.0 - 0.6 * uRead);

  vec3 col = bg + lineCol * (core + glow) * I * uGain;
  float edge = 1.0 - smoothstep(0.55 * uDpr - 0.6, 0.55 * uDpr + 0.6, abs(dcss) * uDpr);
  col += vec3(0.88, 0.98, 1.0) * (edge * 0.65 + exp(-abs(dcss) / 3.5) * 0.10) * uB * uFill;

  // well symbol on the crest (open ring + dot)
  vec2 wp = origin() + crest1(A) * L;
  float wd = length(p - wp);
  float ring = 1.0 - smoothstep(0.5, 1.3, abs(wd - 6.0) * uDpr / max(uDpr, 1.0));
  float dot_ = 1.0 - smoothstep(1.2, 2.2, wd);
  col += vec3(1.0, 0.86, 0.62) * (ring * 0.8 + dot_) * resW * (1.0 - 0.7 * uRead);

  gl_FragColor = vec4(col, 1.0);
}`;

export default {
  name: 'Contour-Line Reveal',
  blurb: 'A top-of-reservoir structure map flattens into strata, then its contours wrap the letters: the zero contour is the name.',
  tech: 'Three.js · fullscreen iso-line shader · glyph distance field',
  duration: 18,
  nameAt: 9.0,
  nameWeight: 700,
  phases: [
    { label: 'Structure map', start: 0, end: 2.4 },
    { label: 'Straighten', start: 2.4, end: 4.6 },
    { label: 'Bend', start: 4.6, end: 7.0 },
    { label: 'Contour name', start: 7.0, end: 9.0 },
  ],
  posterTime: 9.2,
  reveal: {
    colors: {
      name: '#F2F6F6', yijin: 'rgba(186,232,238,0.62)', sub: 'rgba(214,236,240,0.72)',
      accent: '#E9A24C', rule: 'rgba(150,215,225,0.30)', glow: 'rgba(110,205,220,0.45)',
    },
  },

  create(ctx) {
    const renderer = createRenderer(ctx, { clearColor: 0x02060c });
    const scene = new THREE.Scene();
    const camera = new THREE.Camera();
    const T = ctx.timing;
    const low = ctx.quality === 'low';

    const uniforms = {
      uSize: { value: new THREE.Vector2(ctx.width, ctx.height) },
      uTex: { value: new THREE.Vector2(1, 1) },
      uDpr: { value: renderer.getPixelRatio() },
      uRes: { value: 1 }, uPh: { value: 0 }, uS: { value: 0 }, uB: { value: 0 }, uWarp: { value: 0 },
      uGain: { value: 1 }, uRead: { value: 0 }, uSpT: { value: 12 }, uFs: { value: 120 }, uSpS: { value: 30 },
      uFill: { value: 0 }, uBend: { value: 0.4 }, uQ: { value: low ? 0.6 : 1 },
      uDist: { value: null }, uSoft: { value: null }, uSharp: { value: 0 }, uRet: { value: 0 }, uDef: { value: 0 },
    };
    const mat = new THREE.ShaderMaterial({ uniforms, vertexShader: VERT, fragmentShader: FRAG, depthTest: false, depthWrite: false });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat);
    mesh.frustumCulled = false;
    scene.add(mesh);

    // tiny mono map annotations (DOM, pure functions of t)
    const labels = document.createElement('div');
    labels.setAttribute('aria-hidden', 'true');
    Object.assign(labels.style, { position: 'absolute', inset: '0', pointerEvents: 'none', overflow: 'hidden', zIndex: '2',
      fontFamily: MONO_FAMILY, fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase', color: 'rgba(150,215,225,0.55)' });
    const wellLabel = document.createElement('div');
    wellLabel.textContent = 'W-1  − 2 150 m';
    Object.assign(wellLabel.style, { position: 'absolute', left: '0', top: '0', whiteSpace: 'nowrap', willChange: 'transform, opacity' });
    const legend = document.createElement('div');
    legend.textContent = 'Top reservoir · structure map · CI 25 m';
    Object.assign(legend.style, { position: 'absolute', left: '16px', bottom: '14px', whiteSpace: 'nowrap', willChange: 'opacity' });
    labels.append(wellLabel, legend);
    ctx.container.appendChild(labels);

    let tex = null, texSoft = null, cur = ctx;

    function build(c) {
      cur = c;
      const L = c.layout;
      const res = Math.min(2, Math.max(0.5, Math.sqrt(1.4e6 / (c.width * c.height)))) * (c.quality === 'low' ? 0.55 : 1);
      const ras = rasterizeName(L, { which: 'short', res });
      const df = signedDistance(ras, 400 * res, Math.max(2, Math.round(5 * res)));
      const mk = (arr) => {
        const half = new Uint16Array(arr.length);
        for (let i = 0; i < arr.length; i++) half[i] = THREE.DataUtils.toHalfFloat(arr[i]);
        const tx = new THREE.DataTexture(half, ras.cols, ras.rows, THREE.RedFormat, THREE.HalfFloatType);
        tx.minFilter = tx.magFilter = THREE.LinearFilter;
        tx.generateMipmaps = false;
        tx.needsUpdate = true;
        return tx;
      };
      // heavily blurred copy: lets the letters emerge from smooth blobs
      const soft = Float32Array.from(df), tmp = new Float32Array(df.length);
      const sr = Math.max(3, Math.round(L.fontSize * 0.11 * res));
      for (let k = 0; k < 3; k++) boxBlur(soft, ras.cols, ras.rows, sr, tmp);
      if (tex) tex.dispose();
      if (texSoft) texSoft.dispose();
      tex = mk(df); texSoft = mk(soft);
      uniforms.uSoft.value = texSoft;
      uniforms.uDist.value = tex;
      uniforms.uTex.value.set(ras.cols, ras.rows);
      uniforms.uRes.value = res;
      uniforms.uSize.value.set(c.width, c.height);
      uniforms.uSpT.value = Math.max(8, L.fontSize * 0.075);
      uniforms.uFs.value = L.fontSize;
      uniforms.uSpS.value = Math.min(34, Math.max(20, c.height / 26));
      legend.style.display = c.width < 700 ? 'none' : '';
    }
    build(ctx);

    const nameAt = T.nameAt, D = T.duration;
    return {
      stats: `1 fullscreen shader · ${tex.image.width}x${tex.image.height} distance field · 24 levels`,
      update(t) {
        const ph = (TAU * t) / D;
        const S = easeInOutCubic(ramp(t, 2.3, 4.9));
        const ret = ramp(t, D - 2.0, D);
        const Bin = easeInOutCubic(ramp(t, 6.4, nameAt));
        const Bout = easeInOutSine(ramp(t, nameAt + 0.4, nameAt + 1.2));
        const B = Bin * (1 - Bout);
        const dim = easeInOutSine(ramp(t, nameAt + 0.6, nameAt + 2.8)) * (1 - easeInOutSine(ramp(t, D - 1.3, D)));
        const fillIn = ramp(t, nameAt - 1.6, nameAt);
        const fill = fillIn * fillIn * (1 - ramp(t, nameAt + 0.3, nameAt + 1.15));

        const u = uniforms;
        u.uPh.value = ph;
        u.uS.value = S;
        u.uRet.value = ret;
        u.uDef.value = easeInOutCubic(ramp(t, 4.4, 6.8)) * (1 - ramp(t, nameAt, nameAt + 0.3));
        u.uB.value = B;
        const sh = ramp(Bin, 0.5, 0.95);
        u.uSharp.value = sh * sh * (3 - 2 * sh);
        u.uWarp.value = 38 * Math.pow(1 - Bin, 1.5);
        u.uGain.value = 1 - 0.68 * dim;
        u.uRead.value = dim;
        u.uFill.value = fill;
        u.uBend.value = 0.4 + 2.6 * window4(t, 3.2, 4.6, 5.0, 6.8);

        // annotation follows the crest well (same formula as the shader)
        const w = cur.width, h = cur.height;
        const A = Math.min(1.6, Math.max(0.46, w / h));
        const L = Math.max(h, w * 0.5) * 0.5;
        const wx = w * 0.5 + (0.22 * A + 0.025 * Math.sin(ph)) * L;
        const wy = h * 0.47 + (-0.08 + 0.025 * Math.cos(ph)) * L;
        const vis = (1 - S * (1 - ret)) * (1 - B);
        wellLabel.style.transform = `translate(${(wx + 14).toFixed(1)}px, ${(wy - 6).toFixed(1)}px)`;
        wellLabel.style.opacity = vis.toFixed(3);
        legend.style.opacity = (0.9 * vis).toFixed(3);
        renderer.render(scene, camera);
      },
      resize(c) {
        renderer.setSize(c.width, c.height, false);
        uniforms.uDpr.value = renderer.getPixelRatio();
        build(c);
      },
      dispose() {
        if (tex) tex.dispose();
        if (texSoft) texSoft.dispose();
        mesh.geometry.dispose(); mat.dispose(); renderer.dispose();
        renderer.domElement.remove(); labels.remove();
      },
    };
  },
};
