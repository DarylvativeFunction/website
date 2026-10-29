// a-reservoir.js - Prototype A, "Cinematic Reservoir".
//
// A cut-away block of an anticline trap rendered as a seismic-style point cloud.
// One continuous shot: strata emerge behind a scan line -> a wellbore drills into
// the crest -> amber fluid converges along streamlines (pressure drawdown) ->
// a corner-point grid conforms to the horizons -> the continuum snaps to cell
// centres -> those nodes loosen into a k-NN network -> the nodes fly onto the
// shared "Daryl Tan" glyphs while the camera settles into its canonical pose.
//
// Everything is a pure function of t: geometry is built once per size, every
// motion is a shader morph driven by uniforms computed from t.

import * as THREE from 'three';
import { EffectComposer, RenderPass, UnrealBloomPass, OutputPass } from 'three/addons';
import { createRenderer, pixelToPlane, ramp, clamp, lerp, easeInOutCubic, easeOutCubic, easeInOutSine, rng } from '../core/stage.js';
import { sampleName, rasterizeName } from '../core/text.js';

const NAME_AT = 11.0;
const DURATION = 19.8;
const FOV = 32;
const CAM_Z = 20;
const R0 = 1.32; // streamline start radius (ellipse units, clipped to the block)

// Horizons of the anticline. y = b + A * dome(x, z) + U * undulation(x, z, k)
const HZ = [
  { b: 2.0, A: 0.0, U: 0.0 },    // 0 block top (cut)
  { b: 0.55, A: 0.95, U: 1.0 },  // 1 top cap rock
  { b: 0.02, A: 0.98, U: 1.0 },  // 2 top reservoir
  { b: -0.78, A: 0.80, U: 0.8 }, // 3 base reservoir / top shale
  { b: -1.42, A: 0.55, U: 0.6 }, // 4 top basement
  { b: -2.2, A: 0.0, U: 0.0 },   // 5 block bottom (cut)
];
const OWC = -0.12; // oil-water contact

// Layer palette (display values, additive): overburden, cap rock, reservoir (oil), shale, basement
const LAYER_COL = [
  [0.40, 0.43, 0.48],
  [0.30, 0.36, 0.45],
  [0.86, 0.58, 0.26],
  [0.26, 0.30, 0.36],
  [0.36, 0.30, 0.27],
];
const WATER_COL = [0.30, 0.40, 0.46];
const LAYER_DENS = [0.5, 0.3, 1.0, 0.45, 0.7];
const LAYER_GAIN = [0.8, 0.75, 1.2, 0.8, 0.95];

const GLSL_GEO = /* glsl */`
  uniform vec2 uHalf;  // block half extents (x, z)
  uniform vec2 uDome;  // anticline gaussian widths (x, z)
  float dome(vec2 p) { return exp(-(p.x * p.x / (2.0 * uDome.x * uDome.x) + p.y * p.y / (2.0 * uDome.y * uDome.y))); }
  float und(vec2 p, float k) { return 0.09 * sin(0.62 * p.x + 0.38 * p.y + 1.7 * k) + 0.05 * sin(1.35 * p.y - 0.45 * p.x + 0.9 * k); }
  float resTop(vec2 p) { return 0.02 + 0.98 * dome(p) + und(p, 2.0); }
  float resBase(vec2 p) { return -0.78 + 0.80 * dome(p) + 0.8 * und(p, 3.0); }
  uniform vec3 uCamL; // camera position in block space
  uniform float uDens; // brightness compensation when the block is drawn small (narrow screens)
  // fade geometry lying on a vertical face of the block when that face is seen edge-on
  float faceFade(vec3 p) {
    vec3 v = normalize(uCamL - p);
    float f = 1.0;
    if (abs(abs(p.x) - uHalf.x) < 1e-3) f = min(f, smoothstep(0.06, 0.3, abs(v.x)));
    if (abs(abs(p.z) - uHalf.y) < 1e-3) f = min(f, smoothstep(0.06, 0.3, abs(v.z)));
    return f;
  }
`;

const GLSL_FOG = /* glsl */`
  uniform float uFogNear, uFogFar;
  float fogOf(float depth) { return 1.0 - 0.72 * smoothstep(uFogNear, uFogFar, depth); }
`;

// shared node motion (nodes and edges must agree to the pixel)
const GLSL_NODE = /* glsl */`
  uniform float uT, uLoose, uMorph, uDisperse, uDrift;
  vec3 nodePos(vec3 cell, vec3 jit, vec3 tgt, vec3 scat, vec4 meta, out float e) {
    float seed = meta.x;
    vec3 net = cell + jit * uLoose;
    net += vec3(sin(uT * 0.61 + seed * 31.0), sin(uT * 0.47 + seed * 17.0), cos(uT * 0.53 + seed * 23.0)) * 0.06 * uLoose;
    float m = clamp((uMorph - meta.y * 0.42) / 0.58, 0.0, 1.0);
    e = m * m * m * (m * (m * 6.0 - 15.0) + 10.0);
    vec3 p = mix(net, tgt, e);
    p.z += sin(3.14159 * e) * (0.8 + seed * 1.6);
    p.y += sin(3.14159 * e) * (seed - 0.5) * 0.9;
    vec3 sc = scat * uDisperse;
    sc += vec3(sin(uT * 0.21 + seed * 40.0), cos(uT * 0.17 + seed * 13.0), 0.0) * 0.35 * uDrift;
    return p + sc;
  }
`;

const SOFT_DOT = /* glsl */`
  float softDot(vec2 pc) { float d = length(pc - 0.5) * 2.0; float c = smoothstep(1.0, 0.0, d); return c * c; }
`;

// --- JS mirrors of the geometry functions -----------------------------------
function makeGeo(aspect) {
  const W = clamp(aspect * 7.4, 7.6, 12);
  const D = clamp(aspect * 4.4, 5.6, 7);
  return { W, D, AX: W * 0.24, AZ: D * 0.62 };
}
function domeJS(G, x, z) { return Math.exp(-(x * x / (2 * G.AX * G.AX) + z * z / (2 * G.AZ * G.AZ))); }
function undJS(x, z, k) { return 0.09 * Math.sin(0.62 * x + 0.38 * z + 1.7 * k) + 0.05 * Math.sin(1.35 * z - 0.45 * x + 0.9 * k); }
function hz(G, k, x, z) { const h = HZ[k]; return h.b + h.A * domeJS(G, x, z) + h.U * undJS(x, z, k); }
// grid k-surfaces: horizons plus a mid-reservoir surface (7 surfaces, 6 cell layers)
function gs(G, kk, x, z) {
  if (kk <= 2) return hz(G, kk, x, z);
  if (kk === 3) return 0.5 * (hz(G, 2, x, z) + hz(G, 3, x, z));
  return hz(G, kk - 1, x, z);
}

// Hermite interpolation through keyframes (finite-difference tangents, flat ends)
function hermiteKeys(keys, t, ch) {
  const n = keys.length;
  if (t <= keys[0][0]) return keys[0][ch];
  if (t >= keys[n - 1][0]) return keys[n - 1][ch];
  let i = 0;
  while (t > keys[i + 1][0]) i++;
  const t0 = keys[i][0], t1 = keys[i + 1][0], h = t1 - t0;
  const p0 = keys[i][ch], p1 = keys[i + 1][ch];
  const tan = (j) => (j <= 0 || j >= n - 1) ? 0 : (keys[j + 1][ch] - keys[j - 1][ch]) / (keys[j + 1][0] - keys[j - 1][0]);
  const m0 = tan(i) * h, m1 = tan(i + 1) * h;
  const u = (t - t0) / h, u2 = u * u, u3 = u2 * u;
  return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1;
}

// camera: t, azimuth (deg), elevation (deg), distance, target y
const CAM_KEYS = [
  [0.0, 38, 22, 20.5, -0.6],
  [2.7, 32, 19, 18.6, -0.3],
  [5.5, 22, 28, 19.0, -0.5],
  [8.4, 11, 15, 19.6, -0.2],
  [NAME_AT - 0.12, 0, 0, CAM_Z, 0],
];

export default {
  name: 'Cinematic Reservoir',
  blurb: 'A cut-away anticline trap emerges under a scan line, is drilled, flows, is gridded and discretised; its cells become a network that converges into the name.',
  tech: 'Three.js · GPU point morphs · corner-point grid · k-NN graph · bloom',
  duration: DURATION,
  nameAt: NAME_AT,
  nameWeight: 700,
  phases: [
    { label: 'Strata', start: 0, end: 2.3 },
    { label: 'Wellbore', start: 2.3, end: 3.7 },
    { label: 'Flow', start: 3.7, end: 5.3 },
    { label: 'Grid', start: 5.3, end: 6.9 },
    { label: 'Discretise', start: 6.9, end: 8.3 },
    { label: 'Network', start: 8.3, end: 9.5 },
    { label: 'Converge', start: 9.5, end: NAME_AT },
  ],
  reveal: {
    colors: {
      name: '#F3F2EE',
      yijin: 'rgba(243,242,238,0.58)',
      sub: 'rgba(243,242,238,0.70)',
      accent: '#E8A33D',
      rule: 'rgba(92,200,255,0.32)',
      glow: 'rgba(92,200,255,0.42)',
    },
  },
  posterTime: 15.5,

  create(ctx) {
    const high = ctx.quality !== 'low';
    const renderer = createRenderer(ctx, { clearColor: 0x030405 });
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    const dpr = renderer.getPixelRatio();
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(FOV, ctx.width / ctx.height, 0.1, 400);
    const world = new THREE.Group();
    scene.add(world);

    const C = high
      ? { strata: 44000, flow: 6000, lines: 64, NI: 28, NJ: 16 }
      : { strata: 15000, flow: 2000, lines: 32, NI: 16, NJ: 9 };
    const NK = 6;
    const NN = C.NI * C.NJ * NK;

    // ---------- shared uniforms ----------
    const U = {
      uHalf: { value: new THREE.Vector2(6, 3.5) },
      uDome: { value: new THREE.Vector2(3.3, 2.5) },
      uFogNear: { value: 18 }, uFogFar: { value: 30 },
      uT: { value: 0 }, uDpr: { value: dpr }, uCamZ: { value: CAM_Z }, uS: { value: 1 },
      uCamL: { value: new THREE.Vector3(0, 0, CAM_Z) },
      uDens: { value: 1 },
      uLoose: { value: 0 }, uMorph: { value: 0 }, uDisperse: { value: 0 }, uDrift: { value: 0 },
    };
    const shared = (extra) => Object.assign({}, U, extra);
    // premultiplied additive: every shader writes (rgb * a, a)
    const additive = { transparent: true, depthWrite: false, depthTest: false, blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor };

    // ---------- background (very subtle radial falloff) ----------
    const bgMat = new THREE.ShaderMaterial({
      depthWrite: false, depthTest: false,
      uniforms: { uAspect: { value: ctx.width / ctx.height }, uLift: { value: 1 } },
      vertexShader: `varying vec2 vUv; void main() { vUv = position.xy * 0.5 + 0.5; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
      fragmentShader: `
        uniform float uAspect, uLift; varying vec2 vUv;
        void main() {
          vec2 d = (vUv - vec2(0.5, 0.56)) * vec2(uAspect, 1.0);
          float g = smoothstep(1.15, 0.0, length(d));
          vec3 c = mix(vec3(0.010, 0.012, 0.016), vec3(0.030, 0.038, 0.048), g * uLift);
          gl_FragColor = vec4(c, 1.0);
        }`,
    });
    const bgGeo = new THREE.BufferGeometry();
    bgGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
    const bg = new THREE.Mesh(bgGeo, bgMat);
    bg.frustumCulled = false; bg.renderOrder = -10;
    scene.add(bg);

    // ---------- materials ----------
    const strataMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uScan: { value: -1 }, uAlpha: { value: 1 }, uSnap: { value: 0 }, uPx: { value: 1.7 }, uGlint: { value: -1 }, uGlintA: { value: 0 } }),
      vertexShader: `
        attribute vec3 aCol; attribute vec3 aCell; attribute vec2 aSeed;
        uniform float uScan, uAlpha, uSnap, uPx, uDpr, uCamZ, uS, uGlint, uGlintA;
        varying vec3 vCol; varying float vA;
        ${GLSL_GEO} ${GLSL_FOG}
        void main() {
          vec3 p = position;
          float xn = (p.x + uHalf.x) / (2.0 * uHalf.x);
          // seismic scan: revealed behind the line, glint on the line
          float d = uScan - xn;
          float rev = smoothstep(-0.005, 0.07, d);
          float band = exp(-d * d / 0.0009) * step(-0.05, d) * (1.0 - smoothstep(1.02, 1.12, uScan));
          float dg = uGlint - xn;
          float glint = exp(-dg * dg / 0.006) * uGlintA;
          float ff = faceFade(p);
          // discretisation: slide to the containing cell centre and vanish
          float s = clamp(uSnap * 1.55 - xn * 0.4 - aSeed.x * 0.15, 0.0, 1.0);
          float es = s * s * (3.0 - 2.0 * s);
          p = mix(p, aCell, es);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          float depth = -mv.z;
          float light = 0.78 + 0.34 * smoothstep(-1.0, 1.0, (-p.x * 0.10 + p.y * 0.42 + p.z * 0.14));
          vA = uDens * uAlpha * (rev + band * 1.3 + glint) * fogOf(depth) * (1.0 - smoothstep(0.45, 1.0, s)) * ff;
          vCol = aCol * light + vec3(0.55, 0.85, 1.0) * (band * 0.7 + glint * 0.35) + vec3(0.4, 0.7, 1.0) * es * 0.6;
          gl_PointSize = uPx * uDpr * (uCamZ / depth) * (0.7 + 0.6 * aSeed.y) * (1.0 + band * 0.8);
        }`,
      fragmentShader: `
        varying vec3 vCol; varying float vA; ${SOFT_DOT}
        void main() { float a = softDot(gl_PointCoord) * vA; if (a < 0.003) discard; gl_FragColor = vec4(vCol * a, a); }`,
    });

    const flowMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uAlpha: { value: 0 }, uPx: { value: 2.0 } }),
      vertexShader: `
        attribute vec4 aPath; attribute vec2 aPh;
        uniform float uT, uAlpha, uPx, uDpr, uCamZ;
        varying float vA; varying float vHot;
        ${GLSL_GEO} ${GLSL_FOG}
        void main() {
          float u = fract(aPath.w * uT + aPh.x);
          float r = ${R0.toFixed(3)} * sqrt(1.0 - u);
          float th = aPath.x + aPath.y * (1.0 - r / ${R0.toFixed(3)});
          vec2 xz = vec2(cos(th) * uHalf.x, sin(th) * uHalf.y) * r;
          float y = mix(resBase(xz), resTop(xz), aPath.z);
          vec3 p = vec3(xz.x, y, xz.y);
          float inside = step(abs(xz.x), uHalf.x * 0.995) * step(abs(xz.y), uHalf.y * 0.995);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          vHot = smoothstep(0.45, 0.0, r);
          vA = uDens * uAlpha * inside * smoothstep(0.0, 0.08, u) * smoothstep(0.015, 0.09, r) * fogOf(-mv.z) * (0.55 + 0.45 * aPh.y);
          gl_PointSize = uPx * uDpr * (uCamZ / -mv.z) * (0.8 + 0.5 * aPh.y + vHot * 0.6);
        }`,
      fragmentShader: `
        varying float vA; varying float vHot; ${SOFT_DOT}
        void main() {
          float a = softDot(gl_PointCoord) * vA; if (a < 0.003) discard;
          vec3 c = mix(vec3(0.91, 0.64, 0.24), vec3(1.0, 0.88, 0.66), vHot);
          gl_FragColor = vec4(c * a, a);
        }`,
    });

    const streamMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uAlpha: { value: 0 } }),
      vertexShader: `
        attribute float aU; uniform float uAlpha; varying float vU; varying float vA;
        ${GLSL_FOG}
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          vU = aU; vA = sqrt(uDens) * uAlpha * fogOf(-mv.z);
        }`,
      fragmentShader: `
        uniform float uT; varying float vU; varying float vA;
        void main() {
          float flow = 0.6 + 0.4 * sin(vU * 34.0 - uT * 4.0);
          float a = vA * (0.18 + 0.82 * vU * vU) * flow * 0.55;
          gl_FragColor = vec4(vec3(0.91, 0.62, 0.26) * a, a);
        }`,
    });

    const gridMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uSweep: { value: -1 }, uAlpha: { value: 0 } }),
      vertexShader: `
        attribute float aW; uniform float uSweep, uAlpha;
        varying float vA; varying float vBand;
        ${GLSL_GEO} ${GLSL_FOG}
        void main() {
          vec3 p = position;
          float rn = length(p.xz / uHalf) / 1.4142;
          float d = uSweep - rn;
          float ff = faceFade(p);
          vA = sqrt(uDens) * aW * uAlpha * smoothstep(0.0, 0.05, d) * fogOf(-(modelViewMatrix * vec4(p, 1.0)).z) * ff;
          vBand = sqrt(uDens) * ff * exp(-d * d / 0.0016) * aW * step(0.0, d + 0.06) * (1.0 - smoothstep(1.0, 1.12, uSweep)) * step(0.001, uAlpha);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
        }`,
      fragmentShader: `
        varying float vA; varying float vBand;
        void main() {
          float a = vA * 0.42 + vBand * 0.9;
          vec3 c = vec3(0.36, 0.78, 1.0) * vA * 0.42 + vec3(0.75, 0.93, 1.0) * vBand * 0.9;
          gl_FragColor = vec4(c, a);
        }`,
    });

    const wellMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uGrow: { value: 0 }, uAlpha: { value: 0 }, uTop: { value: 3.8 }, uTip: { value: 0.3 } }),
      vertexShader: `
        attribute float aY; attribute float aW; uniform float uGrow, uAlpha, uTop, uTip;
        varying float vA;
        void main() {
          float f = min(aY, uGrow);
          vec3 p = vec3(position.x, mix(uTop, uTip, f), position.z);
          vA = uAlpha * aW * (0.35 + 0.65 * smoothstep(uGrow - 0.5, uGrow, f)) * smoothstep(0.0, 0.12, f + 0.02);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
        }`,
      fragmentShader: `
        varying float vA;
        void main() { gl_FragColor = vec4(vec3(0.62, 0.88, 1.0) * vA, vA); }`,
    });

    const glowMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uGrow: { value: 0 }, uHead: { value: 0 }, uPerf: { value: 0 }, uTop: { value: 3.8 }, uTip: { value: 0.3 } }),
      vertexShader: `
        attribute vec4 aG; // kind (0 head, 1 perforation), size, phase, perf y fraction
        uniform float uGrow, uHead, uPerf, uTop, uTip, uDpr, uCamZ, uT;
        varying float vA; varying float vKind;
        void main() {
          vec3 p = position;
          if (aG.x < 0.5) p.y = mix(uTop, uTip, uGrow);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          float flick = 0.75 + 0.25 * sin(uT * 7.0 + aG.z * 6.28);
          vA = aG.x < 0.5 ? uHead : uPerf * flick;
          vKind = aG.x;
          gl_PointSize = aG.y * uDpr * (uCamZ / -mv.z);
        }`,
      fragmentShader: `
        varying float vA; varying float vKind;
        void main() {
          float d = length(gl_PointCoord - 0.5) * 2.0;
          float core = exp(-d * d * 18.0), halo = exp(-d * d * 3.5) * 0.35;
          float a = (core + halo) * vA; if (a < 0.003) discard;
          vec3 c = vKind < 0.5 ? vec3(0.55, 0.85, 1.0) : mix(vec3(0.91, 0.64, 0.24), vec3(0.62, 0.88, 1.0), core);
          gl_FragColor = vec4(c * a, a);
        }`,
    });

    const ringMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uPhase: { value: 0 }, uAlpha: { value: 0 } }),
      vertexShader: `
        attribute vec2 aRing; uniform float uPhase, uAlpha; varying float vA;
        ${GLSL_GEO} ${GLSL_FOG}
        void main() {
          float r = fract(uPhase + aRing.y / 3.0) * 1.15;
          vec2 xz = vec2(cos(aRing.x) * uHalf.x, sin(aRing.x) * uHalf.y) * r;
          vec3 p = vec3(xz.x, mix(resBase(xz), resTop(xz), 0.62), xz.y);
          float inside = step(abs(xz.x), uHalf.x) * step(abs(xz.y), uHalf.y);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          vA = uAlpha * inside * pow(1.0 - r / 1.15, 1.6) * smoothstep(0.0, 0.06, r) * fogOf(-mv.z);
        }`,
      fragmentShader: `
        varying float vA;
        void main() { gl_FragColor = vec4(vec3(0.95, 0.70, 0.32) * vA * 0.6, vA * 0.6); }`,
    });

    const nodeMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uIn: { value: 0 }, uAlpha: { value: 0 }, uNameA: { value: 0.8 }, uPx: { value: 2.4 }, uNamePx: { value: 3 } }),
      vertexShader: `
        attribute vec3 aCell, aJit, aTgt, aScat; attribute vec4 aMeta;
        uniform float uIn, uAlpha, uNameA, uPx, uNamePx, uDpr, uCamZ;
        varying float vA; varying float vE; varying float vFlash;
        ${GLSL_GEO} ${GLSL_FOG} ${GLSL_NODE}
        void main() {
          float e;
          vec3 p = nodePos(aCell, aJit, aTgt, aScat, aMeta, e);
          float a0 = clamp(uIn * 1.45 - aMeta.z * 0.45, 0.0, 1.0);
          vFlash = a0 * (1.0 - a0) * 4.0;
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          float fog = mix(fogOf(-mv.z), 1.0, e);
          vA = uAlpha * smoothstep(0.0, 0.6, a0) * fog * mix((0.55 + 0.45 * aMeta.w) * uDens, uNameA, e);
          vE = e;
          float px = mix(uPx * (0.75 + 0.5 * aMeta.w), uNamePx, e);
          gl_PointSize = px * uDpr * (uCamZ / -mv.z) * (1.0 + vFlash * 0.8);
        }`,
      fragmentShader: `
        varying float vA; varying float vE; varying float vFlash; ${SOFT_DOT}
        void main() {
          float a = softDot(gl_PointCoord) * vA; if (a < 0.003) discard;
          vec3 c = mix(vec3(0.62, 0.86, 1.0), vec3(0.96, 0.96, 0.95), vE) + vec3(0.3, 0.5, 0.6) * vFlash;
          gl_FragColor = vec4(c * a, a);
        }`,
    });

    const edgeMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uGrow: { value: 0 }, uAlpha: { value: 0 } }),
      vertexShader: `
        attribute vec3 aAc, aAj, aAt, aBc, aBj, aBt; attribute vec4 aAm, aBm; attribute vec4 aE; // end, order, rand, kind
        uniform float uGrow, uAlpha;
        varying float vA; varying float vS; varying float vPulse;
        ${GLSL_GEO} ${GLSL_FOG} ${GLSL_NODE}
        void main() {
          float eA, eB;
          vec3 pa = nodePos(aAc, aAj, aAt, vec3(0.0), aAm, eA);
          vec3 pb = nodePos(aBc, aBj, aBt, vec3(0.0), aBm, eB);
          float g = clamp((uGrow * 1.5 - aE.y * 0.5) * 3.0, 0.0, 1.0);
          g = g * g * (3.0 - 2.0 * g);
          vec3 p = mix(pa, mix(pa, pb, g), aE.x);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          float arrive = max(eA, eB);
          vA = uDens * uAlpha * step(0.001, g) * (1.0 - smoothstep(0.0, 0.55, arrive)) * fogOf(-mv.z) * (aE.w > 0.5 ? 0.45 : 1.0);
          vS = aE.x * g;
          vPulse = aE.z;
        }`,
      fragmentShader: `
        uniform float uT; varying float vA; varying float vS; varying float vPulse;
        void main() {
          float ph = fract(uT * 0.55 + vPulse * 7.0);
          float pulse = step(vPulse, 0.12) * exp(-pow((vS - ph) / 0.05, 2.0));
          float a = vA * (0.22 + pulse * 0.8);
          vec3 c = vec3(0.36, 0.78, 1.0) * vA * 0.22 + vec3(0.95, 0.68, 0.30) * vA * pulse * 0.8;
          gl_FragColor = vec4(c, a);
        }`,
    });

    // ---------- objects (geometry filled in build) ----------
    const strata = new THREE.Points(new THREE.BufferGeometry(), strataMat);
    const flow = new THREE.Points(new THREE.BufferGeometry(), flowMat);
    const stream = new THREE.LineSegments(new THREE.BufferGeometry(), streamMat);
    const grid = new THREE.LineSegments(new THREE.BufferGeometry(), gridMat);
    const well = new THREE.LineSegments(new THREE.BufferGeometry(), wellMat);
    const glow = new THREE.Points(new THREE.BufferGeometry(), glowMat);
    const rings = new THREE.LineSegments(new THREE.BufferGeometry(), ringMat);
    const edges = new THREE.LineSegments(new THREE.BufferGeometry(), edgeMat);
    const nodes = new THREE.Points(new THREE.BufferGeometry(), nodeMat);
    const objs = [strata, stream, flow, rings, grid, well, glow, edges, nodes];
    objs.forEach((o, i) => { o.frustumCulled = false; o.renderOrder = i; world.add(o); });

    // ---------- post ----------
    let composer = null, bloom = null;
    if (high) {
      composer = new EffectComposer(renderer);
      composer.addPass(new RenderPass(scene, camera));
      bloom = new UnrealBloomPass(new THREE.Vector2(ctx.width, ctx.height), 0.55, 0.55, 0.16);
      composer.addPass(bloom);
      composer.addPass(new OutputPass());
    }

    let G = makeGeo(ctx.width / ctx.height);
    let S = 1;          // world scale of the block (fits narrow screens)
    let bgAspect = 1;

    const setGeo = (o, attrs) => {
      o.geometry.dispose();
      const g = new THREE.BufferGeometry();
      for (const k in attrs) g.setAttribute(k, new THREE.BufferAttribute(attrs[k][0], attrs[k][1]));
      g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e4);
      o.geometry = g;
    };

    function build(c) {
      const w = c.width, h = c.height, aspect = w / h;
      G = makeGeo(aspect);
      const { W, D } = G, HW = W / 2, HD = D / 2;
      U.uHalf.value.set(HW, HD);
      U.uDome.value.set(G.AX, G.AZ);
      camera.aspect = aspect;
      camera.updateProjectionMatrix();
      const visH = 2 * CAM_Z * Math.tan(THREE.MathUtils.degToRad(FOV / 2));
      const visW = visH * aspect;
      const az0 = THREE.MathUtils.degToRad(CAM_KEYS[0][1]);
      const proj = W * Math.cos(az0) + D * Math.sin(az0);
      S = Math.min(1, (visW * 0.9) / proj);
      world.scale.setScalar(S);
      U.uS.value = S;
      U.uDens.value = clamp(S * S * 1.15, 0.3, 1);
      if (bloom) bloom.strength = 0.55 * lerp(0.55, 1, S);
      bgAspect = aspect;
      bgMat.uniforms.uAspect.value = aspect;

      const NI = C.NI, NJ = C.NJ;
      const cw = W / NI, cd = D / NJ;
      const cellIndex = (x, y, z) => {
        const i = clamp(Math.floor((x + HW) / cw), 0, NI - 1);
        const j = clamp(Math.floor((z + HD) / cd), 0, NJ - 1);
        const xc = -HW + (i + 0.5) * cw, zc = -HD + (j + 0.5) * cd;
        let k = NK - 1;
        for (let kk = 0; kk < NK; kk++) if (y >= gs(G, kk + 1, x, z)) { k = kk; break; }
        return [i, j, k, xc, zc];
      };
      const cellCentre = (i, j, k) => {
        const xc = -HW + (i + 0.5) * cw, zc = -HD + (j + 0.5) * cd;
        return [xc, 0.5 * (gs(G, k, xc, zc) + gs(G, k + 1, xc, zc)), zc];
      };
      const layerOf = (x, y, z) => { for (let k = 0; k < 4; k++) if (y >= hz(G, k + 1, x, z)) return k; return 4; };

      // ----- strata point cloud -----
      {
        const N = C.strata, r = rng(101);
        const pos = new Float32Array(N * 3), col = new Float32Array(N * 3), cell = new Float32Array(N * 3), seed = new Float32Array(N * 2);
        const gauss = () => (r() + r() + r() - 1.5) / 1.5;
        for (let n = 0; n < N; n++) {
          const q = r();
          let x, y, z, sheet = -1, trace = false;
          const faceY = (fx, fz) => {
            // layer-weighted density on the cut faces: tight rocks sparse, sandstone dense
            for (let tries = 0; tries < 6; tries++) {
              const yy = HZ[5].b + r() * (HZ[0].b - HZ[5].b);
              if (r() < LAYER_DENS[layerOf(fx, yy, fz)]) return yy;
            }
            return HZ[5].b + r() * (HZ[0].b - HZ[5].b);
          };
          if (q < 0.42) { x = -HW + r() * W; z = HD; y = faceY(x, z); }
          else if (q < 0.5) { x = HW; z = -HD + r() * D; y = faceY(x, z); }
          else if (q < 0.58) {
            // reflector traces: horizons where they cut the front and right faces
            trace = true;
            sheet = 1 + Math.floor(r() * 4);
            if (r() < 0.72) { x = -HW + r() * W; z = HD; } else { x = HW; z = -HD + r() * D; }
            y = hz(G, sheet, x, z) + gauss() * 0.01;
          } else if (q < 0.86) {
            const qs = r();
            sheet = qs < 0.18 ? 1 : qs < 0.58 ? 2 : qs < 0.8 ? 3 : 4;
            x = -HW + r() * W; z = -HD + r() * D; y = hz(G, sheet, x, z) + gauss() * 0.018;
          } else {
            x = -HW + r() * W; z = -HD + r() * D;
            const top = hz(G, 2, x, z), bot = hz(G, 3, x, z);
            y = bot + r() * (top - bot);
          }
          const L = sheet >= 0 ? Math.min(sheet, 4) : layerOf(x, y, z);
          let base = LAYER_COL[L];
          if (L === 2 && y < OWC) base = WATER_COL;
          if (sheet === 2) base = hz(G, 2, x, z) > OWC + 0.05 ? [0.95, 0.66, 0.30] : [0.40, 0.52, 0.58];
          if (sheet === 3) base = [0.36, 0.42, 0.50];
          if (trace) base = sheet === 2 ? [1.0, 0.74, 0.40] : [0.62, 0.70, 0.80];
          const topL = hz(G, L, x, z);
          const sl = Math.sin((topL - y) * 46 + x * 0.3 + L * 2.1);
          const lam = 0.45 + 0.55 * sl * sl;
          const grain = 0.5 + 0.5 * r();
          const k = (trace ? 1.5 : sheet >= 0 ? 1.15 : 1.3 * lam) * grain * LAYER_GAIN[L];
          col[n * 3] = base[0] * k; col[n * 3 + 1] = base[1] * k; col[n * 3 + 2] = base[2] * k;
          pos[n * 3] = x; pos[n * 3 + 1] = y; pos[n * 3 + 2] = z;
          const [i, j, kk] = cellIndex(x, y, z);
          const cc = cellCentre(i, j, kk);
          cell.set(cc, n * 3);
          seed[n * 2] = r(); seed[n * 2 + 1] = r();
        }
        setGeo(strata, { position: [pos, 3], aCol: [col, 3], aCell: [cell, 3], aSeed: [seed, 2] });
      }

      // ----- streamlines + flow particles -----
      const lines = [];
      {
        const r = rng(202);
        for (let l = 0; l < C.lines; l++) lines.push({ th: ((l + 0.15 + r() * 0.7) / C.lines) * Math.PI * 2, sw: (r() - 0.5) * 1.3 });
        const Q = 72, segs = [];
        for (const L of lines) {
          let prev = null;
          for (let q = 0; q <= Q; q++) {
            const u = q / Q;
            const rr = R0 * Math.sqrt(1 - u * 0.985);
            const th = L.th + L.sw * (1 - rr / R0);
            const x = Math.cos(th) * HW * rr, z = Math.sin(th) * HD * rr;
            const y = 0.5 * (hz(G, 2, x, z) + hz(G, 3, x, z));
            const ins = Math.abs(x) < HW && Math.abs(z) < HD;
            const cur = ins ? [x, y, z, 1 - rr / R0] : null;
            if (prev && cur) segs.push(prev, cur);
            prev = cur;
          }
        }
        const pos = new Float32Array(segs.length * 3), au = new Float32Array(segs.length);
        segs.forEach((s, i) => { pos[i * 3] = s[0]; pos[i * 3 + 1] = s[1]; pos[i * 3 + 2] = s[2]; au[i] = s[3]; });
        setGeo(stream, { position: [pos, 3], aU: [au, 1] });

        const N = C.flow;
        const path = new Float32Array(N * 4), ph = new Float32Array(N * 2), dummy = new Float32Array(N * 3);
        for (let n = 0; n < N; n++) {
          const L = lines[Math.floor(r() * lines.length)];
          path[n * 4] = L.th + (r() - 0.5) * 0.05;
          path[n * 4 + 1] = L.sw;
          path[n * 4 + 2] = 0.5 + (r() - 0.5) * 0.6;
          path[n * 4 + 3] = 0.07 + r() * 0.06;
          ph[n * 2] = r(); ph[n * 2 + 1] = r();
        }
        setGeo(flow, { position: [dummy, 3], aPath: [path, 4], aPh: [ph, 2] });
      }

      // ----- corner-point grid -----
      {
        const segs = [];
        const poly = (pts, wgt) => { for (let i = 0; i + 1 < pts.length; i++) segs.push([...pts[i], wgt], [...pts[i + 1], wgt]); };
        const sub = 3;
        // full i/j grids on the reservoir surfaces
        for (const [kk, wgt] of [[2, 1.0], [3, 0.45], [4, 0.75]]) {
          for (let i = 0; i <= NI; i++) {
            const x = -HW + i * cw, pts = [];
            for (let q = 0; q <= NJ * sub; q++) { const z = -HD + (q / (NJ * sub)) * D; pts.push([x, gs(G, kk, x, z), z]); }
            poly(pts, wgt);
          }
          for (let j = 0; j <= NJ; j++) {
            const z = -HD + j * cd, pts = [];
            for (let q = 0; q <= NI * sub; q++) { const x = -HW + (q / (NI * sub)) * W; pts.push([x, gs(G, kk, x, z), z]); }
            poly(pts, wgt);
          }
        }
        // other surfaces: outline on the four faces
        for (const kk of [0, 1, 5, 6]) {
          for (const z of [-HD, HD]) { const pts = []; for (let q = 0; q <= NI * sub; q++) { const x = -HW + (q / (NI * sub)) * W; pts.push([x, gs(G, kk, x, z), z]); } poly(pts, z > 0 ? 0.7 : 0.3); }
          for (const x of [-HW, HW]) { const pts = []; for (let q = 0; q <= NJ * sub; q++) { const z = -HD + (q / (NJ * sub)) * D; pts.push([x, gs(G, kk, x, z), z]); } poly(pts, x > 0 ? 0.7 : 0.3); }
        }
        // pillars on the front and right faces (full height), and inside the reservoir
        for (let i = 0; i <= NI; i++) { const x = -HW + i * cw; poly([[x, gs(G, 6, x, HD), HD], [x, gs(G, 0, x, HD), HD]], 0.38); }
        for (let j = 0; j <= NJ; j++) { const z = -HD + j * cd; poly([[HW, gs(G, 6, HW, z), z], [HW, gs(G, 0, HW, z), z]], 0.38); }
        for (const [x, z] of [[-HW, -HD], [-HW, HD], [HW, -HD]]) poly([[x, gs(G, 6, x, z), z], [x, gs(G, 0, x, z), z]], 0.5);
        for (let i = 1; i < NI; i++) for (let j = 1; j < NJ; j++) {
          const x = -HW + i * cw, z = -HD + j * cd;
          poly([[x, gs(G, 4, x, z), z], [x, gs(G, 2, x, z), z]], 0.32);
        }
        const pos = new Float32Array(segs.length * 3), aw = new Float32Array(segs.length);
        segs.forEach((s, i) => { pos[i * 3] = s[0]; pos[i * 3 + 1] = s[1]; pos[i * 3 + 2] = s[2]; aw[i] = s[3]; });
        setGeo(grid, { position: [pos, 3], aW: [aw, 1] });
      }

      // ----- wellbore, glow sprites, pressure rings -----
      {
        const tip = hz(G, 2, 0, 0) - 0.62;
        wellMat.uniforms.uTip.value = tip; glowMat.uniforms.uTip.value = tip;
        const top = HZ[0].b + 1.9;
        wellMat.uniforms.uTop.value = top; glowMat.uniforms.uTop.value = top;
        const Q = 40, segs = [];
        for (const [ox, oz, wgt] of [[0, 0, 1.0], [-0.05, 0.03, 0.32], [0.05, -0.03, 0.32]]) {
          for (let q = 0; q < Q; q++) segs.push([ox, oz, q / Q, wgt], [ox, oz, (q + 1) / Q, wgt]);
        }
        const pos = new Float32Array(segs.length * 3), ay = new Float32Array(segs.length), aw = new Float32Array(segs.length);
        segs.forEach((s, i) => { pos[i * 3] = s[0]; pos[i * 3 + 2] = s[1]; ay[i] = s[2]; aw[i] = s[3]; });
        setGeo(well, { position: [pos, 3], aY: [ay, 1], aW: [aw, 1] });

        const r = rng(303), NP = 26;
        const gp = new Float32Array((NP + 1) * 3), ga = new Float32Array((NP + 1) * 4);
        ga.set([0, 34, 0, 0], 0);
        const yTop = hz(G, 2, 0, 0) - 0.08;
        for (let p = 1; p <= NP; p++) {
          const f = r(), a = p * 2.39996, rad = 0.06 + r() * 0.06;
          gp[p * 3] = Math.cos(a) * rad; gp[p * 3 + 1] = lerp(yTop, tip + 0.06, f); gp[p * 3 + 2] = Math.sin(a) * rad;
          ga.set([1, 5 + r() * 6, r(), f], p * 4);
        }
        setGeo(glow, { position: [gp, 3], aG: [ga, 4] });

        const RS = 120, rp = [], ra = [];
        for (let k = 0; k < 3; k++) for (let q = 0; q < RS; q++) {
          rp.push(0, 0, 0, 0, 0, 0);
          ra.push((q / RS) * Math.PI * 2, k, ((q + 1) / RS) * Math.PI * 2, k);
        }
        setGeo(rings, { position: [new Float32Array(rp), 3], aRing: [new Float32Array(ra), 2] });
      }

      // ----- nodes (cell centres) + k-NN network + name targets -----
      {
        const r = rng(404);
        const cellP = new Float32Array(NN * 3), jit = new Float32Array(NN * 3), meta = new Float32Array(NN * 4);
        const net = new Float32Array(NN * 3);
        let n = 0;
        for (let k = 0; k < NK; k++) for (let j = 0; j < NJ; j++) for (let i = 0; i < NI; i++) {
          const cc = cellCentre(i, j, k);
          const xc = cc[0], zc = cc[2];
          const th = gs(G, k, xc, zc) - gs(G, k + 1, xc, zc);
          cellP.set(cc, n * 3);
          jit[n * 3] = (r() - 0.5) * cw * 1.6;
          jit[n * 3 + 1] = (r() - 0.5) * th * 1.1;
          jit[n * 3 + 2] = (r() - 0.5) * cd * 1.6;
          for (let a = 0; a < 3; a++) net[n * 3 + a] = cellP[n * 3 + a] + jit[n * 3 + a];
          meta[n * 4] = r();
          meta[n * 4 + 2] = (xc + HW) / W;
          meta[n * 4 + 3] = k === 2 || k === 3 ? 1 : 0.35 + r() * 0.3;
          n++;
        }

        // targets on the shared glyphs, mapped to the z = 0 plane of the canonical pose
        const L = c.layout;
        const smp = sampleName(L, NN, { which: 'short', mode: 'mix', seed: 11 }).positions;
        const toPlane = pixelToPlane(camera, CAM_Z, w, h, 0);
        const v = new THREE.Vector3();
        const tg = new Float32Array(NN * 3);
        for (let q = 0; q < NN; q++) { toPlane(smp[q * 2], smp[q * 2 + 1], v); tg[q * 3] = v.x / S; tg[q * 3 + 1] = v.y / S; tg[q * 3 + 2] = 0; }
        // coherent assignment: columns by x, rows by y within each column
        const ni = Array.from({ length: NN }, (_, q) => q).sort((a, b) => net[a * 3] - net[b * 3]);
        const ti = Array.from({ length: NN }, (_, q) => q).sort((a, b) => tg[a * 3] - tg[b * 3]);
        const chunk = Math.max(8, Math.round(NN / (NI * 1.5)));
        const tgt = new Float32Array(NN * 3), scat = new Float32Array(NN * 3);
        const nameL = tg[ti[0] * 3], nameR = tg[ti[NN - 1] * 3];
        for (let s0 = 0; s0 < NN; s0 += chunk) {
          const a = ni.slice(s0, s0 + chunk).sort((p, q) => net[q * 3 + 1] - net[p * 3 + 1]);
          const b = ti.slice(s0, s0 + chunk).sort((p, q) => tg[q * 3 + 1] - tg[p * 3 + 1]);
          for (let m = 0; m < a.length; m++) {
            const nd = a[m], tq = b[m];
            tgt.set([tg[tq * 3], tg[tq * 3 + 1], 0], nd * 3);
            meta[nd * 4 + 1] = clamp(0.55 * (tg[tq * 3] - nameL) / (nameR - nameL || 1) + 0.45 * r());
            const ang = Math.atan2(tg[tq * 3 + 1] + (r() - 0.5) * 2, tg[tq * 3] * 0.6 + (r() - 0.5) * 2);
            const mag = (1.6 + r() * 3.2) / S;
            scat[nd * 3] = Math.cos(ang) * mag; scat[nd * 3 + 1] = Math.sin(ang) * mag * 0.8; scat[nd * 3 + 2] = (r() - 0.6) * 4 / S;
          }
        }
        setGeo(nodes, { position: [new Float32Array(NN * 3), 3], aCell: [cellP, 3], aJit: [jit, 3], aTgt: [tgt, 3], aScat: [scat, 3], aMeta: [meta, 4] });

        // k-NN (k = 3) on the loosened positions, plus a few long links
        const K = 3, key = new Set(), E = [];
        const bd = new Float32Array(K), bi = new Int32Array(K);
        for (let a = 0; a < NN; a++) {
          bd.fill(1e9); bi.fill(-1);
          const ax = net[a * 3], ay = net[a * 3 + 1], az = net[a * 3 + 2];
          for (let b = 0; b < NN; b++) {
            if (b === a) continue;
            const dx = net[b * 3] - ax, dy = (net[b * 3 + 1] - ay) * 1.6, dz = net[b * 3 + 2] - az;
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < bd[K - 1]) {
              let m = K - 1;
              while (m > 0 && bd[m - 1] > d2) { bd[m] = bd[m - 1]; bi[m] = bi[m - 1]; m--; }
              bd[m] = d2; bi[m] = b;
            }
          }
          for (let m = 0; m < K; m++) {
            const b = bi[m]; if (b < 0) continue;
            const kk = a < b ? a * NN + b : b * NN + a;
            if (!key.has(kk)) { key.add(kk); E.push([a, b, 0]); }
          }
        }
        const longN = Math.round(NN * 0.05), cs = Math.max(cw, cd);
        for (let q = 0; q < longN; q++) {
          const a = Math.floor(r() * NN);
          for (let tries = 0; tries < 30; tries++) {
            const b = Math.floor(r() * NN);
            const dx = net[b * 3] - net[a * 3], dy = net[b * 3 + 1] - net[a * 3 + 1], dz = net[b * 3 + 2] - net[a * 3 + 2];
            const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (d > cs * 2.5 && d < cs * 5.5) { E.push([a, b, 1]); break; }
          }
        }
        const NV = E.length * 2;
        const A = {
          aAc: new Float32Array(NV * 3), aAj: new Float32Array(NV * 3), aAt: new Float32Array(NV * 3), aAm: new Float32Array(NV * 4),
          aBc: new Float32Array(NV * 3), aBj: new Float32Array(NV * 3), aBt: new Float32Array(NV * 3), aBm: new Float32Array(NV * 4),
          aE: new Float32Array(NV * 4),
        };
        const wellC = [0, hz(G, 2, 0, 0) - 0.3, 0];
        let maxR = 0;
        const rad = E.map(([a, b]) => {
          const mx = (net[a * 3] + net[b * 3]) / 2 - wellC[0], my = ((net[a * 3 + 1] + net[b * 3 + 1]) / 2 - wellC[1]) * 1.5, mz = (net[a * 3 + 2] + net[b * 3 + 2]) / 2 - wellC[2];
          const d = Math.sqrt(mx * mx + my * my + mz * mz); maxR = Math.max(maxR, d); return d;
        });
        E.forEach(([a, b, kind], e) => {
          const order = clamp(0.8 * rad[e] / maxR + 0.2 * r());
          const rnd = r();
          for (let end = 0; end < 2; end++) {
            const vtx = e * 2 + end;
            for (let c3 = 0; c3 < 3; c3++) {
              A.aAc[vtx * 3 + c3] = cellP[a * 3 + c3]; A.aAj[vtx * 3 + c3] = jit[a * 3 + c3]; A.aAt[vtx * 3 + c3] = tgt[a * 3 + c3];
              A.aBc[vtx * 3 + c3] = cellP[b * 3 + c3]; A.aBj[vtx * 3 + c3] = jit[b * 3 + c3]; A.aBt[vtx * 3 + c3] = tgt[b * 3 + c3];
            }
            for (let c4 = 0; c4 < 4; c4++) { A.aAm[vtx * 4 + c4] = meta[a * 4 + c4]; A.aBm[vtx * 4 + c4] = meta[b * 4 + c4]; }
            A.aE.set([end, order, rnd, kind], vtx * 4);
          }
        });
        setGeo(edges, {
          position: [new Float32Array(NV * 3), 3],
          aAc: [A.aAc, 3], aAj: [A.aAj, 3], aAt: [A.aAt, 3], aAm: [A.aAm, 4],
          aBc: [A.aBc, 3], aBj: [A.aBj, 3], aBt: [A.aBt, 3], aBm: [A.aBm, 4], aE: [A.aE, 4],
        });
        edgeCount = E.length;

        // brightness / size of the formed name from glyph area per node
        const ras = rasterizeName(L, { which: 'short', res: 0.5 });
        let ink = 0; for (let q = 0; q < ras.data.length; q++) if (ras.data[q] >= 128) ink++;
        const pxPerNode = ink / 0.25 / NN;
        nodeMat.uniforms.uNamePx.value = clamp(Math.sqrt(pxPerNode) * 1.35, 1.4, 4.2);
        nodeMat.uniforms.uNameA.value = clamp(0.55 + pxPerNode * 0.06, 0.55, 1.0);
      }

      // read phase: the strata return, face-on and dim, below the subtitle
      {
        const L = c.layout;
        const dFront = CAM_Z - HD * S;
        const pxPerUnit = h / (2 * dFront * Math.tan(THREE.MathUtils.degToRad(FOV / 2)));
        const targetPx = L.subtitle.top + (L.subtitle.narrow ? 96 : 56);
        readOff = ((h / 2 - targetPx) / pxPerUnit) / S - HZ[0].b;
      }

      strataMat.uniforms.uPx.value = lerp(1.25, 1.75, S);
      flowMat.uniforms.uPx.value = lerp(1.6, 2.1, S);
      nodeMat.uniforms.uPx.value = lerp(1.8, 2.6, S);

      renderer.setSize(w, h, false);
      if (composer) { composer.setSize(w, h); bloom.resolution.set(w, h); }
    }

    let edgeCount = 0, readOff = 0;
    build(ctx);

    // ---------- camera ----------
    const T = ctx.timing;
    const tgtV = new THREE.Vector3();
    function placeCamera(t) {
      let az, el, dist, ty;
      if (t <= NAME_AT) {
        az = hermiteKeys(CAM_KEYS, t, 1); el = hermiteKeys(CAM_KEYS, t, 2); dist = hermiteKeys(CAM_KEYS, t, 3); ty = hermiteKeys(CAM_KEYS, t, 4);
      } else {
        // hold the canonical pose, then (as the name leaves) return to the opening pose
        const k = easeInOutSine(ramp(t, T.out[0] + 0.25, T.out[1] - 0.3));
        const K0 = CAM_KEYS[0];
        az = K0[1] * k; el = K0[2] * k; dist = lerp(CAM_Z, K0[3], k); ty = K0[4] * k;
      }
      const a = THREE.MathUtils.degToRad(az), e = THREE.MathUtils.degToRad(el);
      tgtV.set(0, ty, 0);
      camera.position.set(Math.sin(a) * Math.cos(e) * dist, ty + Math.sin(e) * dist, Math.cos(a) * Math.cos(e) * dist);
      camera.lookAt(tgtV);
      camera.updateMatrixWorld();
      U.uCamL.value.copy(camera.position).divideScalar(S);
      U.uFogNear.value = dist - 2.5 * S;
      U.uFogFar.value = dist + 8 * S;
    }

    const nameA = NAME_AT;
    return {
      get stats() { return `${C.strata.toLocaleString()} strata pts · ${C.flow.toLocaleString()} flow · ${NN.toLocaleString()} nodes · ${edgeCount.toLocaleString()} edges`; },
      update(t) {
        U.uT.value = t;
        placeCamera(t);

        // 1. strata: scan sweep, dim under the grid, snap to cells; later a dim face-on
        //    echo during the read, then the opening scan line is rebuilt before the loop
        const SM = strataMat.uniforms;
        const rebuild0 = T.out[1] - 0.3;
        if (t < T.split[1]) {
          SM.uScan.value = lerp(0.0, 1.14, easeInOutSine(ramp(t, 0.0, 2.45)));
          SM.uAlpha.value = 1 - 0.3 * ramp(t, 5.2, 6.4);
          SM.uSnap.value = ramp(t, 6.85, 8.3);
          SM.uGlintA.value = 0;
          strata.position.y = 0;
        } else if (t < rebuild0) {
          SM.uScan.value = 1.14;
          SM.uSnap.value = 0;
          SM.uAlpha.value = 0.27 * easeInOutSine(ramp(t, T.read[0] - 0.8, T.read[0] + 1.2)) * (1 - easeInOutSine(ramp(t, T.out[0] - 0.2, T.out[0] + 0.5)));
          SM.uGlint.value = lerp(-0.25, 1.25, ramp(t, T.read[0] - 0.8, T.out[0]));
          SM.uGlintA.value = 0.9;
          strata.position.y = readOff;
        } else {
          SM.uScan.value = 0.0;
          SM.uSnap.value = 0;
          SM.uAlpha.value = easeInOutSine(ramp(t, rebuild0, T.duration));
          SM.uGlintA.value = 0;
          strata.position.y = 0;
        }

        // 2. wellbore
        const grow = easeInOutCubic(ramp(t, 2.25, 3.45));
        const wellOut = 1 - ramp(t, 7.4, 8.6);
        wellMat.uniforms.uGrow.value = grow; glowMat.uniforms.uGrow.value = grow;
        wellMat.uniforms.uAlpha.value = ramp(t, 2.2, 2.5) * wellOut * 0.9;
        glowMat.uniforms.uHead.value = ramp(t, 2.2, 2.45) * (1 - 0.55 * ramp(t, 3.5, 4.6)) * wellOut;
        glowMat.uniforms.uPerf.value = easeOutCubic(ramp(t, 3.35, 3.8)) * (1 - 0.5 * ramp(t, 4.4, 5.4)) * wellOut;

        // 3. drawdown rings + flow
        ringMat.uniforms.uPhase.value = (t - 3.4) * 0.42;
        ringMat.uniforms.uAlpha.value = ramp(t, 3.4, 3.9) * (1 - ramp(t, 5.4, 6.6));
        streamMat.uniforms.uAlpha.value = ramp(t, 3.6, 4.6) * (1 - 0.35 * ramp(t, 5.2, 6.2)) * (1 - ramp(t, 6.9, 8.0));
        flowMat.uniforms.uAlpha.value = ramp(t, 3.7, 4.5) * (1 - ramp(t, 7.0, 8.1));

        // 4. grid
        gridMat.uniforms.uSweep.value = lerp(-0.05, 1.14, easeInOutSine(ramp(t, 5.2, 6.75)));
        gridMat.uniforms.uAlpha.value = ramp(t, 5.15, 5.3) * (1 - 0.45 * ramp(t, 7.2, 8.0)) * (1 - ramp(t, 8.2, 9.1));

        // 5-7. nodes, network, morph to name, hand-off, quiet read, return
        nodeMat.uniforms.uIn.value = ramp(t, 7.05, 8.45);
        U.uLoose.value = easeInOutSine(ramp(t, 8.1, 9.5));
        U.uMorph.value = ramp(t, 9.25, nameA - 0.12);
        edgeMat.uniforms.uGrow.value = ramp(t, 8.15, 9.6);
        edgeMat.uniforms.uAlpha.value = ramp(t, 8.15, 8.4);
        const disp = easeInOutCubic(ramp(t, nameA + 0.3, T.split[0] + 0.4));
        U.uDisperse.value = disp + 0.08 * Math.max(0, t - T.split[0]);
        U.uDrift.value = disp;
        const fadeName = ramp(t, nameA + 0.25, T.split[0] - 0.05);
        const readLevel = 0.16 * (1 - ramp(t, T.out[0], T.out[1]));
        nodeMat.uniforms.uAlpha.value = lerp(1, readLevel, easeInOutSine(fadeName));

        if (composer) composer.render(); else renderer.render(scene, camera);
      },
      resize(c) {
        build(c);
      },
      dispose() {
        for (const o of objs) o.geometry.dispose();
        for (const m of [strataMat, flowMat, streamMat, gridMat, wellMat, glowMat, ringMat, nodeMat, edgeMat, bgMat]) m.dispose();
        bgGeo.dispose();
        if (composer) { bloom.dispose(); composer.dispose(); }
        renderer.dispose();
        renderer.domElement.remove();
      },
    };
  },
};
