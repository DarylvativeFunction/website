// a-reservoir.js - Prototype A, "Cinematic Reservoir".
//
// A lit cut-away block of an anticline trap, rendered as a seismic-style point
// cloud, is already on screen at t = 0 (a few amber particles drift inside the
// reservoir layer). One continuous move follows: a corner-point grid sweeps out
// from the crest -> its cell centres become nodes while the strata fall away ->
// k-NN edges grow out of the lattice -> the nodes fly onto the shared
// "Daryl Tan" glyphs as the camera settles into its canonical pose.
//
// Everything is a pure function of t: geometry is built once per size, every
// motion is a shader morph driven by uniforms computed from t.

import * as THREE from 'three';
import { EffectComposer, RenderPass, UnrealBloomPass, OutputPass } from 'three/addons';
import { createRenderer, pixelToPlane, ramp, clamp, lerp, easeInOutCubic, easeInOutSine, rng } from '../core/stage.js';
import { sampleName, rasterizeName } from '../core/text.js';

const NAME_AT = 7.0;
const DURATION = 15.7;
const FOV = 32;
const CAM_Z = 20;

// Ambient fluid drifting inside the reservoir layer. Set to false to switch it off.
const AMBIENT_FLOW = true;
const FLOW_CYCLES = 6;   // particle lifecycles per loop (integer keeps the loop seamless)
const NAME_REPLICAS = 3; // particles per network node in the formed name (denser name)
const READ_ALPHA = 0.27; // strata level behind the text during the read phase

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
  float radial(vec3 p) { return length(p.xz / uHalf) / 1.4142; }
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
// portrait: 0 on landscape, 1 on a phone held upright
const portraitOf = (aspect) => clamp((1 - aspect) / 0.55);
function makeGeo(aspect) {
  const p = portraitOf(aspect);
  const W = lerp(clamp(aspect * 7.4, 7.6, 12), 6.8, p);
  const D = lerp(clamp(aspect * 4.4, 5.6, 7), 6.4, p);
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

// camera (landscape): t, azimuth (deg), elevation (deg), distance, target y
const CAM_KEYS = [
  [0.0, 37, 23, 20.0, -0.5],
  [1.8, 32, 25, 19.0, -0.45],
  [3.6, 24, 25, 19.2, -0.3],
  [5.3, 16, 18, 19.8, -0.1],
  [NAME_AT - 0.12, 0, 0, CAM_Z, 0],
];
function camKeysFor(aspect) {
  const p = portraitOf(aspect);
  return CAM_KEYS.map((k, i) => i === CAM_KEYS.length - 1 ? k.slice()
    : [k[0], k[1] * (1 - 0.25 * p), k[2] + 13 * p, k[3] * (1 - 0.08 * p), k[4] + 0.3 * p]);
}

export default {
  name: 'Cinematic Reservoir (previous cut)',
  blurb: 'A lit cut-away anticline trap, oil column glowing amber. A corner-point grid sweeps out from the crest, its cells become a k-NN network, and the network converges into the name.',
  tech: 'Three.js · GPU point morphs · corner-point grid · k-NN graph · bloom',
  duration: DURATION,
  nameAt: NAME_AT,
  nameWeight: 700,
  phases: [
    { label: 'Reservoir', start: 0, end: 1.8 },
    { label: 'Grid', start: 1.8, end: 3.6 },
    { label: 'Network', start: 3.6, end: 5.4 },
    { label: 'Converge', start: 5.4, end: NAME_AT },
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
  posterTime: 12.0,

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
      ? { strata: 44000, flow: 2600, NI: 28, NJ: 16 }
      : { strata: 15000, flow: 900, NI: 16, NJ: 9 };
    const NK = 6;
    const NN = C.NI * C.NJ * NK;
    const NR = NN * NAME_REPLICAS;

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
      uniforms: { uAspect: { value: ctx.width / ctx.height } },
      vertexShader: `varying vec2 vUv; void main() { vUv = position.xy * 0.5 + 0.5; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
      fragmentShader: `
        uniform float uAspect; varying vec2 vUv;
        void main() {
          vec2 d = (vUv - vec2(0.5, 0.56)) * vec2(uAspect, 1.0);
          float g = smoothstep(1.15, 0.0, length(d));
          gl_FragColor = vec4(mix(vec3(0.010, 0.012, 0.016), vec3(0.030, 0.038, 0.048), g), 1.0);
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
      uniforms: shared({ uAlpha: { value: 1 }, uSnap: { value: 0 }, uPx: { value: 1.7 }, uGlint: { value: -1 }, uGlintA: { value: 0 } }),
      vertexShader: `
        attribute vec3 aCol; attribute vec3 aCell; attribute vec2 aSeed;
        uniform float uAlpha, uSnap, uPx, uDpr, uCamZ, uGlint, uGlintA;
        varying vec3 vCol; varying float vA;
        ${GLSL_GEO} ${GLSL_FOG}
        void main() {
          vec3 p = position;
          float xn = (p.x + uHalf.x) / (2.0 * uHalf.x);
          float dg = uGlint - xn;
          float glint = exp(-dg * dg / 0.006) * uGlintA;
          float ff = faceFade(p);
          // the continuum slides into its grid cell and falls away, outward from the crest
          float s = clamp(uSnap * 1.5 - radial(p) * 0.4 - aSeed.x * 0.1, 0.0, 1.0);
          float es = s * s * (3.0 - 2.0 * s);
          p = mix(p, aCell, es);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          float depth = -mv.z;
          float light = 0.78 + 0.34 * smoothstep(-1.0, 1.0, (-p.x * 0.10 + p.y * 0.42 + p.z * 0.14));
          vA = 1.3 * uDens * uAlpha * (1.0 + glint) * fogOf(depth) * (1.0 - smoothstep(0.35, 0.9, s)) * ff;
          vCol = aCol * light + vec3(0.55, 0.85, 1.0) * glint * 0.35 + vec3(0.4, 0.7, 1.0) * es * 0.25;
          gl_PointSize = uPx * uDpr * (uCamZ / depth) * (0.7 + 0.6 * aSeed.y);
        }`,
      fragmentShader: `
        varying vec3 vCol; varying float vA; ${SOFT_DOT}
        void main() { float a = softDot(gl_PointCoord) * vA; if (a < 0.003) discard; gl_FragColor = vec4(vCol * a, a); }`,
    });

    // ambient fluid: short layer-parallel drifts inside the reservoir, respawned at
    // hashed positions FLOW_CYCLES times per loop (loop-exact), each with a faint trail
    const flowMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uAlpha: { value: 0 }, uPx: { value: 2.0 }, uK: { value: FLOW_CYCLES / DURATION }, uLen: { value: 0.9 } }),
      vertexShader: `
        attribute vec4 aF; // seed, layer fraction, phase offset, trail index
        uniform float uT, uAlpha, uPx, uDpr, uCamZ, uK, uLen;
        varying float vA;
        ${GLSL_GEO} ${GLSL_FOG}
        float h11(float n) { return fract(sin(n * 12.9898) * 43758.5453); }
        void main() {
          float ph = uT * uK + aF.z;
          float c = floor(ph), u = fract(ph) - aF.w * 0.018;
          float sx = h11(c * 1.731 + aF.x * 97.13), sz = h11(c * 2.917 + aF.x * 41.77 + 0.5);
          vec2 xz0 = vec2((sx * 2.0 - 1.0) * uHalf.x, (sz * 2.0 - 1.0) * uHalf.y);
          vec2 dir = normalize(vec2(1.0, 0.35 * sin(xz0.y * 0.7 + xz0.x * 0.25)));
          vec2 xz = xz0 + dir * (u - 0.5) * uLen;
          float y = mix(resBase(xz), resTop(xz), aF.y);
          vec3 p = vec3(xz.x, y, xz.y);
          float inside = step(abs(xz.x), uHalf.x * 0.99) * step(abs(xz.y), uHalf.y * 0.99);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          float life = sin(3.14159 * clamp(u, 0.0, 1.0));
          vA = uDens * uAlpha * inside * life * (1.0 - aF.w * 0.38) * fogOf(-mv.z);
          gl_PointSize = uPx * uDpr * (uCamZ / -mv.z) * (1.0 - aF.w * 0.22);
        }`,
      fragmentShader: `
        varying float vA; ${SOFT_DOT}
        void main() {
          float a = softDot(gl_PointCoord) * vA; if (a < 0.003) discard;
          gl_FragColor = vec4(vec3(0.95, 0.68, 0.30) * a, a);
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
          float d = uSweep - radial(p);
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

    const nodeMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uIn: { value: 0 }, uAlpha: { value: 0 }, uNameA: { value: 0.8 }, uPx: { value: 2.4 }, uNamePx: { value: 3 } }),
      vertexShader: `
        attribute vec3 aCell, aJit, aTgt, aScat; attribute vec4 aMeta; attribute float aRep;
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
          // replicas ride hidden inside their node and peel off into the glyphs
          float rep = aRep > 0.5 ? smoothstep(0.35, 0.9, e) : 1.0;
          vA = uAlpha * rep * smoothstep(0.0, 0.6, a0) * fog * mix((0.7 + 0.5 * aMeta.w) * uDens, uNameA, e);
          vE = e;
          float px = mix(uPx * (0.8 + 0.5 * aMeta.w), uNamePx, e);
          gl_PointSize = px * uDpr * (uCamZ / -mv.z) * (1.0 + vFlash * 0.35);
        }`,
      fragmentShader: `
        varying float vA; varying float vE; varying float vFlash; ${SOFT_DOT}
        void main() {
          float a = softDot(gl_PointCoord) * vA; if (a < 0.003) discard;
          vec3 c = mix(vec3(0.70, 0.90, 1.0), vec3(0.97, 0.97, 0.96), vE) + vec3(0.1, 0.2, 0.25) * vFlash;
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
          vA = uDens * uAlpha * step(0.001, g) * (1.0 - smoothstep(0.0, 0.5, arrive)) * fogOf(-mv.z) * (aE.w > 0.5 ? 0.45 : 1.0);
          vS = aE.x * g;
          vPulse = aE.z;
        }`,
      fragmentShader: `
        uniform float uT; varying float vA; varying float vS; varying float vPulse;
        void main() {
          float ph = fract(uT * 0.55 + vPulse * 7.0);
          float pulse = step(vPulse, 0.1) * exp(-pow((vS - ph) / 0.05, 2.0));
          float a = vA * (0.1 + pulse * 0.7);
          vec3 c = vec3(0.36, 0.78, 1.0) * vA * 0.1 + vec3(0.95, 0.68, 0.30) * vA * pulse * 0.7;
          gl_FragColor = vec4(c, a);
        }`,
    });

    // ---------- objects (geometry filled in build) ----------
    const strata = new THREE.Points(new THREE.BufferGeometry(), strataMat);
    const flow = new THREE.Points(new THREE.BufferGeometry(), flowMat);
    const grid = new THREE.LineSegments(new THREE.BufferGeometry(), gridMat);
    const edges = new THREE.LineSegments(new THREE.BufferGeometry(), edgeMat);
    const nodes = new THREE.Points(new THREE.BufferGeometry(), nodeMat);
    const objs = [strata, flow, grid, edges, nodes];
    objs.forEach((o, i) => { o.frustumCulled = false; o.renderOrder = i; world.add(o); });
    flow.visible = AMBIENT_FLOW;

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
    let camKeys = camKeysFor(ctx.width / ctx.height);
    let edgeCount = 0, readOff = 0;

    const setGeo = (o, attrs) => {
      o.geometry.dispose();
      const g = new THREE.BufferGeometry();
      for (const k in attrs) g.setAttribute(k, new THREE.BufferAttribute(attrs[k][0], attrs[k][1]));
      g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e4);
      o.geometry = g;
    };

    function build(c) {
      const w = c.width, h = c.height, aspect = w / h;
      const portrait = portraitOf(aspect);
      G = makeGeo(aspect);
      camKeys = camKeysFor(aspect);
      const { W, D } = G, HW = W / 2, HD = D / 2;
      U.uHalf.value.set(HW, HD);
      U.uDome.value.set(G.AX, G.AZ);
      camera.aspect = aspect;
      camera.updateProjectionMatrix();
      // fit the block's opening 3/4 silhouette to the width
      const d0 = camKeys[0][3];
      const visW = 2 * d0 * Math.tan(THREE.MathUtils.degToRad(FOV / 2)) * aspect;
      const az0 = THREE.MathUtils.degToRad(camKeys[0][1]);
      const proj = W * Math.cos(az0) + D * Math.sin(az0);
      S = Math.min(1, (visW * lerp(0.9, 0.84, portrait)) / proj);
      world.scale.setScalar(S);
      U.uS.value = S;
      U.uDens.value = clamp(Math.pow(S, 1.5) * 1.15, 0.35, 1);
      if (bloom) bloom.strength = 0.55 * lerp(0.55, 1, S);
      bgMat.uniforms.uAspect.value = aspect;

      const NI = C.NI, NJ = C.NJ;
      const cw = W / NI, cd = D / NJ;
      const cellIndex = (x, y, z) => {
        const i = clamp(Math.floor((x + HW) / cw), 0, NI - 1);
        const j = clamp(Math.floor((z + HD) / cd), 0, NJ - 1);
        let k = NK - 1;
        for (let kk = 0; kk < NK; kk++) if (y >= gs(G, kk + 1, x, z)) { k = kk; break; }
        return [i, j, k];
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
        const faceY = (fx, fz) => {
          // layer-weighted density on the cut faces: tight rocks sparse, sandstone dense
          for (let tries = 0; tries < 6; tries++) {
            const yy = HZ[5].b + r() * (HZ[0].b - HZ[5].b);
            if (r() < LAYER_DENS[layerOf(fx, yy, fz)]) return yy;
          }
          return HZ[5].b + r() * (HZ[0].b - HZ[5].b);
        };
        for (let n = 0; n < N; n++) {
          const q = r();
          let x, y, z, sheet = -1, trace = false;
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
          cell.set(cellCentre(i, j, kk), n * 3);
          seed[n * 2] = r(); seed[n * 2 + 1] = r();
        }
        setGeo(strata, { position: [pos, 3], aCol: [col, 3], aCell: [cell, 3], aSeed: [seed, 2] });
      }

      // ----- ambient flow particles (3 trail samples each) -----
      {
        const r = rng(202), N = C.flow, TR = 3;
        const f = new Float32Array(N * TR * 4);
        for (let n = 0; n < N; n++) {
          const sd = r(), lf = 0.2 + r() * 0.6, off = r();
          for (let k = 0; k < TR; k++) f.set([sd, lf, off, k], (n * TR + k) * 4);
        }
        setGeo(flow, { position: [new Float32Array(N * TR * 3), 3], aF: [f, 4] });
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

      // ----- nodes (cell centres) + replicas + k-NN network + name targets -----
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
          meta[n * 4 + 2] = Math.min(1, Math.hypot(xc / HW, zc / HD) / Math.SQRT2); // appear outward from the crest
          meta[n * 4 + 3] = k === 2 || k === 3 ? 1 : 0.35 + r() * 0.3;
          n++;
        }

        // targets on the shared glyphs, mapped to the z = 0 plane of the canonical pose
        const L = c.layout;
        const smp = sampleName(L, NR, { which: 'short', mode: 'mix', seed: 11 }).positions;
        const toPlane = pixelToPlane(camera, CAM_Z, w, h, 0);
        const v = new THREE.Vector3();
        const tg = new Float32Array(NR * 3);
        for (let q = 0; q < NR; q++) { toPlane(smp[q * 2], smp[q * 2 + 1], v); tg[q * 3] = v.x / S; tg[q * 3 + 1] = v.y / S; }
        // coherent assignment of NR particles (each node NAME_REPLICAS times):
        // columns by x, rows by y within each column
        const owner = (q) => q % NN;
        const ni = Array.from({ length: NR }, (_, q) => q).sort((a, b) => net[owner(a) * 3] - net[owner(b) * 3]);
        const ti = Array.from({ length: NR }, (_, q) => q).sort((a, b) => tg[a * 3] - tg[b * 3]);
        const chunk = Math.max(8, Math.round(NR / (NI * 1.5)));
        const pT = new Float32Array(NR * 3), pS = new Float32Array(NR * 3), pM = new Float32Array(NR * 4);
        const pC = new Float32Array(NR * 3), pJ = new Float32Array(NR * 3), pR = new Float32Array(NR);
        const nodeTgt = new Float32Array(NN * 3), nodeOrd = new Float32Array(NN).fill(-1);
        const nameL = tg[ti[0] * 3], nameR = tg[ti[NR - 1] * 3];
        for (let s0 = 0; s0 < NR; s0 += chunk) {
          const a = ni.slice(s0, s0 + chunk).sort((p, q) => net[owner(q) * 3 + 1] - net[owner(p) * 3 + 1]);
          const b = ti.slice(s0, s0 + chunk).sort((p, q) => tg[q * 3 + 1] - tg[p * 3 + 1]);
          for (let m = 0; m < a.length; m++) {
            const pq = a[m], nd = owner(pq), tq = b[m];
            const primary = pq < NN;
            pT[pq * 3] = tg[tq * 3]; pT[pq * 3 + 1] = tg[tq * 3 + 1];
            const ord = clamp(0.55 * (tg[tq * 3] - nameL) / (nameR - nameL || 1) + 0.45 * r());
            pM.set([meta[nd * 4], ord, meta[nd * 4 + 2], meta[nd * 4 + 3]], pq * 4);
            for (let c3 = 0; c3 < 3; c3++) { pC[pq * 3 + c3] = cellP[nd * 3 + c3]; pJ[pq * 3 + c3] = jit[nd * 3 + c3]; }
            pR[pq] = primary ? 0 : 1;
            if (primary) { nodeTgt[nd * 3] = tg[tq * 3]; nodeTgt[nd * 3 + 1] = tg[tq * 3 + 1]; nodeOrd[nd] = ord; }
            const ang = Math.atan2(tg[tq * 3 + 1] + (r() - 0.5) * 2, tg[tq * 3] * 0.6 + (r() - 0.5) * 2);
            const mag = (1.6 + r() * 3.2) / S;
            pS[pq * 3] = Math.cos(ang) * mag; pS[pq * 3 + 1] = Math.sin(ang) * mag * 0.8; pS[pq * 3 + 2] = (r() - 0.6) * 4 / S;
          }
        }
        setGeo(nodes, { position: [new Float32Array(NR * 3), 3], aCell: [pC, 3], aJit: [pJ, 3], aTgt: [pT, 3], aScat: [pS, 3], aMeta: [pM, 4], aRep: [pR, 1] });

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
        const crest = [0, hz(G, 2, 0, 0) - 0.3, 0];
        let maxR = 0;
        const rad = E.map(([a, b]) => {
          const mx = (net[a * 3] + net[b * 3]) / 2 - crest[0], my = ((net[a * 3 + 1] + net[b * 3 + 1]) / 2 - crest[1]) * 1.5, mz = (net[a * 3 + 2] + net[b * 3 + 2]) / 2 - crest[2];
          const d = Math.sqrt(mx * mx + my * my + mz * mz); maxR = Math.max(maxR, d); return d;
        });
        E.forEach(([a, b, kind], e) => {
          const order = clamp(0.8 * rad[e] / maxR + 0.2 * r());
          const rnd = r();
          for (let end = 0; end < 2; end++) {
            const vtx = e * 2 + end;
            for (let c3 = 0; c3 < 3; c3++) {
              A.aAc[vtx * 3 + c3] = cellP[a * 3 + c3]; A.aAj[vtx * 3 + c3] = jit[a * 3 + c3]; A.aAt[vtx * 3 + c3] = nodeTgt[a * 3 + c3];
              A.aBc[vtx * 3 + c3] = cellP[b * 3 + c3]; A.aBj[vtx * 3 + c3] = jit[b * 3 + c3]; A.aBt[vtx * 3 + c3] = nodeTgt[b * 3 + c3];
            }
            A.aAm.set([meta[a * 4], nodeOrd[a], meta[a * 4 + 2], meta[a * 4 + 3]], vtx * 4);
            A.aBm.set([meta[b * 4], nodeOrd[b], meta[b * 4 + 2], meta[b * 4 + 3]], vtx * 4);
            A.aE.set([end, order, rnd, kind], vtx * 4);
          }
        });
        setGeo(edges, {
          position: [new Float32Array(NV * 3), 3],
          aAc: [A.aAc, 3], aAj: [A.aAj, 3], aAt: [A.aAt, 3], aAm: [A.aAm, 4],
          aBc: [A.aBc, 3], aBj: [A.aBj, 3], aBt: [A.aBt, 3], aBm: [A.aBm, 4], aE: [A.aE, 4],
        });
        edgeCount = E.length;

        // brightness / size of the formed name from glyph area per particle
        const ras = rasterizeName(L, { which: 'short', res: 0.5 });
        let ink = 0; for (let q = 0; q < ras.data.length; q++) if (ras.data[q] >= 128) ink++;
        const pxPer = ink / 0.25 / NR;
        nodeMat.uniforms.uNamePx.value = clamp(Math.sqrt(pxPer) * 1.6, 1.8, 4.0);
        nodeMat.uniforms.uNameA.value = clamp(0.75 + pxPer * 0.06, 0.75, 1.0);
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
      flowMat.uniforms.uPx.value = lerp(1.7, 2.3, S);
      nodeMat.uniforms.uPx.value = lerp(2.8, 3.8, S);

      renderer.setSize(w, h, false);
      if (composer) { composer.setSize(w, h); bloom.resolution.set(w, h); }
    }

    build(ctx);

    // ---------- camera ----------
    const T = ctx.timing;
    const rebuild0 = T.out[1] - 0.3;
    const tgtV = new THREE.Vector3();
    function placeCamera(t) {
      let az, el, dist, ty;
      if (t <= NAME_AT) {
        az = hermiteKeys(camKeys, t, 1); el = hermiteKeys(camKeys, t, 2); dist = hermiteKeys(camKeys, t, 3); ty = hermiteKeys(camKeys, t, 4);
      } else {
        // hold the canonical pose; once the name is leaving, return to the opening pose
        const k = easeInOutSine(ramp(t, rebuild0, T.duration));
        const K0 = camKeys[0];
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

    return {
      get stats() { return `${C.strata.toLocaleString()} strata pts · ${AMBIENT_FLOW ? C.flow.toLocaleString() : 0} flow · ${NN.toLocaleString()} nodes (${NR.toLocaleString()} name pts) · ${edgeCount.toLocaleString()} edges`; },
      update(t) {
        U.uT.value = t;
        placeCamera(t);

        // strata: lit from t = 0; dim under the grid and fall into the cells; a dim
        // face-on echo during the read; then rise back into the opening pose
        const SM = strataMat.uniforms;
        let strataA;
        if (t < T.split[1]) {
          strataA = 1 - 0.3 * ramp(t, 1.8, 3.0);
          SM.uSnap.value = ramp(t, 2.9, 4.4);
          SM.uGlintA.value = 0;
          strata.position.y = 0;
        } else if (t < rebuild0) {
          SM.uSnap.value = 0;
          strataA = READ_ALPHA * easeInOutSine(ramp(t, T.read[0] - 0.8, T.read[0] + 1.2));
          SM.uGlint.value = lerp(-0.25, 1.25, ramp(t, T.read[0] - 0.8, rebuild0));
          SM.uGlintA.value = 0.9;
          strata.position.y = readOff;
        } else {
          const k = easeInOutSine(ramp(t, rebuild0, T.duration));
          SM.uSnap.value = 0;
          strataA = lerp(READ_ALPHA, 1, k);
          SM.uGlintA.value = 0;
          strata.position.y = readOff * (1 - k);
        }
        SM.uAlpha.value = strataA;

        // ambient fluid follows the strata, gone before the grid nodes take over
        flow.position.y = strata.position.y;
        flowMat.uniforms.uAlpha.value = AMBIENT_FLOW
          ? (t < T.split[1] ? 1 - ramp(t, 2.4, 3.4) : strataA) * 1.2
          : 0;

        // grid sweeps out from the crest
        gridMat.uniforms.uSweep.value = lerp(-0.05, 1.14, easeInOutSine(ramp(t, 1.75, 3.25)));
        gridMat.uniforms.uAlpha.value = ramp(t, 1.7, 1.85) * (1 - 0.4 * ramp(t, 3.4, 4.0)) * (1 - ramp(t, 3.9, 4.9));

        // nodes -> network -> name -> hand-off -> quiet read
        nodeMat.uniforms.uIn.value = ramp(t, 2.6, 3.8);
        U.uLoose.value = easeInOutSine(ramp(t, 3.5, 5.0));
        U.uMorph.value = ramp(t, 5.2, NAME_AT - 0.12);
        edgeMat.uniforms.uGrow.value = ramp(t, 3.45, 5.0);
        edgeMat.uniforms.uAlpha.value = ramp(t, 3.45, 3.65);
        const disp = easeInOutCubic(ramp(t, NAME_AT + 0.3, T.split[0] + 0.4));
        U.uDisperse.value = disp + 0.08 * Math.max(0, t - T.split[0]);
        U.uDrift.value = disp;
        const fadeName = ramp(t, NAME_AT + 0.25, T.split[0] - 0.05);
        const readLevel = 0.1 * (1 - ramp(t, T.out[0], T.out[1]));
        nodeMat.uniforms.uAlpha.value = lerp(1, readLevel, easeInOutSine(fadeName));

        if (composer) composer.render(); else renderer.render(scene, camera);
      },
      resize(c) {
        build(c);
      },
      dispose() {
        for (const o of objs) o.geometry.dispose();
        for (const m of [strataMat, flowMat, gridMat, nodeMat, edgeMat, bgMat]) m.dispose();
        bgGeo.dispose();
        if (composer) { bloom.dispose(); composer.dispose(); }
        renderer.dispose();
        renderer.domElement.remove();
      },
    };
  },
};
