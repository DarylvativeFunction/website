// a-reservoir.js - Prototype A, "Cinematic Reservoir" (premium art-direction pass).
//
// A lit cut-away block of an anticline trap, shot like a product film: graphite
// ground with an off-axis key light, filmic tone mapping, depth fog, a long lens
// with fake depth of field (rack focus front face -> crest -> network -> name),
// highlight-only bloom and fine grain. One continuous move: a corner-point grid
// sweeps out from the crest -> its cell centres become a k-NN network with a few
// labelled hubs -> the nodes stream (velocity-streaked) onto the shared
// "Daryl Tan" glyphs, left to right, as the camera settles into its canonical pose.
// A small editorial HUD (chapter index, readouts, coordinates) frames the film.
//
// Everything is a pure function of t: geometry is built once per size, every
// motion is a shader morph driven by uniforms computed from t.

import * as THREE from 'three';
import { EffectComposer, RenderPass, UnrealBloomPass, ShaderPass } from 'three/addons';
import { createRenderer, pixelToPlane, ramp, clamp, lerp, easeInOutCubic, easeInOutSine, rng } from '../core/stage.js';
import { sampleName, rasterizeName, MONO_FAMILY } from '../core/text.js';

const NAME_AT = 7.0;
const DURATION = 15.7;
const FOV = 27;          // longer lens, camera further back (telephoto compression)
const CAM_Z = 24;

const HUD = true;            // editorial overlay (chapter index, readouts, coordinates)
const AMBIENT_FLOW = true;   // amber fluid drifting inside the oil column
const FLOW_CYCLES = 6;       // particle lifecycles per loop (integer keeps the loop seamless)
const NAME_REPLICAS = 3;     // particles per network node in the formed name
const READ_ALPHA = 0.22;      // strata level behind the text during the read phase
const N_HUBS = 6;

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

// Linear HDR palette (tone-mapped at the end). Mostly graphite / silver-white.
const LAYER_COL = [
  [0.60, 0.64, 0.70],  // overburden
  [0.46, 0.51, 0.58],  // cap rock
  [0.66, 0.70, 0.75],  // reservoir sand (water leg)
  [0.42, 0.46, 0.52],  // shale
  [0.56, 0.53, 0.50],  // basement
];
const AMBER = [1.0, 0.56, 0.20];
const LAYER_DENS = [0.42, 0.22, 1.0, 0.36, 0.55];
const LAYER_GAIN = [0.5, 0.42, 0.7, 0.42, 0.5];

const HUB_LABELS = ['W-03 · φ 0.24', 'C-117 · k 412 mD', 'N-08 · Sw 0.31', 'W-11 · φ 0.19', 'C-42 · k 286 mD', 'N-21 · p 31.4 MPa'];

const GLSL_GEO = /* glsl */`
  uniform vec2 uHalf;  // block half extents (x, z)
  uniform vec2 uDome;  // anticline gaussian widths (x, z)
  float dome(vec2 p) { return exp(-(p.x * p.x / (2.0 * uDome.x * uDome.x) + p.y * p.y / (2.0 * uDome.y * uDome.y))); }
  float und(vec2 p, float k) { return 0.09 * sin(0.62 * p.x + 0.38 * p.y + 1.7 * k) + 0.05 * sin(1.35 * p.y - 0.45 * p.x + 0.9 * k); }
  float resTop(vec2 p) { return 0.02 + 0.98 * dome(p) + und(p, 2.0); }
  float resBase(vec2 p) { return -0.78 + 0.80 * dome(p) + 0.8 * und(p, 3.0); }
  uniform vec3 uCamL;  // camera position in block space
  uniform float uDens; // brightness compensation when the block is drawn small (narrow screens)
  uniform float uSideA; // right face / back edges (hidden in the face-on read echo)
  float sideOf(vec3 p) { return (abs(p.x - uHalf.x) < 1e-3 || abs(p.x + uHalf.x) < 1e-3 || abs(p.z + uHalf.y) < 1e-3) ? uSideA : 1.0; }
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

// exponential depth fog toward the ground (additive material simply fades), and lens
const GLSL_LENS = /* glsl */`
  uniform float uFogNear, uFogK, uFocus, uAperture, uCocMax, uDpr, uCamZ;
  float fogOf(float depth) { return exp(-uFogK * max(depth - uFogNear, 0.0)); }
  float cocOf(float depth) { return min(uAperture * abs(depth - uFocus) / depth, uCocMax); }
`;

// point sprite: crisp disc in focus, soft bokeh disc (faint rim) out of focus
const GLSL_SPRITE = /* glsl */`
  float sprite(float d, float blur) {
    float crisp = 1.0 - smoothstep(0.5, 1.0, d);
    float disc = (1.0 - smoothstep(0.78, 1.0, d)) * (0.78 + 0.22 * smoothstep(0.45, 0.95, d));
    return mix(crisp, disc, blur);
  }
`;

// shared node motion (nodes, edges, hub rings and HUD labels agree to the pixel)
const GLSL_NODE = /* glsl */`
  uniform float uT, uLoose, uDrift;
  vec3 nodePos(vec3 cell, vec3 jit, vec3 tgt, vec3 scat, vec4 meta, float morph, float disp, out float e) {
    float seed = meta.x;
    vec3 net = cell + jit * uLoose;
    net += vec3(sin(uT * 0.61 + seed * 31.0), sin(uT * 0.47 + seed * 17.0), cos(uT * 0.53 + seed * 23.0)) * 0.06 * uLoose;
    float m = clamp((morph - meta.y * 0.5) / 0.5, 0.0, 1.0);
    e = m * m * m * (m * (m * 6.0 - 15.0) + 10.0);
    vec3 p = mix(net, tgt, e);
    float arc = sin(3.14159 * e);
    p.z += arc * (0.5 + seed * 1.1);
    p.y += arc * (seed - 0.5) * 0.7;
    p += scat * disp;
    p += vec3(sin(uT * 0.21 + seed * 40.0), cos(uT * 0.17 + seed * 13.0), 0.0) * 0.35 * uDrift;
    return p;
  }
`;

// --- JS mirrors of the geometry functions -----------------------------------
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

// Cubic Hermite through keyframes. Tangents: finite differences inside, an explicit
// start tangent (so the loop never starts from a dead stop), flat at the final pose.
function keyTangent(keys, j, ch) {
  const n = keys.length;
  if (j >= n - 1) return 0;
  if (j === 0) return 0.7 * (keys[1][ch] - keys[0][ch]) / (keys[1][0] - keys[0][0]);
  return (keys[j + 1][ch] - keys[j - 1][ch]) / (keys[j + 1][0] - keys[j - 1][0]);
}
function hermite(p0, p1, m0, m1, u) {
  const u2 = u * u, u3 = u2 * u;
  return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1;
}
function hermiteKeys(keys, t, ch) {
  const n = keys.length;
  if (t <= keys[0][0]) return keys[0][ch];
  if (t >= keys[n - 1][0]) return keys[n - 1][ch];
  let i = 0;
  while (t > keys[i + 1][0]) i++;
  const h = keys[i + 1][0] - keys[i][0];
  return hermite(keys[i][ch], keys[i + 1][ch], keyTangent(keys, i, ch) * h, keyTangent(keys, i + 1, ch) * h, (t - keys[i][0]) / h);
}

// camera (landscape): t, azimuth (deg), elevation (deg), distance, target y
const CAM_KEYS = [
  [0.0, 37, 22, 24.2, -0.5],
  [1.8, 31, 25, 22.9, -0.42],
  [3.6, 23, 24, 23.0, -0.3],
  [5.3, 13, 14, 23.8, -0.1],
  [NAME_AT - 0.12, 0, 0, CAM_Z, 0],
];
function camKeysFor(aspect) {
  const p = portraitOf(aspect);
  return CAM_KEYS.map((k, i) => i === CAM_KEYS.length - 1 ? k.slice()
    : [k[0], k[1] * (1 - 0.25 * p), k[2] + 13 * p, k[3] * (1 - 0.08 * p), k[4] + 0.3 * p]);
}

const lin = (hex) => { const c = new THREE.Color(hex); return [c.r, c.g, c.b]; }; // sRGB hex -> linear

export default {
  name: 'Cinematic Reservoir',
  blurb: 'A lit cut-away anticline trap, shot like a product film. A corner-point grid sweeps out from the crest, its cells become a k-NN network with labelled hubs, and the network streams into the name.',
  tech: 'Three.js · filmic tone mapping · fake DOF · corner-point grid · k-NN graph · DOM HUD',
  duration: DURATION,
  nameAt: NAME_AT,
  nameWeight: 600,
  nameTracking: -0.035,
  phases: [
    { label: 'Reservoir', start: 0, end: 1.8 },
    { label: 'Grid', start: 1.8, end: 3.6 },
    { label: 'Network', start: 3.6, end: 5.4 },
    { label: 'Converge', start: 5.4, end: NAME_AT },
  ],
  reveal: {
    gradient: { name: ['#FFFFFF', '#AEB4BD'], yijin: ['#9AA1AB', '#5F6670'] },
    colors: {
      sub: 'rgba(235,238,242,0.62)',
      accent: '#B08A55',
      rule: 'rgba(235,238,242,0.22)',
    },
  },
  posterTime: 12.0,

  create(ctx) {
    const high = ctx.quality !== 'low';
    const renderer = createRenderer(ctx, { clearColor: 0x000000 });
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace; // the final pass encodes sRGB itself
    renderer.toneMapping = THREE.NoToneMapping;
    const dpr = renderer.getPixelRatio();
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(FOV, ctx.width / ctx.height, 0.1, 400);
    const world = new THREE.Group();
    scene.add(world);

    const C = high
      ? { strata: 28000, flow: 2200, NI: 28, NJ: 16, coc: 8, ap: 1 }
      : { strata: 11000, flow: 800, NI: 16, NJ: 9, coc: 3, ap: 0.4 };
    const NK = 6;
    const NN = C.NI * C.NJ * NK;
    const NR = NN * NAME_REPLICAS;

    // ---------- shared uniforms ----------
    const U = {
      uHalf: { value: new THREE.Vector2(6, 3.5) },
      uDome: { value: new THREE.Vector2(3.3, 2.5) },
      uFogNear: { value: 22 }, uFogK: { value: 0.08 },
      uFocus: { value: 22 }, uAperture: { value: 20 }, uCocMax: { value: C.coc * dpr },
      uT: { value: 0 }, uDpr: { value: dpr }, uCamZ: { value: CAM_Z },
      uCamL: { value: new THREE.Vector3(0, 0, CAM_Z) },
      uDens: { value: 1 }, uSideA: { value: 1 },
      uLoose: { value: 0 }, uDrift: { value: 0 },
      uMorph: { value: 0 }, uMorphPrev: { value: 0 }, uDisp: { value: 0 }, uDispPrev: { value: 0 },
      uRes: { value: new THREE.Vector2(ctx.width * dpr, ctx.height * dpr) },
    };
    const shared = (extra) => Object.assign({}, U, extra);
    // premultiplied additive: every shader writes (rgb * a, a)
    const additive = { transparent: true, depthWrite: false, depthTest: false, blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor };

    // ---------- materials ----------
    const strataMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uAlpha: { value: 1 }, uSnap: { value: 0 }, uPx: { value: 1.3 }, uGlint: { value: -1 }, uGlintA: { value: 0 } }),
      vertexShader: `
        attribute vec3 aCol; attribute vec3 aCell; attribute vec2 aSeed;
        uniform float uAlpha, uSnap, uPx, uGlint, uGlintA;
        varying vec3 vCol; varying float vA; varying float vBlur;
        ${GLSL_GEO} ${GLSL_LENS}
        void main() {
          vec3 p = position;
          float xn = (p.x + uHalf.x) / (2.0 * uHalf.x);
          float dg = uGlint - xn;
          float glint = exp(-dg * dg / 0.006) * uGlintA;
          float ff = faceFade(p) * sideOf(p);
          // the continuum slides into its grid cell and falls away, outward from the crest
          float s = clamp(uSnap * 1.5 - radial(p) * 0.4 - aSeed.x * 0.1, 0.0, 1.0);
          float es = s * s * (3.0 - 2.0 * s);
          p = mix(p, aCell, es);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          float depth = -mv.z;
          float base = uPx * uDpr * (uCamZ / depth) * (0.75 + 0.5 * aSeed.y);
          float coc = cocOf(depth);
          float size = base + coc;
          vBlur = coc / size;
          float energy = max((base * base) / (size * size), 0.05);
          float light = 0.8 + 0.3 * smoothstep(-1.0, 1.0, (-p.x * 0.10 + p.y * 0.42 + p.z * 0.14));
          vA = uDens * uAlpha * (1.0 + glint) * fogOf(depth) * (1.0 - smoothstep(0.35, 0.9, s)) * ff * energy;
          vCol = aCol * light + vec3(0.5, 0.75, 0.9) * glint * 0.3;
          gl_PointSize = size;
        }`,
      fragmentShader: `
        varying vec3 vCol; varying float vA; varying float vBlur; ${GLSL_SPRITE}
        void main() { float a = sprite(length(gl_PointCoord - 0.5) * 2.0, vBlur) * vA; if (a < 0.002) discard; gl_FragColor = vec4(vCol * a, a); }`,
    });

    // crisp hairlines along the horizons where they cut the faces (+ dashed oil-water contact)
    const hairMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uAlpha: { value: 1 }, uSnap: { value: 0 } }),
      vertexShader: `
        attribute vec4 aH; // r, g, b intensity (linear), dash flag
        uniform float uAlpha, uSnap;
        varying vec3 vCol; varying float vA; varying float vDash; varying float vU;
        ${GLSL_GEO} ${GLSL_LENS}
        void main() {
          vec3 p = position;
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          float depth = -mv.z;
          float s = clamp(uSnap * 1.5 - radial(p) * 0.4, 0.0, 1.0);
          vA = uDens * uAlpha * faceFade(p) * sideOf(p) * fogOf(depth) * (1.0 - smoothstep(0.0, 0.6, s)) / (1.0 + cocOf(depth) * 0.12);
          vCol = aH.rgb; vDash = aH.a; vU = p.x * 3.0 + p.z * 3.0;
        }`,
      fragmentShader: `
        varying vec3 vCol; varying float vA; varying float vDash; varying float vU;
        void main() {
          float dash = vDash > 0.5 ? step(0.5, fract(vU)) : 1.0;
          float a = vA * dash; gl_FragColor = vec4(vCol * a, a);
        }`,
    });

    // ambient fluid: short layer-parallel drifts inside the oil column, respawned at hashed
    // positions FLOW_CYCLES times per loop (loop-exact), each with a faint trail
    const flowMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uAlpha: { value: 0 }, uPx: { value: 1.6 }, uK: { value: FLOW_CYCLES / DURATION }, uLen: { value: 0.9 }, uOWC: { value: OWC } }),
      vertexShader: `
        attribute vec4 aF; // seed, layer fraction, phase offset, trail index
        uniform float uAlpha, uPx, uK, uLen, uOWC;
        varying float vA; varying float vBlur;
        ${GLSL_GEO} ${GLSL_LENS}
        uniform float uT;
        float h11(float n) { return fract(sin(n * 12.9898) * 43758.5453); }
        void main() {
          float ph = uT * uK + aF.z;
          float c = floor(ph), u = fract(ph) - aF.w * 0.018;
          float sx = h11(c * 1.731 + aF.x * 97.13), sz = h11(c * 2.917 + aF.x * 41.77 + 0.5);
          vec2 xz0 = vec2((sx * 2.0 - 1.0) * uHalf.x, (sz * 2.0 - 1.0) * uHalf.y) * 0.8;
          vec2 dir = normalize(vec2(1.0, 0.35 * sin(xz0.y * 0.7 + xz0.x * 0.25)));
          vec2 xz = xz0 + dir * (u - 0.5) * uLen;
          float y = mix(resBase(xz), resTop(xz), aF.y);
          vec3 p = vec3(xz.x, y, xz.y);
          float oil = smoothstep(uOWC, uOWC + 0.3, y) * (0.3 + 0.7 * dome(xz));
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          float depth = -mv.z;
          float base = uPx * uDpr * (uCamZ / depth) * (1.0 - aF.w * 0.22);
          float coc = cocOf(depth);
          float size = base + coc;
          vBlur = coc / size;
          float life = sin(3.14159 * clamp(u, 0.0, 1.0));
          vA = uDens * uAlpha * oil * life * (1.0 - aF.w * 0.38) * fogOf(depth) * max((base * base) / (size * size), 0.05);
          gl_PointSize = size;
        }`,
      fragmentShader: `
        varying float vA; varying float vBlur; ${GLSL_SPRITE}
        void main() {
          float a = sprite(length(gl_PointCoord - 0.5) * 2.0, vBlur) * vA; if (a < 0.002) discard;
          gl_FragColor = vec4(vec3(${AMBER.map((v) => (v * 1.6).toFixed(3)).join(',')}) * a, a);
        }`,
    });

    const GRID_COL = lin('#8FC3D8');
    const gridMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uSweep: { value: -1 }, uAlpha: { value: 0 } }),
      vertexShader: `
        attribute float aW; uniform float uSweep, uAlpha;
        varying float vA; varying float vBand;
        ${GLSL_GEO} ${GLSL_LENS}
        void main() {
          vec3 p = position;
          float d = uSweep - radial(p);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          float depth = -mv.z;
          float k = sqrt(uDens) * faceFade(p) * fogOf(depth) / (1.0 + cocOf(depth) * 0.1);
          vA = k * aW * uAlpha * smoothstep(0.0, 0.08, d);
          vBand = k * aW * exp(-d * d / 0.0005) * (1.0 - smoothstep(1.0, 1.1, uSweep)) * step(0.001, uAlpha);
        }`,
      fragmentShader: `
        varying float vA; varying float vBand;
        void main() {
          vec3 c = vec3(${GRID_COL.map((v) => v.toFixed(3)).join(',')}) * vA * 0.22 + vec3(1.4, 1.6, 1.75) * vBand;
          gl_FragColor = vec4(c, vA * 0.22 + vBand);
        }`,
    });

    // tiny intersection ticks on the top-reservoir surface
    const tickMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uSweep: { value: -1 }, uAlpha: { value: 0 } }),
      vertexShader: `
        uniform float uSweep, uAlpha; varying float vA; varying float vBlur;
        ${GLSL_GEO} ${GLSL_LENS}
        void main() {
          vec3 p = position;
          float d = uSweep - radial(p);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          float depth = -mv.z;
          float base = 2.2 * uDpr * (uCamZ / depth), coc = cocOf(depth), size = base + coc;
          vBlur = coc / size;
          vA = sqrt(uDens) * uAlpha * smoothstep(0.0, 0.05, d) * (1.0 + 2.0 * exp(-d * d / 0.002)) * fogOf(depth) * max((base * base) / (size * size), 0.06);
          gl_PointSize = size;
        }`,
      fragmentShader: `
        varying float vA; varying float vBlur; ${GLSL_SPRITE}
        void main() { float a = sprite(length(gl_PointCoord - 0.5) * 2.0, vBlur) * vA * 0.7; if (a < 0.002) discard; gl_FragColor = vec4(vec3(0.75, 0.88, 0.95) * a, a); }`,
    });

    const NAME_TOP = lin('#FFFFFF'), NAME_BOT = lin('#AEB4BD');
    const nodeMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uIn: { value: 0 }, uAlpha: { value: 0 }, uNameA: { value: 1 }, uPx: { value: 2.2 }, uNamePx: { value: 3 },
        uNameY: { value: new THREE.Vector2(1, -1) }, uStreak: { value: 1 } }),
      vertexShader: `
        attribute vec3 aCell, aJit, aTgt, aScat; attribute vec4 aMeta; attribute float aRep;
        uniform float uIn, uAlpha, uNameA, uPx, uNamePx, uMorph, uMorphPrev, uDisp, uDispPrev, uStreak;
        uniform vec2 uNameY, uRes;
        varying float vA; varying float vBlur; varying float vSize; varying float vL; varying vec2 vDir; varying vec3 vCol;
        ${GLSL_GEO} ${GLSL_LENS} ${GLSL_NODE}
        void main() {
          float e, ep;
          vec3 p = nodePos(aCell, aJit, aTgt, aScat, aMeta, uMorph, uDisp, e);
          vec3 pp = nodePos(aCell, aJit, aTgt, aScat, aMeta, uMorphPrev, uDispPrev, ep);
          float a0 = clamp(uIn * 1.45 - aMeta.z * 0.45, 0.0, 1.0);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          vec4 cp = projectionMatrix * mv;
          vec4 cq = projectionMatrix * modelViewMatrix * vec4(pp, 1.0);
          gl_Position = cp;
          // screen-space velocity -> streak (cheap motion blur)
          vec2 v = (cp.xy / cp.w - cq.xy / cq.w) * 0.5 * uRes;
          float vl = length(v);
          vDir = vl > 1e-3 ? v / vl : vec2(1.0, 0.0);
          vL = min(vl * uStreak, 34.0 * uDpr);
          float depth = -mv.z;
          float hub = step(1.5, aMeta.w);
          float base = mix(uPx * (hub > 0.5 ? 1.7 : (0.85 + 0.3 * aMeta.w)), uNamePx, e) * uDpr * (uCamZ / depth);
          // in flight the dot reads as a thin streak, not a fat bokeh capsule
          float coc = cocOf(depth) / (1.0 + vL * 0.25);
          vSize = base + coc;
          vBlur = coc / vSize;
          float energy = max((base * base) / (vSize * vSize), 0.05) * (vSize / (vSize + vL * 0.6));
          float fog = mix(fogOf(depth), 1.0, e);
          // replicas ride hidden inside their node and peel off into the glyphs
          float rep = aRep > 0.5 ? smoothstep(0.3, 0.85, e) : 1.0;
          float netA = (0.35 + 0.55 * min(aMeta.w, 1.0) + hub * 0.6) * uDens;
          vA = uAlpha * rep * smoothstep(0.0, 0.6, a0) * fog * mix(netA, uNameA, e) * energy;
          // name light: brighter top, cooler bottom (matches the HTML gradient type)
          float gy = clamp((aTgt.y - uNameY.y) / (uNameY.x - uNameY.y), 0.0, 1.0);
          vec3 nameCol = mix(vec3(${NAME_BOT.map((v) => v.toFixed(3)).join(',')}), vec3(${NAME_TOP.map((v) => v.toFixed(3)).join(',')}), gy);
          vCol = mix(vec3(0.78, 0.86, 0.94), nameCol, e);
          gl_PointSize = vSize + vL;
        }`,
      fragmentShader: `
        varying float vA; varying float vBlur; varying float vSize; varying float vL; varying vec2 vDir; varying vec3 vCol;
        ${GLSL_SPRITE}
        void main() {
          vec2 q = (gl_PointCoord - 0.5) * (vSize + vL); q.y = -q.y;
          float h = clamp(dot(q, vDir), -0.5 * vL, 0.5 * vL);
          float d = length(q - vDir * h) / (0.5 * vSize);
          float a = sprite(d, vBlur) * vA; if (a < 0.002) discard;
          gl_FragColor = vec4(vCol * a, a);
        }`,
    });

    const edgeMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uGrow: { value: 0 }, uAlpha: { value: 0 } }),
      vertexShader: `
        attribute vec3 aAc, aAj, aAt, aBc, aBj, aBt; attribute vec4 aAm, aBm; attribute vec4 aE; // end, order, rand, kind
        uniform float uGrow, uAlpha, uMorph, uDisp;
        varying float vA; varying float vS; varying float vPulse;
        ${GLSL_GEO} ${GLSL_LENS} ${GLSL_NODE}
        void main() {
          float eA, eB;
          vec3 pa = nodePos(aAc, aAj, aAt, vec3(0.0), aAm, uMorph, 0.0, eA);
          vec3 pb = nodePos(aBc, aBj, aBt, vec3(0.0), aBm, uMorph, 0.0, eB);
          float g = clamp((uGrow * 1.5 - aE.y * 0.5) * 3.0, 0.0, 1.0);
          g = g * g * (3.0 - 2.0 * g);
          vec3 p = mix(pa, mix(pa, pb, g), aE.x);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          float depth = -mv.z;
          float arrive = max(eA, eB);
          vA = uDens * uAlpha * step(0.001, g) * (1.0 - smoothstep(0.0, 0.35, arrive)) * fogOf(depth) / (1.0 + cocOf(depth) * 0.25) * (aE.w > 0.5 ? 0.5 : 1.0);
          vS = aE.x * g;
          vPulse = aE.z;
        }`,
      fragmentShader: `
        uniform float uT; varying float vA; varying float vS; varying float vPulse;
        void main() {
          float ph = fract(uT * 0.32 + vPulse * 7.0);
          float pulse = step(vPulse, 0.045) * exp(-pow((vS - ph) / 0.06, 2.0));
          float grad = 0.35 + 0.65 * abs(vS * 2.0 - 1.0); // brighter at the nodes
          float a = vA * (0.12 * grad + pulse * 0.9);
          vec3 c = vec3(0.55, 0.72, 0.82) * vA * 0.12 * grad + vec3(1.3, 0.8, 0.35) * vA * pulse * 0.9;
          gl_FragColor = vec4(c, a);
        }`,
    });

    // thin ring markers around the hub nodes
    const ringMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: shared({ uAlpha: { value: 0 } }),
      vertexShader: `
        attribute vec3 aCell, aJit, aTgt; attribute vec4 aMeta;
        uniform float uAlpha, uMorph; varying float vA; varying float vSize;
        ${GLSL_GEO} ${GLSL_LENS} ${GLSL_NODE}
        void main() {
          float e;
          vec3 p = nodePos(aCell, aJit, aTgt, vec3(0.0), aMeta, uMorph, 0.0, e);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          vSize = 15.0 * uDpr;
          vA = uAlpha * (1.0 - smoothstep(0.0, 0.2, e));
          gl_PointSize = vSize;
        }`,
      fragmentShader: `
        varying float vA; varying float vSize;
        void main() {
          float d = length(gl_PointCoord - 0.5) * vSize; // px from centre
          float r = 0.5 * vSize - 1.5;
          float a = (1.0 - smoothstep(0.0, 0.9, abs(d - r))) * vA;
          if (a < 0.002) discard;
          gl_FragColor = vec4(vec3(0.85, 0.92, 1.0) * a, a);
        }`,
    });

    // ---------- objects (geometry filled in build) ----------
    const strata = new THREE.Points(new THREE.BufferGeometry(), strataMat);
    const hair = new THREE.LineSegments(new THREE.BufferGeometry(), hairMat);
    const flow = new THREE.Points(new THREE.BufferGeometry(), flowMat);
    const grid = new THREE.LineSegments(new THREE.BufferGeometry(), gridMat);
    const ticks = new THREE.Points(new THREE.BufferGeometry(), tickMat);
    const edges = new THREE.LineSegments(new THREE.BufferGeometry(), edgeMat);
    const nodes = new THREE.Points(new THREE.BufferGeometry(), nodeMat);
    const rings = new THREE.Points(new THREE.BufferGeometry(), ringMat);
    const objs = [strata, hair, flow, grid, ticks, edges, nodes, rings];
    objs.forEach((o, i) => { o.frustumCulled = false; o.renderOrder = i; world.add(o); });
    flow.visible = AMBIENT_FLOW;
    const mats = [strataMat, hairMat, flowMat, gridMat, tickMat, nodeMat, edgeMat, ringMat];

    // ---------- post: HDR (MSAA) -> highlight bloom -> filmic grade ----------
    const rt = new THREE.WebGLRenderTarget(ctx.width * dpr, ctx.height * dpr, { type: THREE.HalfFloatType, samples: high ? 4 : 0 });
    const composer = new EffectComposer(renderer, rt);
    composer.addPass(new RenderPass(scene, camera));
    let bloom = null;
    if (high) {
      bloom = new UnrealBloomPass(new THREE.Vector2(ctx.width, ctx.height), 0.22, 0.8, 1.1);
      composer.addPass(bloom);
    }
    const gradeMat = new THREE.ShaderMaterial({
      uniforms: {
        tDiffuse: { value: null }, uExposure: { value: 1.25 }, uGrain: { value: high ? 0.045 : 0 }, uSeed: { value: 0 },
        uAspect: { value: ctx.width / ctx.height }, uRes: { value: new THREE.Vector2(ctx.width * dpr, ctx.height * dpr) },
      },
      vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
      fragmentShader: `
        uniform sampler2D tDiffuse; uniform float uExposure, uGrain, uSeed, uAspect; uniform vec2 uRes;
        varying vec2 vUv;
        vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
        vec3 srgb(vec3 c) { return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
        float hash(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
        void main() {
          vec3 hdr = texture2D(tDiffuse, vUv).rgb * uExposure;
          vec3 col = srgb(aces(hdr));
          // graphite ground, soft off-axis key light from the upper left (display space, screen-blended)
          vec2 kd = (vUv - vec2(0.18, 0.92)) * vec2(uAspect, 1.0);
          float key = exp(-dot(kd, kd) * 0.9);
          vec3 ground = vec3(0.043, 0.048, 0.056) + vec3(0.050, 0.054, 0.060) * key;
          col = 1.0 - (1.0 - col) * (1.0 - ground);
          vec2 q = (vUv - 0.5) * vec2(uAspect, 1.0);
          col *= 1.0 - 0.42 * smoothstep(0.3, 1.1, length(q * vec2(0.8, 1.0)));
          // film grain, strongest in the mid-tones
          float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
          col += (hash(vUv * uRes + uSeed) - 0.5) * uGrain * (0.35 + 0.65 * smoothstep(0.03, 0.3, luma));
          gl_FragColor = vec4(col, 1.0);
        }`,
      depthTest: false, depthWrite: false,
    });
    composer.addPass(new ShaderPass(gradeMat));

    // ---------- editorial HUD (DOM) ----------
    let hud = null;
    const hudEls = {};
    if (HUD) {
      hud = document.createElement('div');
      hud.setAttribute('aria-hidden', 'true');
      Object.assign(hud.style, {
        position: 'absolute', inset: '0', pointerEvents: 'none', zIndex: '4', fontFamily: MONO_FAMILY,
        fontSize: '10.5px', letterSpacing: '0.16em', textTransform: 'uppercase', color: 'rgba(235,238,242,0.5)', lineHeight: '1.6',
      });
      const chapters = ['Reservoir', 'Grid', 'Network'];
      hud.innerHTML = `
        <div data-k="tl" style="position:absolute;left:var(--m);top:var(--m)">
          ${chapters.map((c, i) => `<div data-ch="${i}" style="position:absolute;left:0;top:0;white-space:nowrap"><span style="color:rgba(235,238,242,0.78)">0${i + 1}</span> / 03&nbsp;&nbsp;·&nbsp;&nbsp;${c}</div>`).join('')}
        </div>
        <div data-k="bl" style="position:absolute;left:var(--m);bottom:var(--m)">
          <div data-rd="0" style="position:absolute;left:0;bottom:0;white-space:nowrap">φ <span data-v="phi">0.21</span>&nbsp;&nbsp;·&nbsp;&nbsp;k <span data-v="perm">340</span> mD&nbsp;&nbsp;·&nbsp;&nbsp;Sw <span data-v="sw">0.28</span></div>
          <div data-rd="1" style="position:absolute;left:0;bottom:0;white-space:nowrap">Corner-point grid&nbsp;&nbsp;·&nbsp;&nbsp;cells <span data-v="cells">0</span> / ${NN.toLocaleString('en-US')}</div>
          <div data-rd="2" style="position:absolute;left:0;bottom:0;white-space:nowrap">k-NN graph&nbsp;&nbsp;·&nbsp;&nbsp;edges <span data-v="edges">0</span></div>
        </div>
        <div data-k="br" style="position:absolute;right:var(--m);bottom:var(--m);text-align:right;white-space:nowrap">
          35.21° N&nbsp;&nbsp;·&nbsp;&nbsp;97.44° W
          <div style="position:relative;height:1px;margin-top:9px;background:rgba(235,238,242,0.14)"><div data-k="prog" style="position:absolute;inset:0;background:rgba(235,238,242,0.55);transform-origin:0 50%"></div></div>
        </div>
        ${HUB_LABELS.map((s, i) => `<div data-hub="${i}" style="position:absolute;left:0;top:0;white-space:nowrap;font-size:9.5px;letter-spacing:0.12em;color:rgba(235,238,242,0.85);text-shadow:0 0 6px #0b0d10,0 0 2px #0b0d10">${s}</div>`).join('')}`;
      ctx.container.appendChild(hud);
      hudEls.tl = hud.querySelector('[data-k="tl"]');
      hudEls.bl = hud.querySelector('[data-k="bl"]');
      hudEls.br = hud.querySelector('[data-k="br"]');
      hudEls.prog = hud.querySelector('[data-k="prog"]');
      hudEls.ch = [...hud.querySelectorAll('[data-ch]')];
      hudEls.rd = [...hud.querySelectorAll('[data-rd]')];
      hudEls.hub = [...hud.querySelectorAll('[data-hub]')];
      hudEls.v = {};
      for (const el of hud.querySelectorAll('[data-v]')) hudEls.v[el.dataset.v] = { el, last: null };
    }
    const setText = (key, val) => { const r = hudEls.v[key]; if (r.last !== val) { r.last = val; r.el.textContent = val; } };

    let G = makeGeo(ctx.width / ctx.height);
    let S = 1;
    let camKeys = camKeysFor(ctx.width / ctx.height);
    let edgeCount = 0, readOff = 0, narrow = false, vw = ctx.width, vh = ctx.height;
    const hubData = new Float32Array(N_HUBS * 10); // cell(3) jit(3) seed, unused
    const F = { front: new THREE.Vector3(), crest: new THREE.Vector3(), centre: new THREE.Vector3() };

    const setGeo = (o, attrs) => {
      o.geometry.dispose();
      const g = new THREE.BufferGeometry();
      for (const k in attrs) g.setAttribute(k, new THREE.BufferAttribute(attrs[k][0], attrs[k][1]));
      g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e4);
      o.geometry = g;
    };

    function build(c) {
      const w = c.width, h = c.height, aspect = w / h;
      vw = w; vh = h; narrow = w < 600;
      const portrait = portraitOf(aspect);
      G = makeGeo(aspect);
      camKeys = camKeysFor(aspect);
      const { W, D } = G, HW = W / 2, HD = D / 2;
      U.uHalf.value.set(HW, HD);
      U.uDome.value.set(G.AX, G.AZ);
      camera.aspect = aspect;
      camera.updateProjectionMatrix();
      const d0 = camKeys[0][3];
      const visW = 2 * d0 * Math.tan(THREE.MathUtils.degToRad(FOV / 2)) * aspect;
      const az0 = THREE.MathUtils.degToRad(camKeys[0][1]);
      const proj = W * Math.cos(az0) + D * Math.sin(az0);
      S = Math.min(1, (visW * lerp(0.86, 0.84, portrait)) / proj);
      world.scale.setScalar(S);
      U.uDens.value = clamp(Math.pow(S, 1.5) * 1.15, 0.35, 1);
      if (bloom) bloom.strength = 0.22 * lerp(0.4, 1, S);
      U.uRes.value.set(w * dpr, h * dpr);
      gradeMat.uniforms.uAspect.value = aspect;
      gradeMat.uniforms.uGrain.value = high ? (narrow ? 0.03 : 0.045) : 0;
      gradeMat.uniforms.uRes.value.set(w * dpr, h * dpr);
      F.front.set(0, -0.2 * S, HD * S);
      F.crest.set(0, hz(G, 2, 0, 0) * S, 0);
      F.centre.set(0, 0, 0);

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
      const oilOf = (x, y, z) => clamp((y - OWC) / 0.32) * (0.3 + 0.7 * domeJS(G, x, z));

      // ----- strata point cloud: fine points between the hairlines -----
      {
        const N = C.strata, r = rng(101);
        const pos = new Float32Array(N * 3), col = new Float32Array(N * 3), cell = new Float32Array(N * 3), seed = new Float32Array(N * 2);
        const gauss = () => (r() + r() + r() - 1.5) / 1.5;
        const faceY = (fx, fz) => {
          for (let tries = 0; tries < 6; tries++) {
            const yy = HZ[5].b + r() * (HZ[0].b - HZ[5].b);
            if (r() < LAYER_DENS[layerOf(fx, yy, fz)]) return yy;
          }
          return HZ[5].b + r() * (HZ[0].b - HZ[5].b);
        };
        for (let n = 0; n < N; n++) {
          const q = r();
          let x, y, z, sheet = -1;
          if (q < 0.45) { x = -HW + r() * W; z = HD; y = faceY(x, z); }
          else if (q < 0.57) { x = HW; z = -HD + r() * D; y = faceY(x, z); }
          else if (q < 0.8) {
            const qs = r();
            sheet = qs < 0.2 ? 1 : qs < 0.62 ? 2 : qs < 0.82 ? 3 : 4;
            x = -HW + r() * W; z = -HD + r() * D; y = hz(G, sheet, x, z) + gauss() * 0.015;
          } else {
            // sand body volume, concentrated toward the crest
            const a = r() * Math.PI * 2, rr = Math.sqrt(r()) * 1.15;
            x = clamp(Math.cos(a) * rr * HW, -HW, HW); z = clamp(Math.sin(a) * rr * HD, -HD, HD);
            const top = hz(G, 2, x, z), bot = hz(G, 3, x, z);
            y = bot + r() * (top - bot);
          }
          const L = sheet >= 0 ? Math.min(sheet, 4) : layerOf(x, y, z);
          let c3 = LAYER_COL[L];
          const grain = 0.55 + 0.45 * r();
          let k = LAYER_GAIN[L] * grain;
          if (L === 2 || sheet === 2) {
            const oil = oilOf(x, sheet === 2 ? hz(G, 2, x, z) : y, z);
            c3 = [lerp(c3[0], AMBER[0], oil), lerp(c3[1], AMBER[1], oil), lerp(c3[2], AMBER[2], oil)];
            k *= 1 + 0.9 * oil;
          }
          if (sheet >= 0) k *= 0.8;
          col[n * 3] = c3[0] * k; col[n * 3 + 1] = c3[1] * k; col[n * 3 + 2] = c3[2] * k;
          pos[n * 3] = x; pos[n * 3 + 1] = y; pos[n * 3 + 2] = z;
          const [i, j, kk] = cellIndex(x, y, z);
          cell.set(cellCentre(i, j, kk), n * 3);
          seed[n * 2] = r(); seed[n * 2 + 1] = r();
        }
        setGeo(strata, { position: [pos, 3], aCol: [col, 3], aCell: [cell, 3], aSeed: [seed, 2] });
      }

      // ----- hairlines: horizons on the cut faces, block edges, dashed oil-water contact -----
      {
        const segs = [];
        const Q = 96;
        const face = (fn, k, colr, dash = 0) => {
          let prev = null;
          for (let q = 0; q <= Q; q++) {
            const cur = fn(q / Q, k);
            if (prev && cur) segs.push([...prev, ...colr, dash], [...cur, ...colr, dash]);
            prev = cur;
          }
        };
        const front = (u, k) => { const x = -HW + u * W; return [x, hz(G, k, x, HD), HD]; };
        const right = (u, k) => { const z = HD - u * D; return [HW, hz(G, k, HW, z), z]; };
        const silver = [0.85, 0.89, 0.95], dim = [0.42, 0.45, 0.5];
        for (const k of [1, 2, 3, 4]) {
          const cl = k === 2 ? [1.6, 0.95, 0.42] : silver.map((v) => v * (k === 3 ? 0.75 : 0.6));
          face(front, k, cl); face(right, k, cl);
        }
        for (const k of [0, 5]) { face(front, k, dim); face(right, k, dim); }
        // top cut edges (back and left) and the three visible vertical edges
        face((u) => { const x = -HW + u * W; return [x, HZ[0].b, -HD]; }, 0, dim.map((v) => v * 0.4));
        face((u) => { const z = -HD + u * D; return [-HW, HZ[0].b, z]; }, 0, dim.map((v) => v * 0.4));
        for (const [x, z] of [[-HW, HD], [HW, HD], [HW, -HD]]) segs.push([x, HZ[5].b, z, ...dim, 0], [x, HZ[0].b, z, ...dim, 0]);
        // oil-water contact: dashed, only inside the sand body
        const owc = (fn) => {
          let prev = null;
          for (let q = 0; q <= Q; q++) {
            const p = fn(q / Q);
            const ok = hz(G, 2, p[0], p[2]) > OWC && hz(G, 3, p[0], p[2]) < OWC;
            const cur = ok ? [p[0], OWC, p[2]] : null;
            if (prev && cur) segs.push([...prev, 0.7, 0.45, 0.2, 1], [...cur, 0.7, 0.45, 0.2, 1]);
            prev = cur;
          }
        };
        owc((u) => [-HW + u * W, 0, HD]); owc((u) => [HW, 0, HD - u * D]);
        const pos = new Float32Array(segs.length * 3), ah = new Float32Array(segs.length * 4);
        segs.forEach((s, i) => { pos.set(s.slice(0, 3), i * 3); ah.set(s.slice(3, 7), i * 4); });
        setGeo(hair, { position: [pos, 3], aH: [ah, 4] });
      }

      // ----- ambient flow particles (3 trail samples each) -----
      {
        const r = rng(202), N = C.flow, TR = 3;
        const f = new Float32Array(N * TR * 4);
        for (let n = 0; n < N; n++) {
          const sd = r(), lf = 0.25 + r() * 0.6, off = r();
          for (let k = 0; k < TR; k++) f.set([sd, lf, off, k], (n * TR + k) * 4);
        }
        setGeo(flow, { position: [new Float32Array(N * TR * 3), 3], aF: [f, 4] });
      }

      // ----- corner-point grid hairlines + intersection ticks -----
      {
        const segs = [];
        const poly = (pts, wgt) => { for (let i = 0; i + 1 < pts.length; i++) segs.push([...pts[i], wgt], [...pts[i + 1], wgt]); };
        const sub = 3;
        for (const [kk, wgt] of [[2, 1.0], [4, 0.5]]) {
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
        // k-layer traces and pillars on the front and right faces
        for (const kk of [0, 1, 3, 5, 6]) {
          { const pts = []; for (let q = 0; q <= NI * sub; q++) { const x = -HW + (q / (NI * sub)) * W; pts.push([x, gs(G, kk, x, HD), HD]); } poly(pts, 0.55); }
          { const pts = []; for (let q = 0; q <= NJ * sub; q++) { const z = -HD + (q / (NJ * sub)) * D; pts.push([HW, gs(G, kk, HW, z), z]); } poly(pts, 0.55); }
        }
        for (let i = 0; i <= NI; i++) { const x = -HW + i * cw; poly([[x, gs(G, 6, x, HD), HD], [x, gs(G, 0, x, HD), HD]], 0.42); }
        for (let j = 0; j <= NJ; j++) { const z = -HD + j * cd; poly([[HW, gs(G, 6, HW, z), z], [HW, gs(G, 0, HW, z), z]], 0.42); }
        const pos = new Float32Array(segs.length * 3), aw = new Float32Array(segs.length);
        segs.forEach((s, i) => { pos[i * 3] = s[0]; pos[i * 3 + 1] = s[1]; pos[i * 3 + 2] = s[2]; aw[i] = s[3]; });
        setGeo(grid, { position: [pos, 3], aW: [aw, 1] });

        const tp = new Float32Array((NI + 1) * (NJ + 1) * 3);
        let n = 0;
        for (let i = 0; i <= NI; i++) for (let j = 0; j <= NJ; j++) {
          const x = -HW + i * cw, z = -HD + j * cd;
          tp.set([x, gs(G, 2, x, z) + 0.004, z], n * 3); n++;
        }
        setGeo(ticks, { position: [tp, 3] });
      }

      // ----- nodes (cell centres) + replicas + hubs + k-NN network + name targets -----
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
          jit[n * 3] = (r() - 0.5) * cw * 1.7;
          jit[n * 3 + 1] = (r() - 0.5) * th * 1.3;
          jit[n * 3 + 2] = (r() - 0.5) * cd * 1.7;
          for (let a = 0; a < 3; a++) net[n * 3 + a] = cellP[n * 3 + a] + jit[n * 3 + a];
          meta[n * 4] = r();
          meta[n * 4 + 2] = Math.min(1, Math.hypot(xc / HW, zc / HD) / Math.SQRT2);
          meta[n * 4 + 3] = k === 2 || k === 3 ? 1 : 0.2 + r() * 0.35;
          n++;
        }
        // hubs: farthest-point picks among sand-body nodes in the inner 80% of the block
        const hubs = [];
        {
          const cand = [];
          for (let q = 0; q < NN; q++) {
            const kq = Math.floor(q / (NI * NJ));
            if ((kq === 2 || kq === 3) && Math.abs(net[q * 3]) < HW * 0.78 && Math.abs(net[q * 3 + 2]) < HD * 0.7) cand.push(q);
          }
          hubs.push(cand[Math.floor(cand.length * 0.37)]);
          while (hubs.length < N_HUBS) {
            let best = -1, bd = -1;
            for (const q of cand) {
              let dmin = 1e9;
              for (const hq of hubs) { const dx = net[q * 3] - net[hq * 3], dz = net[q * 3 + 2] - net[hq * 3 + 2]; dmin = Math.min(dmin, dx * dx + dz * dz); }
              if (dmin > bd) { bd = dmin; best = q; }
            }
            hubs.push(best);
          }
          for (const hq of hubs) meta[hq * 4 + 3] = 2;
        }

        // targets on the shared glyphs, mapped to the z = 0 plane of the canonical pose
        const L = c.layout;
        const smp = sampleName(L, NR, { which: 'short', mode: 'mix', seed: 11 }).positions;
        const toPlane = pixelToPlane(camera, CAM_Z, w, h, 0);
        const v = new THREE.Vector3();
        const tg = new Float32Array(NR * 3);
        for (let q = 0; q < NR; q++) { toPlane(smp[q * 2], smp[q * 2 + 1], v); tg[q * 3] = v.x / S; tg[q * 3 + 1] = v.y / S; }
        toPlane(0, L.baseline - 0.78 * L.fontSize, v); const yTop = v.y / S;
        toPlane(0, L.baseline + 0.06 * L.fontSize, v); const yBot = v.y / S;
        nodeMat.uniforms.uNameY.value.set(yTop, yBot);
        // coherent assignment of NR particles (each node NAME_REPLICAS times)
        const owner = (q) => q % NN;
        const ni = Array.from({ length: NR }, (_, q) => q).sort((a, b) => net[owner(a) * 3] - net[owner(b) * 3]);
        const ti = Array.from({ length: NR }, (_, q) => q).sort((a, b) => tg[a * 3] - tg[b * 3]);
        const chunk = Math.max(8, Math.round(NR / (NI * 1.5)));
        const pT = new Float32Array(NR * 3), pS = new Float32Array(NR * 3), pM = new Float32Array(NR * 4);
        const pC = new Float32Array(NR * 3), pJ = new Float32Array(NR * 3), pR = new Float32Array(NR);
        const nodeTgt = new Float32Array(NN * 3), nodeOrd = new Float32Array(NN);
        const nameL = tg[ti[0] * 3], nameR = tg[ti[NR - 1] * 3];
        for (let s0 = 0; s0 < NR; s0 += chunk) {
          const a = ni.slice(s0, s0 + chunk).sort((p, q) => net[owner(q) * 3 + 1] - net[owner(p) * 3 + 1]);
          const b = ti.slice(s0, s0 + chunk).sort((p, q) => tg[q * 3 + 1] - tg[p * 3 + 1]);
          for (let m = 0; m < a.length; m++) {
            const pq = a[m], nd = owner(pq), tq = b[m];
            const primary = pq < NN;
            pT[pq * 3] = tg[tq * 3]; pT[pq * 3 + 1] = tg[tq * 3 + 1];
            // strong left-to-right stagger so letters resolve in reading order
            const ord = clamp(0.82 * (tg[tq * 3] - nameL) / (nameR - nameL || 1) + 0.18 * r());
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

        // hub rings + CPU copies for the HUD labels
        {
          const hc = new Float32Array(N_HUBS * 3), hj = new Float32Array(N_HUBS * 3), ht = new Float32Array(N_HUBS * 3), hm = new Float32Array(N_HUBS * 4);
          hubs.forEach((hq, i) => {
            for (let c3 = 0; c3 < 3; c3++) { hc[i * 3 + c3] = cellP[hq * 3 + c3]; hj[i * 3 + c3] = jit[hq * 3 + c3]; ht[i * 3 + c3] = nodeTgt[hq * 3 + c3]; }
            hm.set([meta[hq * 4], nodeOrd[hq], meta[hq * 4 + 2], meta[hq * 4 + 3]], i * 4);
            hubData.set([cellP[hq * 3], cellP[hq * 3 + 1], cellP[hq * 3 + 2], jit[hq * 3], jit[hq * 3 + 1], jit[hq * 3 + 2], meta[hq * 4], meta[hq * 4 + 2]], i * 10);
          });
          setGeo(rings, { position: [new Float32Array(N_HUBS * 3), 3], aCell: [hc, 3], aJit: [hj, 3], aTgt: [ht, 3], aMeta: [hm, 4] });
        }

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
        // hubs reach further: a few long links each
        const cs = Math.max(cw, cd);
        for (const hq of hubs) {
          let made = 0;
          for (let tries = 0; tries < 400 && made < 7; tries++) {
            const b = Math.floor(r() * NN);
            const dx = net[b * 3] - net[hq * 3], dy = net[b * 3 + 1] - net[hq * 3 + 1], dz = net[b * 3 + 2] - net[hq * 3 + 2];
            const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (d > cs * 1.5 && d < cs * 4.5) { E.push([hq, b, 1]); made++; }
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
        nodeMat.uniforms.uNamePx.value = clamp(Math.sqrt(pxPer) * 2.4, 2.0, 5.2);
        nodeMat.uniforms.uNameA.value = clamp(pxPer * 0.066, 0.15, 0.62);
      }

      // read phase: the strata return, face-on and dim, below the subtitle
      {
        const L = c.layout;
        const dFront = CAM_Z - HD * S;
        const pxPerUnit = h / (2 * dFront * Math.tan(THREE.MathUtils.degToRad(FOV / 2)));
        const targetPx = L.subtitle.top + (L.subtitle.narrow ? 120 : 96);
        readOff = ((h / 2 - targetPx) / pxPerUnit) / S - HZ[0].b;
      }

      strataMat.uniforms.uPx.value = lerp(1.0, 1.25, S);
      flowMat.uniforms.uPx.value = lerp(1.5, 1.9, S);
      nodeMat.uniforms.uPx.value = lerp(2.0, 2.6, S);

      if (hud) {
        hud.style.setProperty('--m', narrow ? '20px' : '32px');
        hudEls.bl.style.display = narrow ? 'none' : '';
        hudEls.br.style.display = narrow ? 'none' : '';
        for (const el of hudEls.hub) el.style.display = narrow ? 'none' : '';
      }

      renderer.setSize(w, h, false);
      composer.setSize(w, h);
      if (bloom) bloom.resolution.set(w, h);
    }

    build(ctx);

    // ---------- camera, focus ----------
    const T = ctx.timing;
    const rebuild0 = T.out[1] - 0.3;
    const tgtV = new THREE.Vector3(), fwd = new THREE.Vector3(), tmp = new THREE.Vector3(), fp = new THREE.Vector3();
    let camDist = CAM_Z;
    function placeCamera(t) {
      let az, el, dist, ty;
      if (t <= NAME_AT) {
        az = hermiteKeys(camKeys, t, 1); el = hermiteKeys(camKeys, t, 2); dist = hermiteKeys(camKeys, t, 3); ty = hermiteKeys(camKeys, t, 4);
      } else {
        // hold the canonical pose; as the name leaves, return to the opening pose arriving
        // with the opening drift velocity, so the loop point never stops dead
        const h = T.duration - rebuild0;
        const u = ramp(t, rebuild0, T.duration);
        const K0 = camKeys[0], KN = camKeys[camKeys.length - 1];
        const ch = (c) => hermite(KN[c], K0[c], 0, keyTangent(camKeys, 0, c) * h, u);
        az = ch(1); el = ch(2); dist = ch(3); ty = ch(4);
      }
      camDist = dist;
      const a = THREE.MathUtils.degToRad(az), e = THREE.MathUtils.degToRad(el);
      tgtV.set(0, ty, 0);
      camera.position.set(Math.sin(a) * Math.cos(e) * dist, ty + Math.sin(e) * dist, Math.cos(a) * Math.cos(e) * dist);
      camera.lookAt(tgtV);
      camera.updateMatrixWorld();
      U.uCamL.value.copy(camera.position).divideScalar(S);
      U.uFogNear.value = dist - 1.0 * S;
      U.uFogK.value = 0.075 / S;
    }
    // rack focus: front cut face -> crest -> network centre / name plane -> back for the loop
    function placeFocus(t) {
      const k1 = easeInOutSine(ramp(t, 1.5, 2.7));
      const k2 = easeInOutSine(ramp(t, 3.3, 4.4));
      const kb = easeInOutSine(ramp(t, rebuild0, T.duration));
      if (t < T.split[1]) fp.copy(F.front).lerp(F.crest, k1).lerp(F.centre, k2);
      else fp.copy(F.centre).lerp(F.front, kb);
      camera.getWorldDirection(fwd);
      U.uFocus.value = tmp.copy(fp).sub(camera.position).dot(fwd);
      const ap = t < T.split[1]
        ? lerp(lerp(14, 11, ramp(t, 1.8, 3.0)), 16, ramp(t, 3.4, 4.4)) * (1 - 0.15 * ramp(t, 5.4, NAME_AT))
        : lerp(12, 14, kb);
      U.uAperture.value = ap * C.ap * dpr;
    }

    // HUD label projection (mirrors the GLSL nodePos for the network phase)
    function hubScreen(i, t, out) {
      const o = i * 10, seed = hubData[o + 6], L = U.uLoose.value;
      out.set(
        hubData[o] + hubData[o + 3] * L + Math.sin(t * 0.61 + seed * 31) * 0.06 * L,
        hubData[o + 1] + hubData[o + 4] * L + Math.sin(t * 0.47 + seed * 17) * 0.06 * L,
        hubData[o + 2] + hubData[o + 5] * L + Math.cos(t * 0.53 + seed * 23) * 0.06 * L,
      ).multiplyScalar(S).project(camera);
      out.x = (out.x * 0.5 + 0.5) * vw; out.y = (-out.y * 0.5 + 0.5) * vh;
      return out;
    }

    function updateHud(t) {
      if (!hud) return;
      const show = t < NAME_AT ? 1 - ramp(t, NAME_AT - 0.5, NAME_AT - 0.1) : ramp(t, rebuild0, T.duration);
      hud.style.opacity = show.toFixed(3);
      if (show <= 0) return;
      // chapter (and readout) crossfade; the loop's last window belongs to chapter 1
      const tc = t < NAME_AT ? t : 0;
      const w0 = 1 - ramp(tc, 1.65, 1.95), w1 = Math.min(ramp(tc, 1.75, 2.05), 1 - ramp(tc, 3.45, 3.75)), w2 = ramp(tc, 3.55, 3.85);
      const ws = [w0, w1, w2];
      for (let i = 0; i < 3; i++) {
        hudEls.ch[i].style.opacity = ws[i].toFixed(3);
        hudEls.ch[i].style.transform = `translateY(${((1 - ws[i]) * (tc < 1.9 + i * 1.8 ? 4 : -4)).toFixed(2)}px)`;
        hudEls.rd[i].style.opacity = ws[i].toFixed(3);
      }
      hudEls.prog.style.transform = `scaleX(${(t / T.duration).toFixed(4)})`;
      if (narrow) return;
      setText('phi', (0.21 + 0.012 * Math.sin(t * 1.7)).toFixed(3));
      setText('perm', String(Math.round(340 + 14 * Math.sin(t * 1.1 + 0.6))));
      setText('sw', (0.28 + 0.01 * Math.sin(t * 1.3 + 2.0)).toFixed(3));
      setText('cells', Math.round(NN * clamp(gridMat.uniforms.uSweep.value / 1.05)).toLocaleString('en-US'));
      setText('edges', Math.round(edgeCount * ramp(t, 3.45, 5.0)).toLocaleString('en-US'));
      // hub labels
      const la = Math.min(ramp(t, 3.9, 4.4), 1 - ramp(t, 5.15, 5.45));
      for (let i = 0; i < N_HUBS; i++) {
        const el = hudEls.hub[i];
        const a = la * clamp(1 - Math.abs(i / (N_HUBS - 1) - 0.5) * 0.4);
        el.style.opacity = a.toFixed(3);
        if (a > 0) { hubScreen(i, t, tmp); el.style.transform = `translate(${(tmp.x + 11).toFixed(1)}px, ${(tmp.y - 18).toFixed(1)}px)`; }
      }
    }

    return {
      get stats() { return `${C.strata.toLocaleString()} strata pts · ${AMBIENT_FLOW ? C.flow.toLocaleString() : 0} flow · ${NN.toLocaleString()} nodes (${NR.toLocaleString()} name pts) · ${edgeCount.toLocaleString()} edges`; },
      update(t) {
        U.uT.value = t;
        placeCamera(t);
        placeFocus(t);

        // strata + hairlines: lit from t = 0; dim under the grid and fall into the cells; a
        // dim face-on echo during the read; then rise back into the opening pose
        const SM = strataMat.uniforms;
        let strataA, snap;
        if (t < T.split[1]) {
          strataA = 1 - 0.35 * ramp(t, 1.8, 3.0);
          snap = ramp(t, 2.9, 4.4);
          SM.uGlintA.value = 0;
          strata.position.y = 0;
        } else if (t < rebuild0) {
          snap = 0;
          strataA = READ_ALPHA * (narrow ? 0.7 : 1) * easeInOutSine(ramp(t, T.read[0] - 0.8, T.read[0] + 1.2));
          SM.uGlint.value = lerp(-0.25, 1.25, ramp(t, T.read[0] - 0.8, rebuild0));
          SM.uGlintA.value = 0.8;
          strata.position.y = readOff;
        } else {
          const k = easeInOutSine(ramp(t, rebuild0, T.duration));
          snap = 0;
          strataA = lerp(READ_ALPHA * (narrow ? 0.7 : 1), 1, k);
          SM.uGlintA.value = 0;
          strata.position.y = readOff * (1 - k);
        }
        SM.uAlpha.value = strataA; SM.uSnap.value = snap;
        const kRet = ramp(t, rebuild0, T.duration);
        hairMat.uniforms.uAlpha.value = t < T.split[1] ? strataA : t < rebuild0 ? strataA * 0.3 : strataA * lerp(0.3, 1, kRet * kRet);
        U.uSideA.value = t < T.split[1] ? 1 : t < rebuild0 ? 0 : kRet * kRet; hairMat.uniforms.uSnap.value = snap;
        hair.position.y = strata.position.y;

        flow.position.y = strata.position.y;
        flowMat.uniforms.uAlpha.value = AMBIENT_FLOW ? (t < T.split[1] ? 1 - ramp(t, 2.4, 3.4) : strataA) : 0;

        // grid sweeps out from the crest; behind the front it settles to a low hairline
        const sweep = lerp(-0.05, 1.12, easeInOutSine(ramp(t, 1.75, 3.3)));
        const gridA = ramp(t, 1.7, 1.85) * (1 - 0.35 * ramp(t, 3.4, 4.0)) * (1 - ramp(t, 3.9, 4.9));
        gridMat.uniforms.uSweep.value = sweep; gridMat.uniforms.uAlpha.value = gridA;
        tickMat.uniforms.uSweep.value = sweep; tickMat.uniforms.uAlpha.value = gridA;

        // nodes -> network -> name -> hand-off -> quiet read
        nodeMat.uniforms.uIn.value = ramp(t, 2.6, 3.8);
        U.uLoose.value = easeInOutSine(ramp(t, 3.4, 5.0));
        U.uDrift.value = easeInOutCubic(ramp(t, NAME_AT + 0.3, T.split[0] + 0.4));
        const morphAt = (x) => ramp(x, 5.15, NAME_AT - 0.1);
        const dispAt = (x) => easeInOutCubic(ramp(x, NAME_AT + 0.3, T.split[0] + 0.4)) + 0.08 * Math.max(0, x - T.split[0]);
        U.uMorph.value = morphAt(t); U.uMorphPrev.value = morphAt(t - 0.045);
        U.uDisp.value = dispAt(t); U.uDispPrev.value = dispAt(t - 0.045);
        edgeMat.uniforms.uGrow.value = ramp(t, 3.45, 5.0);
        edgeMat.uniforms.uAlpha.value = ramp(t, 3.45, 3.65);
        ringMat.uniforms.uAlpha.value = Math.min(ramp(t, 3.8, 4.3), 1 - ramp(t, 5.1, 5.45)) * 0.8;
        const fadeName = ramp(t, NAME_AT + 0.25, T.split[0] - 0.05);
        const readLevel = 0.018 * (1 - ramp(t, T.out[0], T.out[1]));
        nodeMat.uniforms.uAlpha.value = lerp(1, readLevel, easeInOutSine(fadeName));

        gradeMat.uniforms.uSeed.value = Math.floor(t * 24) * 17.31;
        updateHud(t);
        composer.render();
      },
      resize(c) {
        build(c);
      },
      dispose() {
        for (const o of objs) o.geometry.dispose();
        for (const m of mats) m.dispose();
        gradeMat.dispose();
        if (bloom) bloom.dispose();
        composer.dispose();
        rt.dispose();
        renderer.dispose();
        renderer.domElement.remove();
        if (hud) hud.remove();
      },
    };
  },
};
