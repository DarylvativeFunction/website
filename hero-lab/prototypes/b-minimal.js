// b-minimal.js - Prototype B, "Minimal Premium".
//
// One restrained scene, monochrome. Thousands of tiny near-white particles drift in a
// soft pool of light, settle into a stratified, folded formation (an abstract anticline
// seen slightly from above), get traced by hairline contours and a sparse computational
// grid, loosen into flowing data streams with hairline links, and converge into
// "Daryl Tan". Everything is a pure function of t: positions are derived in the vertex
// shader from per-particle attributes + a handful of timeline uniforms.

import * as THREE from 'three';
import { createRenderer, pixelToPlane, ramp, smoothstep, rng } from '../core/stage.js';
import { sampleName, traceContours } from '../core/text.js';

const FOV = 32;
const CAM_Z = 1000;
const LAYERS = 9;
const TAU = Math.PI * 2;

// ---- shared GLSL ---------------------------------------------------------------------
const COMMON = /* glsl */`
  uniform float uFW, uFD, uTH, uA, uY0, uSlope, uVisH, uFlow, uLoose, uT;
  float h11(float n) { return fract(sin(n * 127.1 + 11.7) * 43758.5453); }
  float quint(float x) { return x * x * x * (x * (x * 6.0 - 15.0) + 10.0); }
  float flowU(vec4 A) {
    float lay = floor(A.z);
    float sp = 0.55 + 0.85 * h11(lay + 3.0);
    return fract((A.x + 1.0) * 0.5 + uFlow * sp);
  }
  float edgeOf(vec4 A) { float u = flowU(A); return smoothstep(0.0, 0.1, u) * smoothstep(1.0, 0.9, u); }
  vec3 formPos(vec4 A, vec4 B) {
    float u = flowU(A);
    float xs = u * 2.0 - 1.0;
    float k = A.z, zn = A.y;
    float g = xs * 1.6;
    float dome = exp(-g * g) * (1.0 - 0.45 * zn * zn);
    float rip = 0.16 * sin(xs * 2.7 + zn * 0.9 + 0.6);
    float y = uY0 - k * uTH + uA * (dome + rip) * (1.0 - 0.045 * k) + zn * uSlope;
    y += A.w * uTH * (1.0 + 2.5 * uLoose);
    y += uLoose * sin(xs * 4.0 + k * 0.8 + uT * 0.4) * 0.02 * uVisH;
    float z = zn * uFD * 0.5 + uLoose * (B.w - 0.5) * uFD * 0.14;
    return vec3(xs * uFW * 0.5, y, z);
  }
`;

const POINTS_VS = /* glsl */`
  attribute vec4 aA; attribute vec4 aB; attribute vec3 aFloat; attribute vec3 aName;
  uniform float uPhase, uForm, uConv, uRel, uDpr, uSize, uCamDist, uGate, uTop;
  varying float vAlpha; varying vec3 vCol;
  ${COMMON}
  void main() {
    float seed = aB.x;
    float k = aA.z;
    // --- float state (periodic in the cycle)
    float f1 = 1.0 + floor(h11(seed * 91.7) * 3.0), f2 = 1.0 + floor(h11(seed * 53.1) * 3.0), f3 = 1.0 + floor(h11(seed * 17.3) * 3.0);
    vec3 dr = vec3(sin(uPhase * f1 + seed * 40.0), sin(uPhase * f2 + aB.w * 33.0), sin(uPhase * f3 + aB.y * 27.0));
    vec3 pFloat = aFloat + dr * vec3(38.0 / f1, 30.0 / f2, 44.0 / f3);
    // --- formation (staggered, layer by layer)
    float sF = 0.30 * seed + 0.28 * (k / 9.0);
    float eF = quint(clamp((uForm - sF) / 0.42, 0.0, 1.0));
    vec3 pForm = formPos(aA, aB) + dr * 1.5;
    vec3 swirl = vec3(sin(aB.w * 50.0), cos(seed * 37.0), sin(aB.y * 23.0)) * 55.0 * sin(3.14159265 * eF);
    vec3 P = mix(pFloat, pForm, eF) + swirl;
    // --- converge onto the name
    vec3 pName = vec3(aName.xy, 0.0);
    float sC = 0.30 * seed + 0.20 * aName.z;
    float eC = quint(clamp((uConv - sC) / 0.5, 0.0, 1.0));
    vec3 dir = vec3(sin(aB.w * 61.0 + 1.3), cos(seed * 44.0), sin(aB.y * 19.0)) * (0.35 + 0.65 * seed);
    P = mix(P, pName, eC) + dir * 62.0 * sin(3.14159265 * eC);
    // --- release back into the float state
    float sR = 0.30 * aB.w + 0.18 * (1.0 - aName.z);
    float eR = quint(clamp((uRel - sR) / 0.52, 0.0, 1.0));
    vec3 dir2 = vec3(cos(aB.w * 29.0), sin(seed * 71.0 + 2.0), cos(aB.y * 13.0)) * (0.35 + 0.65 * aB.y);
    P = mix(P, pFloat, eR) + dir2 * 48.0 * sin(3.14159265 * eR);

    float nameness = eC * (1.0 - eR);
    float formness = eF * (1.0 - eC);
    vec4 mv = modelViewMatrix * vec4(P, 1.0);
    float depth = -mv.z;
    float f = clamp(abs(depth - uCamDist) / 380.0, 0.0, 1.0);
    f *= 1.0 - 0.85 * nameness;                       // crisp at the name
    float persp = uCamDist / depth;

    // stratigraphy: alternating band brightness, one very faint warm interval
    float lay = floor(k);
    float band = 0.55 + 0.45 * pow(0.5 + 0.5 * sin(lay * 2.3 + 0.8), 1.5);
    float pay = step(2.5, lay) * step(lay, 3.5);
    float xs = flowU(aA) * 2.0 - 1.0;
    float pulse = pow(0.5 + 0.5 * sin(xs * 9.0 - uT * 2.2 + aB.w * 6.2831), 6.0);

    float a = mix(0.22 + 0.42 * aB.w * aB.w, 0.88 * band, formness);
    a = mix(a, 0.8, nameness);
    a *= 1.0 + uLoose * formness * 1.5 * pulse;
    a *= mix(1.0, 0.42, f);
    a *= mix(1.0, edgeOf(aA), formness);
    a *= uGate;

    vec3 warm = vec3(1.0, 0.95, 0.88), cool = vec3(0.86, 0.92, 1.0);
    vec3 col = mix(cool, warm, aB.y);
    col = mix(col, vec3(1.0), 0.6 * nameness);
    col = mix(col, vec3(1.0, 0.82, 0.58), 0.20 * pay * formness);
    vCol = col; vAlpha = a;

    float size = mix(uSize, uSize * 1.3, nameness) * (1.0 + 1.5 * f * f) * persp;
    size *= 1.0 + 0.5 * step(0.975, seed) * formness * uLoose;      // the odd "data packet"
    gl_PointSize = max(1.25, size) * uDpr;
    gl_Position = projectionMatrix * mv;
  }
`;

const POINTS_FS = /* glsl */`
  varying float vAlpha; varying vec3 vCol;
  void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    float a = smoothstep(1.0, 0.25, d);
    gl_FragColor = vec4(vCol, a * vAlpha);
  }
`;

// hairline links between neighbours (they ride the same conveyor as the particles)
const LINK_VS = /* glsl */`
  attribute vec4 aA0; attribute vec4 aB0; attribute vec4 aA1; attribute vec4 aB1; attribute float aEnd;
  uniform float uLink, uMaxLen;
  varying float vAlpha;
  ${COMMON}
  void main() {
    vec3 p0 = formPos(aA0, aB0), p1 = formPos(aA1, aB1);
    vec3 P = aEnd < 0.5 ? p0 : p1;
    float len = length(p0 - p1);
    float e = min(edgeOf(aA0), edgeOf(aA1));
    float xs = flowU(aA0) * 2.0 - 1.0;
    float pulse = 0.55 + 0.45 * sin(xs * 7.0 - uT * 1.7 + aB0.w * 6.2831);
    vAlpha = uLink * e * (1.0 - smoothstep(0.35 * uMaxLen, uMaxLen, len)) * pulse;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(P, 1.0);
  }
`;
const LINK_FS = /* glsl */`
  varying float vAlpha;
  void main() { gl_FragColor = vec4(vec3(0.82, 0.88, 1.0), vAlpha); }
`;

// contour hairlines, drawn on along their length
const CONTOUR_VS = /* glsl */`
  attribute float aU; attribute float aLd; attribute float aAl;
  uniform float uDraw, uAlpha, uFW;
  varying float vU; varying float vP; varying float vAl;
  void main() {
    float p = clamp((uDraw - aLd * 0.5) / 0.5, 0.0, 1.0);
    p = p * p * (3.0 - 2.0 * p);
    vU = aU; vP = p;
    float xs = abs(position.x / (uFW * 0.5));
    vAl = aAl * uAlpha * smoothstep(1.0, 0.8, xs);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;
const CONTOUR_FS = /* glsl */`
  varying float vU; varying float vP; varying float vAl;
  void main() {
    if (vU > vP) discard;
    float head = smoothstep(0.07, 0.0, vP - vU) * (1.0 - step(0.999, vP));
    gl_FragColor = vec4(vec3(0.9, 0.94, 1.0), vAl * (1.0 + 2.2 * head));
  }
`;

// grid hairlines: radial "scan" reveal from the crest outward
const GRID_VS = /* glsl */`
  attribute float aR;
  uniform float uReveal, uAlpha, uFW;
  varying float vAl;
  void main() {
    float xs = abs(position.x / (uFW * 0.5));
    float on = smoothstep(uReveal, uReveal - 0.22, aR);
    float front = smoothstep(0.16, 0.0, abs(uReveal - 0.11 - aR)) * 0.8;
    vAl = uAlpha * on * (1.0 + front) * smoothstep(1.0, 0.8, xs);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;
const GRID_FS = /* glsl */`
  varying float vAl;
  void main() { gl_FragColor = vec4(vec3(0.82, 0.88, 0.98), vAl); }
`;
const TICK_VS = /* glsl */`
  attribute float aR;
  uniform float uReveal, uAlpha, uFW, uDpr;
  varying float vAl;
  void main() {
    float xs = abs(position.x / (uFW * 0.5));
    float on = smoothstep(uReveal - 0.04, uReveal - 0.2, aR);
    vAl = uAlpha * on * smoothstep(1.0, 0.8, xs);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = 7.0 * uDpr;
  }
`;
const TICK_FS = /* glsl */`
  varying float vAl;
  void main() {
    vec2 c = abs(gl_PointCoord - 0.5) * 7.0;       // css px from centre
    float arm = max(c.x, c.y);
    float thin = min(c.x, c.y);
    float a = (1.0 - smoothstep(0.35, 0.8, thin)) * (1.0 - smoothstep(2.6, 3.4, arm));
    gl_FragColor = vec4(vec3(0.9, 0.94, 1.0), a * vAl);
  }
`;

const BG_VS = /* glsl */`
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = vec4(position.xy, 1.0, 1.0); }
`;
const BG_FS = /* glsl */`
  varying vec2 vUv;
  uniform float uAspect, uHorizon, uLift;
  float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
  void main() {
    vec2 p = vec2((vUv.x - 0.5) * uAspect, vUv.y - 0.54);
    float r = length(p * vec2(0.82, 1.08));
    vec3 c = mix(vec3(0.074, 0.082, 0.098), vec3(0.010, 0.011, 0.014), smoothstep(0.0, 1.0, r));
    c *= 1.0 + 0.35 * uLift;
    float hy = 0.60;
    float g = exp(-pow((vUv.y - hy) * 8.0, 2.0)) * exp(-p.x * p.x * 1.1) * uHorizon * 0.055;
    c += vec3(0.56, 0.63, 0.75) * g;
    c += (hash(gl_FragCoord.xy) - 0.5) / 255.0;
    gl_FragColor = vec4(c, 1.0);
  }
`;

// ---- JS mirror of the fold, for lines -------------------------------------------------
function makeFold(D) {
  const shape = (xs, zn) => {
    const g = xs * 1.6;
    return Math.exp(-g * g) * (1 - 0.45 * zn * zn) + 0.16 * Math.sin(xs * 2.7 + zn * 0.9 + 0.6);
  };
  return (xs, zn, k) => D.y0 - k * D.th + D.A * shape(xs, zn) * (1 - 0.045 * k) + zn * D.slope;
}

function dims(width, height) {
  const visH = 2 * CAM_Z * Math.tan(THREE.MathUtils.degToRad(FOV) / 2);
  const visW = visH * (width / height);
  const fw = Math.max(visW * 0.9, visH * 0.82);
  return {
    visH, visW, fw,
    fd: Math.min(fw * 0.5, visH * 0.62),
    th: visH * 0.04, A: visH * 0.17, y0: visH * 0.1, slope: -visH * 0.04,
  };
}

// ---- prototype ------------------------------------------------------------------------
export default {
  name: 'Minimal Premium',
  blurb: 'Weightless particles settle into a folded stratigraphy, are traced by hairline contours and a sparse grid, flow as data, and converge into the name.',
  tech: 'Three.js · GPU-derived points · hairline GL lines · 24k particles',
  duration: 17,
  nameAt: 8.3,
  nameWeight: 600,
  phases: [
    { label: 'Float', start: 0, end: 1.6 },
    { label: 'Strata', start: 1.6, end: 3.6 },
    { label: 'Contours + Grid', start: 3.6, end: 5.4 },
    { label: 'Data network', start: 5.4, end: 6.8 },
    { label: 'Converge', start: 6.8, end: 8.3 },
  ],
  reveal: {
    colors: {
      name: '#F7F7F5',
      yijin: 'rgba(247,247,245,0.55)',
      sub: 'rgba(247,247,245,0.6)',
      accent: 'rgba(212,190,152,0.8)',
      rule: 'rgba(247,247,245,0.2)',
      glow: 'rgba(236,241,255,0.30)',
    },
  },

  create(ctx) {
    const low = ctx.quality === 'low';
    const renderer = createRenderer(ctx);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(FOV, ctx.width / ctx.height, 40, 5000);
    camera.position.set(0, 0, CAM_Z);
    const N = low ? 8000 : 24000;
    const T = ctx.timing;
    const dpr = renderer.getPixelRatio();

    // shared uniforms (one object so every material sees the same values)
    const U = {
      uT: { value: 0 }, uPhase: { value: 0 }, uDpr: { value: dpr }, uSize: { value: 1.7 },
      uForm: { value: 0 }, uConv: { value: 0 }, uRel: { value: 0 }, uGate: { value: 1 },
      uFlow: { value: 0 }, uLoose: { value: 0 }, uCamDist: { value: CAM_Z },
      uFW: { value: 1 }, uFD: { value: 1 }, uTH: { value: 1 }, uA: { value: 1 }, uY0: { value: 0 },
      uSlope: { value: 0 }, uVisH: { value: 1 },
    };
    const own = (extra) => Object.assign({}, U, extra);

    // background
    const bgMat = new THREE.ShaderMaterial({
      vertexShader: BG_VS, fragmentShader: BG_FS, depthTest: false, depthWrite: false,
      uniforms: { uAspect: { value: ctx.width / ctx.height }, uHorizon: { value: 0 }, uLift: { value: 0 } },
    });
    const bg = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), bgMat);
    bg.frustumCulled = false; bg.renderOrder = -10;
    scene.add(bg);

    // particles
    const pGeo = new THREE.BufferGeometry();
    const aA = new Float32Array(N * 4), aB = new Float32Array(N * 4), aFloat = new Float32Array(N * 3), aName = new Float32Array(N * 3);
    pGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    pGeo.setAttribute('aA', new THREE.BufferAttribute(aA, 4));
    pGeo.setAttribute('aB', new THREE.BufferAttribute(aB, 4));
    pGeo.setAttribute('aFloat', new THREE.BufferAttribute(aFloat, 3));
    pGeo.setAttribute('aName', new THREE.BufferAttribute(aName, 3));
    const pMat = new THREE.ShaderMaterial({
      vertexShader: POINTS_VS, fragmentShader: POINTS_FS, transparent: true, depthWrite: false, depthTest: false,
      blending: THREE.AdditiveBlending, uniforms: own({ uTop: { value: 0 } }),
    });
    const points = new THREE.Points(pGeo, pMat);
    points.frustumCulled = false; points.renderOrder = 3;

    // links
    const lMat = new THREE.ShaderMaterial({
      vertexShader: LINK_VS, fragmentShader: LINK_FS, transparent: true, depthWrite: false, depthTest: false,
      blending: THREE.AdditiveBlending, uniforms: own({ uLink: { value: 0 }, uMaxLen: { value: 60 } }),
    });
    let lGeo = null, links = null;

    // contours / grid / ticks
    const cMat = new THREE.ShaderMaterial({
      vertexShader: CONTOUR_VS, fragmentShader: CONTOUR_FS, transparent: true, depthWrite: false, depthTest: false,
      blending: THREE.AdditiveBlending, uniforms: { uDraw: { value: 0 }, uAlpha: { value: 0 }, uFW: U.uFW },
    });
    const gMat = new THREE.ShaderMaterial({
      vertexShader: GRID_VS, fragmentShader: GRID_FS, transparent: true, depthWrite: false, depthTest: false,
      blending: THREE.AdditiveBlending, uniforms: { uReveal: { value: 0 }, uAlpha: { value: 0 }, uFW: U.uFW },
    });
    const tMat = new THREE.ShaderMaterial({
      vertexShader: TICK_VS, fragmentShader: TICK_FS, transparent: true, depthWrite: false, depthTest: false,
      blending: THREE.AdditiveBlending, uniforms: { uReveal: { value: 0 }, uAlpha: { value: 0 }, uFW: U.uFW, uDpr: { value: dpr } },
    });
    let cGeo = null, gGeo = null, tGeo = null, cLines = null, gLines = null, ticks = null;

    const tmp = new THREE.Vector3();

    function disposeDynamic() {
      for (const o of [links, cLines, gLines, ticks]) if (o) scene.remove(o);
      for (const g of [lGeo, cGeo, gGeo, tGeo]) g?.dispose();
    }

    function build(c) {
      const W = c.width, H = c.height;
      const D = dims(W, H);
      const fold = makeFold(D);
      U.uFW.value = D.fw; U.uFD.value = D.fd; U.uTH.value = D.th; U.uA.value = D.A;
      U.uY0.value = D.y0; U.uSlope.value = D.slope; U.uVisH.value = D.visH;
      camera.aspect = W / H; camera.updateProjectionMatrix();
      bgMat.uniforms.uAspect.value = W / H;
      U.uDpr.value = tMat.uniforms.uDpr.value = renderer.getPixelRatio();

      // --- particles
      const r = rng(11);
      const gauss = () => (r() + r() + r() - 1.5) / 0.75;
      const keys = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        const front = r() < 0.42;
        const xn = r() * 2 - 1;
        let zn, k, off;
        if (front) { zn = 1; k = Math.floor(r() * (LAYERS - 1)) + 0.14 + r() * 0.68; off = 0; }
        else { zn = r() * 2 - 1; k = Math.floor(r() * LAYERS); off = gauss() * 0.02; }
        aA[i * 4] = xn; aA[i * 4 + 1] = zn; aA[i * 4 + 2] = k; aA[i * 4 + 3] = off;
        aB[i * 4] = r(); aB[i * 4 + 1] = r(); aB[i * 4 + 2] = front ? 1 : 0; aB[i * 4 + 3] = r();
        aFloat[i * 3] = gauss() * Math.max(D.visW, D.visH * 1.1) * 0.34;
        aFloat[i * 3 + 1] = gauss() * D.visH * 0.34;
        aFloat[i * 3 + 2] = -460 + r() * 820;
        keys[i] = xn + (r() - 0.5) * 0.35;
      }
      // name targets, matched left-to-right so the converge flows coherently
      const tg = sampleName(c.layout, N, { mode: 'mix' }).positions;
      const toWorld = pixelToPlane(camera, CAM_Z, W, H, 0);
      const tk = new Float32Array(N);
      for (let i = 0; i < N; i++) tk[i] = tg[i * 2] + (r() - 0.5) * c.layout.shortBox.w * 0.09;
      const pi = Array.from({ length: N }, (_, i) => i).sort((a, b) => keys[a] - keys[b]);
      const ti = Array.from({ length: N }, (_, i) => i).sort((a, b) => tk[a] - tk[b]);
      const bx = c.layout.shortBox;
      for (let j = 0; j < N; j++) {
        const p = pi[j], t = ti[j];
        toWorld(tg[t * 2], tg[t * 2 + 1], tmp);
        aName[p * 3] = tmp.x; aName[p * 3 + 1] = tmp.y;
        aName[p * 3 + 2] = Math.min(1, Math.max(0, (tg[t * 2] - bx.x) / bx.w));
      }
      for (const k of ['aA', 'aB', 'aFloat', 'aName']) pGeo.attributes[k].needsUpdate = true;
      pGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);

      // --- dynamic line geometry
      disposeDynamic();

      // links: neighbours on the upper sheets
      const cells = new Map();
      const NCX = 22, NCZ = 12;
      for (let i = 0; i < N; i++) {
        if (aB[i * 4 + 2] > 0.5 || aA[i * 4 + 2] > 4) continue;
        const cx = Math.floor((aA[i * 4] + 1) * 0.5 * NCX), cz = Math.floor((aA[i * 4 + 1] + 1) * 0.5 * NCZ);
        const key = (aA[i * 4 + 2] * 100 + cx) * 100 + cz;
        let l = cells.get(key); if (!l) cells.set(key, l = []); l.push(i);
      }
      const pool = [...cells.values()].filter((l) => l.length >= 2);
      const M = low ? 170 : 560;
      const lA0 = new Float32Array(M * 8), lB0 = new Float32Array(M * 8), lA1 = new Float32Array(M * 8), lB1 = new Float32Array(M * 8), lE = new Float32Array(M * 2);
      const rl = rng(5);
      for (let m = 0; m < M; m++) {
        const l = pool[Math.floor(rl() * pool.length)];
        const a = l[Math.floor(rl() * l.length)];
        let b = l[Math.floor(rl() * l.length)]; if (b === a) b = l[(l.indexOf(a) + 1) % l.length];
        for (let e = 0; e < 2; e++) {
          const o = (m * 2 + e) * 4;
          for (let q = 0; q < 4; q++) { lA0[o + q] = aA[a * 4 + q]; lB0[o + q] = aB[a * 4 + q]; lA1[o + q] = aA[b * 4 + q]; lB1[o + q] = aB[b * 4 + q]; }
          lE[m * 2 + e] = e;
        }
      }
      lGeo = new THREE.BufferGeometry();
      lGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(M * 2 * 3), 3));
      lGeo.setAttribute('aA0', new THREE.BufferAttribute(lA0, 4)); lGeo.setAttribute('aB0', new THREE.BufferAttribute(lB0, 4));
      lGeo.setAttribute('aA1', new THREE.BufferAttribute(lA1, 4)); lGeo.setAttribute('aB1', new THREE.BufferAttribute(lB1, 4));
      lGeo.setAttribute('aEnd', new THREE.BufferAttribute(lE, 1));
      lGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);
      lMat.uniforms.uMaxLen.value = D.fw / NCX * 1.5;
      links = new THREE.LineSegments(lGeo, lMat);
      links.frustumCulled = false; links.renderOrder = 2;

      // contours
      const cP = [], cU = [], cL = [], cA = [];
      let lineNo = 0;
      const pushLine = (pts, delay, alpha) => {
        // pts: flat xyz; builds segment pairs with arclength param
        const n = pts.length / 3;
        const len = new Float32Array(n);
        for (let i = 1; i < n; i++) len[i] = len[i - 1] + Math.hypot(pts[i * 3] - pts[i * 3 - 3], pts[i * 3 + 1] - pts[i * 3 - 2], pts[i * 3 + 2] - pts[i * 3 - 1]);
        const tot = len[n - 1] || 1;
        for (let i = 0; i < n - 1; i++) {
          for (const j of [i, i + 1]) { cP.push(pts[j * 3], pts[j * 3 + 1], pts[j * 3 + 2]); cU.push(len[j] / tot); cL.push(delay); cA.push(alpha); }
        }
        lineNo++;
      };
      // (a) cross-section stratigraphy on the front face
      const fronts = [0, 2, 4, 6, 9];
      fronts.forEach((k, idx) => {
        const pts = [];
        for (let s = 0; s <= 90; s++) { const xs = -1 + (2 * s) / 90; pts.push(xs * D.fw / 2, fold(xs, 1, k) , D.fd / 2); }
        pushLine(pts, idx * 0.1, 0.17);
      });
      // (b) elevation contours on the top surface (marching squares)
      {
        const cols = 100, rows = 56;
        const field = new Float32Array(cols * rows);
        let lo = 1e9, hi = -1e9;
        for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
          const v = fold(-1 + (2 * i) / (cols - 1), -1 + (2 * j) / (rows - 1), 0);
          field[j * cols + i] = v; lo = Math.min(lo, v); hi = Math.max(hi, v);
        }
        const levels = 7;
        for (let L = 0; L < levels; L++) {
          const iso = lo + (hi - lo) * (0.1 + (0.82 * (L + 0.5)) / levels);
          const polys = traceContours(field, cols, rows, iso).filter((p) => p.length >= 12);
          polys.forEach((p) => {
            const pts = [];
            for (let q = 0; q < p.length; q += 2) pts.push((-1 + (2 * p[q]) / (cols - 1)) * D.fw / 2, iso, (-1 + (2 * p[q + 1]) / (rows - 1)) * D.fd / 2);
            pushLine(pts, 0.25 + (L / levels) * 0.5, 0.14);
          });
        }
      }
      cGeo = new THREE.BufferGeometry();
      cGeo.setAttribute('position', new THREE.Float32BufferAttribute(cP, 3));
      cGeo.setAttribute('aU', new THREE.Float32BufferAttribute(cU, 1));
      cGeo.setAttribute('aLd', new THREE.Float32BufferAttribute(cL, 1));
      cGeo.setAttribute('aAl', new THREE.Float32BufferAttribute(cA, 1));
      cGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);
      cLines = new THREE.LineSegments(cGeo, cMat);
      cLines.frustumCulled = false; cLines.renderOrder = 1;

      // grid: draped on the top surface + a column grid on the cross-section
      const gP = [], gR = [], tP = [], tR = [];
      const NX = 13, NZ = 7;
      const rad = (xs, zn) => Math.hypot(xs * 0.85, zn * 0.9) * 0.95;
      const seg = (a, b) => { gP.push(...a.p, ...b.p); gR.push(a.r, b.r); };
      const surf = (xs, zn) => ({ p: [xs * D.fw / 2, fold(xs, zn, 0) + 0.8, zn * D.fd / 2], r: rad(xs, zn) });
      for (let j = 0; j < NZ; j++) {
        const zn = -1 + (2 * j) / (NZ - 1);
        for (let s = 0; s < 72; s++) seg(surf(-1 + (2 * s) / 72, zn), surf(-1 + (2 * (s + 1)) / 72, zn));
      }
      for (let i = 0; i < NX; i++) {
        const xs = -1 + (2 * i) / (NX - 1);
        for (let s = 0; s < 36; s++) seg(surf(xs, -1 + (2 * s) / 36), surf(xs, -1 + (2 * (s + 1)) / 36));
        for (let j = 0; j < NZ; j++) {
          const zn = -1 + (2 * j) / (NZ - 1);
          const q = surf(xs, zn); tP.push(...q.p); tR.push(q.r);
        }
        // cross-section columns
        const top = { p: [xs * D.fw / 2, fold(xs, 1, 0), D.fd / 2], r: rad(xs, 1) };
        const bot = { p: [xs * D.fw / 2, fold(xs, 1, 9), D.fd / 2], r: rad(xs, 1) };
        seg(top, bot);
      }
      gGeo = new THREE.BufferGeometry();
      gGeo.setAttribute('position', new THREE.Float32BufferAttribute(gP, 3));
      gGeo.setAttribute('aR', new THREE.Float32BufferAttribute(gR, 1));
      gGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);
      gLines = new THREE.LineSegments(gGeo, gMat);
      gLines.frustumCulled = false; gLines.renderOrder = 1;
      tGeo = new THREE.BufferGeometry();
      tGeo.setAttribute('position', new THREE.Float32BufferAttribute(tP, 3));
      tGeo.setAttribute('aR', new THREE.Float32BufferAttribute(tR, 1));
      tGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);
      ticks = new THREE.Points(tGeo, tMat);
      ticks.frustumCulled = false; ticks.renderOrder = 2;

      scene.add(cLines, gLines, ticks, links);
      if (!points.parent) scene.add(points);
    }
    build(ctx);

    const nameAt = T.nameAt;
    const sm = (a, b, t) => smoothstep(a, b, t);

    return {
      stats: `${N.toLocaleString()} particles · ${low ? 170 : 560} links`,
      update(t) {
        // --- timeline
        U.uT.value = t;
        U.uPhase.value = (TAU * t) / T.duration;
        U.uForm.value = ramp(t, 1.3, 3.7);
        U.uConv.value = ramp(t, 6.35, nameAt);
        U.uRel.value = ramp(t, nameAt + 0.7, nameAt + 5.3);
        const loose = sm(5.0, 6.6, t);
        U.uLoose.value = loose;
        const s = Math.max(0, Math.min(3.4, t - 5.0));
        U.uFlow.value = 0.03 * s * s;

        // global brightness: full -> name -> hand-off gap -> quiet -> back to opening level
        const dim = 0.42;
        U.uGate.value = (1 - sm(nameAt + 0.3, nameAt + 1.1, t))
          + dim * sm(nameAt + 1.3, nameAt + 3.8, t)
          + (1 - dim) * sm(T.out[0], T.duration, t);

        // structure layers
        const fade = (1 - 0.6 * sm(5.2, 6.2, t)) * (1 - sm(6.5, 7.5, t));
        cMat.uniforms.uDraw.value = ramp(t, 3.5, 5.7);
        cMat.uniforms.uAlpha.value = fade;
        const reveal = ramp(t, 4.0, 5.9) * 1.45;
        gMat.uniforms.uReveal.value = reveal; gMat.uniforms.uAlpha.value = 0.13 * fade;
        tMat.uniforms.uReveal.value = reveal; tMat.uniforms.uAlpha.value = 0.38 * fade;
        lMat.uniforms.uLink.value = 0.20 * sm(5.3, 6.2, t) * (1 - sm(6.5, 7.4, t));

        // background light
        bgMat.uniforms.uHorizon.value = sm(1.6, 4.2, t) * (1 - sm(6.2, 7.8, t));
        bgMat.uniforms.uLift.value = 0.3 * sm(5.5, nameAt, t) * (1 - sm(nameAt + 1.0, nameAt + 3.0, t));

        // camera: elevated and drifting for the formation, canonical pose at nameAt
        const lock = sm(6.0, nameAt, t) * (1 - sm(nameAt + 1.5, nameAt + 5.0, t));
        const env = 1 - lock;
        const ph = (TAU * t) / T.duration;
        const pitch = env * (0.31 + 0.03 * Math.sin(ph + 1.1));
        const yaw = env * 0.15 * Math.sin(ph + 0.5);
        const d = CAM_Z * (1 + 0.1 * env);
        const cp = Math.cos(pitch);
        camera.position.set(d * Math.sin(yaw) * cp, d * Math.sin(pitch), d * Math.cos(yaw) * cp);
        camera.lookAt(0, 0, 0);
        U.uCamDist.value = d;

        renderer.render(scene, camera);
      },
      resize(c) {
        renderer.setSize(c.width, c.height, false);
        build(c);
      },
      dispose() {
        disposeDynamic();
        pGeo.dispose(); pMat.dispose(); lMat.dispose(); cMat.dispose(); gMat.dispose(); tMat.dispose();
        bg.geometry.dispose(); bgMat.dispose();
        renderer.dispose(); renderer.domElement.remove();
      },
    };
  },
};
