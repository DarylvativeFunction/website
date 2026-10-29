// e-voxel.js - Prototype E, "Voxel Collapse".
// A layered anticline reservoir block, built from one InstancedMesh of cubes, first reads as a smooth
// continuous body, then reveals its cell discretisation (edges), opens into an exploded view, and the
// cells tumble along arcs into a volumetric voxel rendering of "Daryl Tan". All motion is computed in the
// vertex shader from per-instance attributes and a handful of uniforms, so update(t) is a pure function of t.

import * as THREE from 'three';
import { EffectComposer, RenderPass, UnrealBloomPass } from 'three/addons';
import { createRenderer, pixelToPlane, ramp, easeInOutCubic, easeOutCubic, smoothstep, rng } from '../core/stage.js';
import { rasterizeName } from '../core/text.js';

const FOV = 30;
const AZ_S = 0.62, EL_S = 0.42;          // opening 3/4 isometric orbit pose

export default {
  name: 'Voxel Collapse',
  blurb: 'A layered reservoir block reveals its cell grid, opens into an exploded view, and the voxels tumble into a volumetric name.',
  tech: 'Three.js · InstancedMesh · GPU-driven voxels · 7.8k cubes',
  duration: 18,
  nameAt: 9.0,
  nameWeight: 700,
  phases: [
    { label: 'Reservoir', start: 0, end: 2.6 },
    { label: 'Grid', start: 2.6, end: 4.4 },
    { label: 'Explode', start: 4.4, end: 5.8 },
    { label: 'Collapse', start: 5.8, end: 9.0 },
  ],
  reveal: {
    colors: {
      name: '#F4F6F6', yijin: 'rgba(205,228,236,0.62)', sub: 'rgba(205,228,236,0.72)',
      accent: '#E2A44C', rule: 'rgba(170,215,230,0.28)', glow: 'rgba(120,205,225,0.5)',
    },
  },
  posterTime: 8.8,

  create(ctx) {
    const low = ctx.quality === 'low';
    const T = ctx.timing;
    const [NX, NY, NZ] = low ? [19, 13, 9] : [30, 20, 13];
    const N = NX * NY * NZ;
    const LAYERS = low ? 2 : 3;
    const TC0 = 5.7; const EXK = ctx.width < 700 ? 0.22 : 0.58;                       // collapse starts

    const renderer = createRenderer(ctx, { clearColor: 0x070b10 });
    const scene = new THREE.Scene();
    let camera = new THREE.PerspectiveCamera(FOV, ctx.width / ctx.height, 10, 8000);
    let camZ = 1000;

    let composer = null;
    if (!low) {
      composer = new EffectComposer(renderer);
      composer.setPixelRatio(renderer.getPixelRatio());
      composer.setSize(ctx.width, ctx.height);
    }

    // ---------------------------------------------------------------- shared materials
    const uni = {
      uMorph: { value: 0 }, uExplode: { value: 0 }, uGap: { value: 0 }, uEdge: { value: 0 },
      uFade: { value: 0 }, uDebris: { value: 0 }, uBuild: { value: 1 }, uFront: { value: 0 }, uFrontAmp: { value: 0 },
      uCs: { value: 20 }, uSpan: { value: 100 }, uTime: { value: 0 }, uDim: { value: 1 },
    };
    const mat = new THREE.ShaderMaterial({
      uniforms: uni,
      vertexShader: /* glsl */`
        attribute vec3 aStart; attribute vec2 aTilt; attribute vec3 aTarget; attribute vec3 aAxis;
        attribute vec4 aRand; attribute vec3 aColor; attribute vec3 aInfo;
        uniform float uDebris, uMorph, uExplode, uGap, uFade, uBuild, uCs, uSpan, uTime, uFront, uFrontAmp;
        varying vec3 vN; varying vec3 vWPos; varying vec3 vLocal; varying vec3 vCol; varying float vE; varying float vGlow;

        vec3 rodrigues(vec3 v, vec3 k, float a) {
          float c = cos(a), s = sin(a);
          return v * c + cross(k, v) * s + k * dot(k, v) * (1.0 - c);
        }
        vec3 bed(vec3 v, vec2 t) {   // small bed tilt: about x then about z
          float cx = cos(t.x), sx = sin(t.x), cz = cos(t.y), sz = sin(t.y);
          v = vec3(v.x, v.y * cx - v.z * sx, v.y * sx + v.z * cx);
          return vec3(v.x * cz - v.y * sz, v.x * sz + v.y * cz, v.z);
        }
        void main() {
          // rest pose: reservoir cell, spread apart by uGap, layers lifted apart by uExplode
          vec3 p0 = aStart * (1.0 + 0.34 * uGap);
          p0.y += (aInfo.x - 0.5) * uSpan * uExplode;
          float restSize = uCs * mix(1.06, 0.80, uGap);

          // collapse: staggered per-instance progress
          float m = clamp((uMorph - aRand.x * 0.45) / 0.55, 0.0, 1.0);
          float e = m * m * (3.0 - 2.0 * m);
          float bump = sin(3.14159265 * m);
          vec3 side = normalize(cross(aAxis, vec3(0.0, 1.0, 0.001)));
          vec3 pos = mix(p0, aTarget, e) + (vec3(0.0, 1.0, 0.0) + side * 0.55 * sign(aRand.y)) * aRand.z * bump;
          pos.xy += vec2(sin(uTime * 0.6 + aRand.w * 40.0), cos(uTime * 0.5 + aRand.w * 23.0)) * 12.0 * e * (1.0 - aInfo.z);

          float size = mix(restSize, aInfo.y, e);
          // handoff: letter voxels shrink away, staggered
          float f = clamp(uFade * 1.6 - aRand.w * 0.6, 0.0, 1.0);
          f = f * f * (3.0 - 2.0 * f);
          size *= (1.0 - f);
          size *= mix(1.0, 1.0 - uDebris, step(aInfo.z, 0.9));
          pos.z -= f * 30.0;
          // loop return: block re-grows from the bottom layer upward
          float b = clamp(uBuild * 1.7 - aInfo.x * 0.7, 0.0, 1.0);
          b = b * b * (3.0 - 2.0 * b);
          size *= b;
          pos.z -= (1.0 - b) * (1.0 - b) * 500.0;

          // orientation: bed tilt (settles to zero) after a tumble about aAxis that also settles
          float ang = aRand.y * 6.2831853 * e;   // whole turns, so the cube lands axis-aligned
          vec3 lp = position * size;
          vec3 nl = normal;
          lp = bed(lp, aTilt * (1.0 - e)); nl = bed(nl, aTilt * (1.0 - e));
          lp = rodrigues(lp, aAxis, ang);  nl = rodrigues(nl, aAxis, ang);

          vec3 world = pos + lp;
          vN = nl; vWPos = world; vLocal = position;
          vE = e;
          vec3 col = aColor;
          // letters pale to a cool tint except amber, debris dimmer
          float amber = clamp((aColor.r - aColor.b) * 3.0, 0.0, 1.0);
          float lum = dot(aColor, vec3(0.3, 0.5, 0.2));
          vec3 tint = vec3(0.74, 0.92, 0.98) * (0.78 + 0.5 * lum);
          col = mix(col, tint, 0.6 * e * (1.0 - amber));
          col *= mix(1.0, aInfo.z, e);
          vCol = col;
          float dx = (aStart.x - uFront) / (uCs * 3.2);
          vGlow = uFrontAmp * exp(-dx * dx) * (1.0 - e);
          gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.0);
        }`,
      fragmentShader: /* glsl */`
        uniform float uEdge, uDim;
        varying vec3 vN; varying vec3 vWPos; varying vec3 vLocal; varying vec3 vCol; varying float vE; varying float vGlow;
        void main() {
          vec3 n = normalize(vN);
          vec3 v = normalize(cameraPosition - vWPos);
          vec3 L = normalize(vec3(-0.6, 0.85, 0.3));
          float diff = max(dot(n, L), 0.0);
          float hemi = n.y * 0.5 + 0.5;
          float fill = max(dot(n, normalize(vec3(0.7, 0.1, 0.6))), 0.0);
          float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
          vec3 col = vCol * (0.05 + 0.06 * hemi + 1.05 * diff + 0.04 * fill);
          col *= mix(1.0, 1.7, vE);
          col += vec3(0.30, 0.62, 0.72) * rim * 0.15;
          col += vec3(0.2, 0.6, 0.7) * vGlow * 0.4;
          // cube edges: pale hairlines on the grid, dark creases on the voxel letters
          vec3 q = abs(vLocal);
          float a = max(q.x, max(q.y, q.z)), c = min(q.x, min(q.y, q.z));
          float mid = q.x + q.y + q.z - a - c;
          float edgeF = smoothstep(0.43, 0.5, mid);
          vec3 lineCol = mix(vec3(0.45, 0.72, 0.82), vec3(0.01, 0.03, 0.05), vE);
          col = mix(col, lineCol, edgeF * uEdge * mix(0.6, 0.7, vE));
          col *= uDim;
          gl_FragColor = vec4(col, 1.0);
        }`,
    });

    // fullscreen backdrop: slate radial gradient
    const bgMat = new THREE.ShaderMaterial({
      depthTest: false, depthWrite: false,
      uniforms: { uAsp: { value: ctx.width / ctx.height }, uGlow: { value: 0 } },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.9999, 1.0); }',
      fragmentShader: `varying vec2 vUv; uniform float uAsp;
        void main(){ vec2 p = (vUv - vec2(0.5, 0.52)) * vec2(uAsp, 1.0);
          float d = length(p + vec2(0.1, -0.12));
          vec3 c = mix(vec3(0.011, 0.016, 0.024), vec3(0.0012, 0.0016, 0.0024), smoothstep(0.0, 0.9, d));
          gl_FragColor = vec4(c, 1.0); }`,
    });
    const bg = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), bgMat);
    bg.frustumCulled = false; bg.renderOrder = -100;
    scene.add(bg);

    const ghostMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uAlpha: { value: 0 }, uBand: { value: new THREE.Vector2(0, 0.2) } },
      vertexShader: 'uniform vec2 uBand; attribute vec3 color; varying vec3 vC; void main(){ gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); float a = smoothstep(uBand.y, uBand.y + 0.3, abs(gl_Position.y / gl_Position.w - uBand.x)); vC = color * mix(0.12, 1.0, a); }',
      fragmentShader: 'uniform float uAlpha; varying vec3 vC; void main(){ gl_FragColor = vec4(vC * uAlpha, 1.0); }',
    });

    // ---------------------------------------------------------------- scene content (depends on layout)
    let mesh = null, ghost = null, ghostGroup = null, geo = null, ghostGeo = null, nLetters = 0, nDebris = 0;
    let W = 600, cs = 20, A = 60;

    const domeF = (u, v) => Math.exp(-((u - 0.08) * (u - 0.08) / 0.5 + v * v / 0.85));
    const ampF = (k) => 0.32 + 0.68 * (k / (NZ - 1));

    function build() {
      const L = ctx.layout, w = ctx.width, h = ctx.height;
      camZ = (h / 2) / Math.tan(THREE.MathUtils.degToRad(FOV / 2));
      camera.aspect = w / h; camera.near = 10; camera.far = camZ * 4; camera.updateProjectionMatrix();
      bgMat.uniforms.uAsp.value = w / h;
      const toWorld = pixelToPlane(camera, camZ, w, h, 0);
      const narrow = w < 700;
      W = narrow ? w * 0.8 : Math.min(w * 0.5, h * 0.9);
      cs = W / NX; A = 0.13 * W;
      const D = cs * NY;
      uni.uCs.value = cs;
      uni.uSpan.value = (NZ - 1) * cs * 0.8;
      const yOff = -A * 0.28;

      // ----- letter voxels: supersampled coverage of the 'short' raster on a regular grid
      const c = L.fontSize / (low ? 12 : 15);
      const r = 4 / c;
      const ras = rasterizeName(L, { which: 'short', res: r });
      const box = L.shortBox;
      const gx0 = Math.floor(box.x / c) * c, gy0 = Math.floor((box.y - c) / c) * c;
      const gcols = Math.ceil((box.w + 2 * c) / c), grows = Math.ceil((box.h + 3 * c) / c);
      const cells = [];
      for (let j = 0; j < grows; j++) for (let i = 0; i < gcols; i++) {
        let s = 0;
        for (let b = 0; b < 4; b++) for (let a = 0; a < 4; a++) {
          const px = Math.floor((gx0 + (i + (a + 0.5) / 4) * c) * r), py = Math.floor((gy0 + (j + (b + 0.5) / 4) * c) * r);
          if (px >= 0 && py >= 0 && px < ras.cols && py < ras.rows && ras.data[py * ras.cols + px] > 127) s++;
        }
        if (s >= 7) cells.push([gx0 + (i + 0.5) * c, gy0 + (j + 0.5) * c]);
      }
      const layers = Math.max(1, Math.min(LAYERS, Math.floor(N * 0.5 / Math.max(1, cells.length))));
      const targets = []; // {x,y,z,size,dim}
      const tmp = new THREE.Vector3();
      for (let l = 0; l < layers; l++) for (const [px, py] of cells) {
        toWorld(px, py, tmp);
        targets.push({ x: tmp.x, y: tmp.y, z: -c * (0.5 + l), size: c * 0.97, dim: 1, tx: px });
      }
      nLetters = targets.length;

      // ----- reservoir cells
      const rand = rng(11);
      const lay = new Float32Array(NZ);
      for (let k = 0; k < NZ; k++) lay[k] = (rand() - 0.5) * 0.34;
      for (let k = 1; k < NZ - 1; k++) lay[k] = 0.5 * lay[k] + 0.25 * (lay[k - 1] + lay[k + 1]);
      const cellsR = new Array(N);
      let n = 0;
      const zb = (u, v, k) => (k - (NZ - 1) / 2) * cs + A * ampF(k) * domeF(u, v) + yOff;
      for (let k = 0; k < NZ; k++) for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
        const bx = (i - (NX - 1) / 2) * cs, by = (j - (NY - 1) / 2) * cs;
        const u = bx / (W / 2), v = by / (D / 2);
        const y = zb(u, v, k);
        const eps = cs * 0.5;
        const sx = (zb((bx + eps) / (W / 2), v, k) - zb((bx - eps) / (W / 2), v, k)) / (2 * eps);
        const sy = (zb(u, (by + eps) / (D / 2), k) - zb(u, (by - eps) / (D / 2), k)) / (2 * eps);
        const dt = k / (NZ - 1), dm = domeF(u, v);
        const pn = Math.sin(u * 2.9 + k * 0.9 + 1.3) * Math.cos(v * 2.3 - k * 0.55);
        const phi = Math.min(1, Math.max(0, 0.5 + lay[k] + 0.2 * pn + (rand() - 0.5) * 0.07 + 0.08 * (dt - 0.5)));
        const so = Math.pow(dt, 4) * Math.pow(dm, 2.2) * 1.25;
        cellsR[n++] = { x: bx, y, z: -by, tx: sx, ty: sy, k, dt, phi, so, rnd: rand() };
      }
      // order by x so cells map to letters left-to-right (monotone, coherent flow)
      const orderCells = cellsR.map((q, i) => [q.x + (rand() - 0.5) * cs * 3, i]).sort((a, b) => a[0] - b[0]).map((o) => o[1]);
      targets.sort((a, b) => (a.x + (rand() - 0.5) * c * 3) - (b.x + (rand() - 0.5) * c * 3));
      const isLetter = new Uint8Array(N); const tgtOf = new Int32Array(N).fill(-1);
      for (let q = 0; q < nLetters; q++) {
        const idx = orderCells[Math.min(N - 1, Math.floor((q + 0.5) * N / nLetters))];
        isLetter[idx] = 1; tgtOf[idx] = q;
      }
      // debris: a sparse dim scatter of cubes behind the letters; the rest shrink away
      nDebris = Math.round(N * 0.055);
      const leftovers = orderCells.filter((i) => !isLetter[i]);
      const stride = Math.max(1, Math.floor(leftovers.length / nDebris));
      const hx = box.w / 2 + 60, hy = box.h / 2 + 60;
      for (let q = 0; q < leftovers.length; q++) {
        const idx = leftovers[q];
        if (q % stride === 0 && nDebris-- > 0) {
          let px, py, tries = 0;
          do { px = rand() * w; py = rand() * h; tries++; }
          while (tries < 20 && Math.abs(px - L.centerX) < hx && Math.abs(py - L.centerY) < hy);
          toWorld(px, py, tmp);
          const dz = -(250 + rand() * 650);
          targets.push({ x: tmp.x * (1 - dz / camZ), y: tmp.y * (1 - dz / camZ), z: dz, size: c * (0.6 + rand() * 1.1), dim: 0.28, tx: px });
          tgtOf[idx] = targets.length - 1;
        }
      }
      nDebris = targets.length - nLetters;

      // ----- instanced geometry
      const base = new THREE.BoxGeometry(1, 1, 1);
      geo = new THREE.InstancedBufferGeometry();
      geo.index = base.index; geo.attributes.position = base.attributes.position; geo.attributes.normal = base.attributes.normal;
      geo.instanceCount = N;
      const aStart = new Float32Array(N * 3), aTilt = new Float32Array(N * 2), aTarget = new Float32Array(N * 3), aAxis = new Float32Array(N * 3);
      const aRandA = new Float32Array(N * 4), aColor = new Float32Array(N * 3), aInfo = new Float32Array(N * 3);
      const ramp3 = (p, o) => {
        const A0 = [0.012, 0.03, 0.075], B0 = [0.035, 0.15, 0.19], C0 = [0.36, 0.52, 0.56];
        p = Math.pow(p, 1.9);
        const s = p < 0.5 ? p * 2 : (p - 0.5) * 2, a = p < 0.5 ? A0 : B0, b = p < 0.5 ? B0 : C0;
        for (let q = 0; q < 3; q++) o[q] = a[q] + (b[q] - a[q]) * s;
      };
      const col = [0, 0, 0], amber = [0.72, 0.36, 0.07];
      const WXmax = w * 0.5;
      for (let i = 0; i < N; i++) {
        const q = cellsR[i];
        aStart.set([q.x, q.y, q.z], i * 3);
        aTilt.set([Math.atan(q.ty), Math.atan(q.tx)], i * 2);
        ramp3(q.phi, col);
        const am = smoothstep(0.08, 0.85, q.so);
        for (let z = 0; z < 3; z++) col[z] += (amber[z] - col[z]) * am * 0.78;
        const stripe = 0.9 + 0.1 * Math.sin(q.k * 2.1);
        aColor.set([col[0] * stripe, col[1] * stripe, col[2] * stripe], i * 3);
        let t = targets[tgtOf[i]];
        let tx, ty, tz, sz, dim;
        if (t) { tx = t.x; ty = t.y; tz = t.z; sz = t.size; dim = t.dim; }
        else { // vanish: drift outward and shrink to nothing
          const ang = rand() * 6.283;
          tx = q.x * 1.8 + Math.cos(ang) * cs * 6; ty = q.y * 1.5 + 120 + rand() * 200; tz = q.z * 1.8 + (rand() - 0.5) * 200; sz = 0; dim = 1;
        }
        aTarget.set([tx, ty, tz], i * 3);
        // random unit axis
        const th = rand() * 6.283, zz = rand() * 2 - 1, rr = Math.sqrt(1 - zz * zz);
        aAxis.set([rr * Math.cos(th), zz, rr * Math.sin(th)], i * 3);
        const txn = t ? (t.tx - L.shortBox.x) / L.shortBox.w : q.x / W + 0.5;
        const delay = Math.min(1, Math.max(0, 0.72 * txn + 0.28 * rand()));
        aRandA.set([delay, (rand() < 0.5 ? -1 : 1) * (1 + Math.floor(rand() * 3)), (0.12 + rand() * 0.22) * W * (t ? 1 : 0.8), rand()], i * 4);
        aInfo.set([q.dt, sz, dim], i * 3);
      }
      const addA = (name, arr, size) => geo.setAttribute(name, new THREE.InstancedBufferAttribute(arr, size));
      addA('aStart', aStart, 3); addA('aTilt', aTilt, 2); addA('aTarget', aTarget, 3); addA('aAxis', aAxis, 3);
      addA('aRand', aRandA, 4); addA('aColor', aColor, 3); addA('aInfo', aInfo, 3);
      mesh = new THREE.Mesh(geo, mat);
      mesh.frustumCulled = false;
      scene.add(mesh);

      // ----- ghost lattice (faint, slowly rotating, shown during the read phase)
      const node = (i, j, k, o) => {
        const bx = (i - NX / 2) * cs, by = (j - NY / 2) * cs, u = bx / (W / 2), v = by / (D / 2);
        o[0] = bx; o[1] = (k - NZ / 2) * cs + A * ampF(Math.min(NZ - 1, k)) * domeF(u, v) + yOff; o[2] = -by;
        o[3] = k >= NZ - 1 ? smoothstep(0.3, 0.9, domeF(u, v)) : 0;
      };
      const segs = [];
      const a1 = [0, 0, 0, 0], b1 = [0, 0, 0, 0];
      const stepI = 1;
      for (let k = 0; k <= NZ; k++) for (let j = 0; j <= NY; j++) for (let i = 0; i <= NX; i++) {
        const edgeish = (i === 0 || i === NX) + (j === 0 || j === NY) + (k === 0 || k === NZ);
        node(i, j, k, a1);
        const dirs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        for (const [di, dj, dk] of dirs) {
          const ni = i + di, nj = j + dj, nk = k + dk;
          if (ni > NX || nj > NY || nk > NZ) continue;
          // interior lines sparse, outer shell and every 3rd line dense
          const onShell = (di === 0 ? (i === 0 || i === NX) : false) || (dj === 0 ? (j === 0 || j === NY) : false) || (dk === 0 ? (k === 0 || k === NZ) : false);
          if (!onShell && !((i + j + k) % 3 === 0 && edgeish === 0)) continue;
          if (!onShell && (di ? (j % 3 || k % 3) : dj ? (i % 3 || k % 3) : (i % 3 || j % 3))) continue;
          node(ni, nj, nk, b1);
          segs.push(a1[0], a1[1], a1[2], a1[3], b1[0], b1[1], b1[2], b1[3]);
        }
      }
      const gp = new Float32Array(segs.length / 8 * 6), gc = new Float32Array(segs.length / 8 * 6);
      for (let s = 0; s < segs.length / 8; s++) {
        for (let e2 = 0; e2 < 2; e2++) {
          const o = s * 8 + e2 * 4, d = s * 6 + e2 * 3;
          gp[d] = segs[o]; gp[d + 1] = segs[o + 1]; gp[d + 2] = segs[o + 2];
          const am = segs[o + 3];
          gc[d] = 0.45 + 0.45 * am; gc[d + 1] = 0.72 - 0.1 * am; gc[d + 2] = 0.85 - 0.5 * am;
        }
      }
      ghostGeo = new THREE.BufferGeometry();
      ghostGeo.setAttribute('position', new THREE.BufferAttribute(gp, 3));
      ghostGeo.setAttribute('color', new THREE.BufferAttribute(gc, 3));
      ghost = new THREE.LineSegments(ghostGeo, ghostMat);
      ghost.frustumCulled = false;
      ghostGroup = new THREE.Group();
      ghostGroup.add(ghost);
      ghostGroup.position.set(0, 0, -(narrow ? 120 : 260));
      { const y0 = L.baseline - L.capHeight, y1 = L.subtitle.top + 50;
        ghostMat.uniforms.uBand.value.set(1 - (y0 + y1) / h, (y1 - y0) / h + 0.04); }
      const gs = narrow ? 1.1 : 1.45;
      ghostGroup.scale.setScalar(gs);
      scene.add(ghostGroup);

      if (composer) {
        composer.passes.length = 0;
        composer.addPass(new RenderPass(scene, camera));
        composer.addPass(new UnrealBloomPass(new THREE.Vector2(w, h), 0.12, 0.5, 0.95));
      }
    }

    function teardown() {
      if (mesh) { scene.remove(mesh); geo.dispose(); mesh = null; }
      if (ghostGroup) { scene.remove(ghostGroup); ghostGeo.dispose(); ghostGroup = null; }
    }
    build();

    // ---------------------------------------------------------------- pure-in-t update
    const out0 = T.out[0], dur = T.duration, nameAt = T.nameAt, split0 = T.split[0];
    const lerp = (a, b, s) => a + (b - a) * s;
    function pose(t) {
      let az, el, dist;
      const azHold = AZ_S + 0.05 * TC0, elHold = EL_S + 0.012 * TC0;
      if (t < TC0) { az = AZ_S + 0.05 * t; el = EL_S + 0.012 * t; dist = camZ * (0.98 + EXK * easeInOutCubic(ramp(t, 3.9, 5.4))); }
      else if (t < out0) {
        const s = easeInOutCubic(ramp(t, TC0, nameAt));
        az = lerp(azHold, 0, s); el = lerp(elHold, 0, s); dist = lerp(camZ * (0.98 + EXK), camZ, s);
      } else {
        const s = easeInOutCubic(ramp(t, out0, dur));
        az = lerp(0, AZ_S, s); el = lerp(0, EL_S, s); dist = lerp(camZ, camZ * 0.98, s);
      }
      const ce = Math.cos(el);
      camera.position.set(dist * Math.sin(az) * ce, dist * Math.sin(el), dist * Math.cos(az) * ce);
      camera.up.set(0, 1, 0);
      camera.lookAt(0, 0, 0);
    }

    return {
      stats: `${N.toLocaleString()} cubes (${nLetters} letter + ${nDebris} debris)`,
      update(t) {
        const inOut = t >= out0;
        const bs = T.out[1] - 0.3;
        uni.uTime.value = t;
        uni.uEdge.value = easeInOutCubic(ramp(t, 2.5, 3.9)) * (1 - easeInOutCubic(ramp(t, bs, dur - 0.1)));
        uni.uGap.value = inOut ? 0 : easeInOutCubic(ramp(t, 3.7, 5.1));
        uni.uExplode.value = inOut ? 0 : easeInOutCubic(ramp(t, 4.3, 5.6));
        uni.uDim.value = inOut ? 0.12 + 0.88 * ramp(t, bs, dur - 0.1) : 1;
        uni.uDebris.value = inOut ? 0 : easeInOutCubic(ramp(t, nameAt - 1.5, nameAt - 0.3));
        uni.uMorph.value = inOut ? 0 : ramp(t, TC0, nameAt);
        uni.uFade.value = inOut ? 0 : ramp(t, nameAt + 0.3, split0 - 0.05);
        uni.uBuild.value = inOut ? easeInOutCubic(ramp(t, bs, dur - 0.25)) : 1;
        // fluid front sweeping through the intact block
        const fw = ramp(t, 0.4, 4.6);
        uni.uFront.value = lerp(-W * 0.7, W * 0.7, fw);
        uni.uFrontAmp.value = smoothstep(0.0, 0.12, fw) * (1 - smoothstep(0.82, 1.0, fw)) * (1 - ramp(t, 3.4, 4.6)) * 0.4;
        ghostMat.uniforms.uAlpha.value = 0.05 * smoothstep(nameAt + 0.8, split0 + 1.2, t) * (1 - smoothstep(out0, T.out[1], t));
        ghostGroup.rotation.y = 0.5 + (t - nameAt) * 0.04;
        ghostGroup.rotation.x = 0.1;
        pose(t);
        if (composer) composer.render(); else renderer.render(scene, camera);
      },
      resize(c) {
        renderer.setSize(c.width, c.height, false);
        if (composer) { composer.setSize(c.width, c.height); }
        teardown();
        build();
      },
      dispose() {
        teardown();
        mat.dispose(); bgMat.dispose(); ghostMat.dispose(); bg.geometry.dispose();
        if (composer) composer.dispose();
        renderer.dispose(); renderer.domElement.remove();
      },
    };
  },
};
