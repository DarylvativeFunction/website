// _reference.js - the smallest complete prototype. Not shown in the lab; it
// documents the contract (see README.md) and the pattern most prototypes use:
// GPU-morphed points whose targets are sampled from the shared name layout.

import * as THREE from 'three';
import { createRenderer, pixelOrthoCamera, ramp, easeInOutCubic, GLSL, rng } from '../core/stage.js';
import { sampleName } from '../core/text.js';

export default {
  name: 'Reference',
  blurb: 'Points drift, then gather onto the name glyphs.',
  tech: 'Three.js · points shader',
  duration: 15,
  nameAt: 6.5,
  nameWeight: 700,
  phases: [
    { label: 'Drift', start: 0, end: 2.5 },
    { label: 'Gather', start: 2.5, end: 6.5 },
  ],
  reveal: { colors: { glow: 'rgba(120,200,255,0.5)' } },

  create(ctx) {
    const renderer = createRenderer(ctx);
    const scene = new THREE.Scene();
    let camera = pixelOrthoCamera(ctx.width, ctx.height);
    const N = ctx.quality === 'low' ? 5000 : 14000;

    const geo = new THREE.BufferGeometry();
    const aStart = new Float32Array(N * 2), aTarget = new Float32Array(N * 2), aSeed = new Float32Array(N);
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    geo.setAttribute('aStart', new THREE.BufferAttribute(aStart, 2));
    geo.setAttribute('aTarget', new THREE.BufferAttribute(aTarget, 2));
    geo.setAttribute('aSeed', new THREE.BufferAttribute(aSeed, 1));

    const mat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uT: { value: 0 }, uMorph: { value: 0 }, uAlpha: { value: 1 }, uSize: { value: 2 }, uDpr: { value: renderer.getPixelRatio() } },
      vertexShader: `
        attribute vec2 aStart; attribute vec2 aTarget; attribute float aSeed;
        uniform float uT, uMorph, uSize, uDpr; varying float vA;
        void main() {
          float m = clamp(uMorph * 1.25 - aSeed * 0.25, 0.0, 1.0);
          m = m * m * (3.0 - 2.0 * m);
          vec2 drift = vec2(sin(uT * 0.4 + aSeed * 40.0), cos(uT * 0.33 + aSeed * 23.0)) * 14.0 * (1.0 - m);
          vec2 p = mix(aStart + drift, aTarget, m);
          vA = 0.35 + 0.65 * m;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 0.0, 1.0);
          gl_PointSize = uSize * uDpr * (1.0 + (1.0 - m) * aSeed);
        }`,
      fragmentShader: `
        uniform float uAlpha; varying float vA; ${GLSL.softDot}
        void main() { float a = softDot(gl_PointCoord) * vA * uAlpha; gl_FragColor = vec4(vec3(0.75, 0.88, 1.0) * a, a); }`,
    });
    scene.add(new THREE.Points(geo, mat));

    function build() {
      const r = rng(3);
      for (let i = 0; i < N; i++) { aStart[i * 2] = r() * ctx.width; aStart[i * 2 + 1] = r() * ctx.height; aSeed[i] = r(); }
      aTarget.set(sampleName(ctx.layout, N, { mode: 'mix' }).positions);
      for (const k of ['aStart', 'aTarget', 'aSeed']) geo.attributes[k].needsUpdate = true;
      geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);
    }
    build();

    const T = ctx.timing;
    return {
      stats: `${N.toLocaleString()} points`,
      update(t) {
        // gather by nameAt, hand over to HTML, then drift back out for a seamless loop
        const gather = easeInOutCubic(ramp(t, 2.5, T.nameAt));
        const release = easeInOutCubic(ramp(t, T.out[0], T.duration));
        mat.uniforms.uMorph.value = gather * (1 - release);
        mat.uniforms.uAlpha.value = 1 - 0.85 * ramp(t, T.nameAt + 0.1, T.nameAt + 0.8) * (1 - ramp(t, T.out[0], T.duration));
        mat.uniforms.uT.value = t;
        renderer.render(scene, camera);
      },
      resize(c) {
        renderer.setSize(c.width, c.height, false);
        camera = pixelOrthoCamera(c.width, c.height);
        build();
      },
      dispose() { geo.dispose(); mat.dispose(); renderer.dispose(); renderer.domElement.remove(); },
    };
  },
};
