// stage.js - small helpers shared by prototypes. Nothing here is mandatory;
// a prototype may use WebGL, 2D canvas or SVG as suits it.

import * as THREE from 'three';
export { rng } from './text.js';

export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
/** 0..1 progress of t through [a, b], clamped. */
export const ramp = (t, a, b) => clamp((t - a) / (b - a));
export const smoothstep = (a, b, t) => { const x = ramp(t, a, b); return x * x * (3 - 2 * x); };
export const easeInOutCubic = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
export const easeOutCubic = (x) => 1 - Math.pow(1 - x, 3);
export const easeInCubic = (x) => x * x * x;
export const easeOutExpo = (x) => (x >= 1 ? 1 : 1 - Math.pow(2, -10 * x));
export const easeInOutSine = (x) => -(Math.cos(Math.PI * x) - 1) / 2;
/** Bell 0 -> 1 -> 0 across [a, d], flat-topped between b and c. */
export const window4 = (t, a, b, c, d) => Math.min(ramp(t, a, b), 1 - ramp(t, c, d));

/** Device pixel ratio cap for a quality tier. */
export function dprFor(quality) {
  const d = window.devicePixelRatio || 1;
  return quality === 'low' ? Math.min(d, 1) : Math.min(d, 1.75);
}

/**
 * WebGL renderer sized to ctx.width x ctx.height, canvas appended to ctx.container.
 * The canvas is absolutely positioned behind the reveal overlay.
 */
export function createRenderer(ctx, opts = {}) {
  const renderer = new THREE.WebGLRenderer({
    antialias: opts.antialias ?? ctx.quality !== 'low',
    alpha: opts.alpha ?? false,
    powerPreference: 'high-performance',
    preserveDrawingBuffer: false,
  });
  renderer.setPixelRatio(dprFor(ctx.quality));
  renderer.setSize(ctx.width, ctx.height, false);
  renderer.setClearColor(opts.clearColor ?? 0x000000, opts.clearAlpha ?? 1);
  const cv = renderer.domElement;
  cv.className = 'hl-canvas';
  cv.setAttribute('aria-hidden', 'true');
  Object.assign(cv.style, { position: 'absolute', inset: '0', width: '100%', height: '100%', display: 'block' });
  ctx.container.appendChild(cv);
  return renderer;
}

/** Visible height of a perspective camera at distance d. */
export function visibleHeightAt(camera, d) {
  return 2 * d * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
}

/**
 * Map container pixels to world space on the plane z = planeZ, for a perspective
 * camera in its canonical "name" pose: positioned at (0, 0, camZ) looking at
 * (0, 0, planeZ), with no roll. Use it to place particle targets so they land
 * exactly under the HTML typography. Returns (px, py, out?) -> THREE.Vector3.
 */
export function pixelToPlane(camera, camZ, width, height, planeZ = 0) {
  const visH = visibleHeightAt(camera, camZ - planeZ);
  const s = visH / height;
  return (px, py, out = new THREE.Vector3()) => out.set((px - width / 2) * s, -(py - height / 2) * s, planeZ);
}

/** Orthographic camera whose world units are CSS pixels, origin top-left, y down. */
export function pixelOrthoCamera(width, height) {
  const cam = new THREE.OrthographicCamera(0, width, 0, height, -2000, 2000);
  cam.position.z = 1000;
  return cam;
}

/** Shared GLSL snippets. */
export const GLSL = {
  // soft round sprite with a hot core; use in a points fragment shader
  softDot: `
    float softDot(vec2 pc) {
      float d = length(pc - 0.5) * 2.0;
      float core = smoothstep(1.0, 0.0, d);
      return core * core;
    }`,
  // hash + value noise
  noise: `
    float hash13(vec3 p) { p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }
    float vnoise(vec3 p) {
      vec3 i = floor(p); vec3 f = fract(p); f = f * f * (3.0 - 2.0 * f);
      return mix(mix(mix(hash13(i), hash13(i + vec3(1,0,0)), f.x), mix(hash13(i + vec3(0,1,0)), hash13(i + vec3(1,1,0)), f.x), f.y),
                 mix(mix(hash13(i + vec3(0,0,1)), hash13(i + vec3(1,0,1)), f.x), mix(hash13(i + vec3(0,1,1)), hash13(i + vec3(1,1,1)), f.x), f.y), f.z);
    }`,
};
