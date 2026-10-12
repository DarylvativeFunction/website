// molecule.js - Three.js CO2-capture scene for the pinned #intro (vanilla ES module).
// THREE comes from the global set by assets/vendor/three.min.js (r149, loaded with defer).

const ACCENT_FALLBACK = '#A31F34';
const RIPPLE = 2;                // original "ripple" prop default
const ACTIVE_WINDOW_MS = 1500;   // full frame rate this long after the last scroll
const IDLE_FRAME_MS = 50;        // ~20 fps ambient spin when idle
const RESIZE_DEBOUNCE_MS = 150;
const THREE_POLL_MS = 100;
const THREE_POLL_TRIES = 20;     // 2 s, then give up silently

const reducedQuery = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const seg = (v, a, b) => clamp((v - a) / (b - a || 1), 0, 1);
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

// Elements, cached once at start.
const els = {};

// Three.js objects.
const gl = {
  THREE: null, renderer: null, scene: null, camera: null, grp: null,
  co2: null, rock: null, rockMat: null, oxyMat: null, rimLight: null,
  rippleUniforms: null,
};

// Mutable runtime state (exposed on window.__mol for debugging).
const state = {
  running: false,
  raf: 0,
  idleTimer: 0,
  resizeTimer: 0,
  vw: window.innerWidth,
  vh: window.innerHeight,
  rawY: window.scrollY,
  scrollT: 0,
  sy: undefined,
  lastScrollY: undefined,
  lastT: 0,
  rot: { x: 0.15, y: 0.4 },
  vel: { x: 0, y: 0.4 },
  dragging: false,
  flung: false,
  hoverOn: false,
  rippleKick: 0,
  xray: 0,
  xrayTarget: 0,
  bursted: false,
  offscreen: false,
};

/* ---------- style writes (skip identical values) ---------- */

const styleCache = new WeakMap();

function setStyle(el, prop, value) {
  if (!el) return;
  let c = styleCache.get(el);
  if (!c) { c = {}; styleCache.set(el, c); }
  if (c[prop] === value) return;
  c[prop] = value;
  el.style[prop] = value;
}

function clearStyles(el, props) {
  if (!el) return;
  props.forEach((p) => { el.style[p] = ''; });
  styleCache.delete(el);
}

/* ---------- capability checks ---------- */

function prefersReduced() {
  return !!(reducedQuery && reducedQuery.matches);
}

function hasWebGL() {
  try {
    if (!window.WebGLRenderingContext) return false;
    const c = document.createElement('canvas');
    const ctx = c.getContext('webgl2') || c.getContext('webgl') || c.getContext('experimental-webgl');
    if (!ctx) return false;
    const lose = ctx.getExtension && ctx.getExtension('WEBGL_lose_context');
    if (lose) lose.loseContext();
    return true;
  } catch (err) {
    return false;
  }
}

function readAccent() {
  const v = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
  return /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(v) ? v : ACCENT_FALLBACK;
}

// three.min.js is a defer script ahead of this module, so THREE normally exists already.
// Still check on 'load' and poll briefly, then give up silently.
function waitForThree(onReady) {
  let done = false;
  let tries = 0;
  let timer = 0;
  const attempt = () => {
    if (done) return true;
    if (!window.THREE) return false;
    done = true;
    clearTimeout(timer);
    window.removeEventListener('load', attempt);
    onReady(window.THREE);
    return true;
  };
  const poll = () => {
    if (attempt()) return;
    if (++tries > THREE_POLL_TRIES) {
      window.removeEventListener('load', attempt);
      return;
    }
    timer = setTimeout(poll, THREE_POLL_MS);
  };
  window.addEventListener('load', attempt);
  poll();
}

/* ---------- setup ---------- */

function cacheElements() {
  const ids = {
    layer: 'mol-layer', hit: 'mol-hit', intro: 'intro', top: 'top',
    dock: 'mol-dock', dockLabel: 'mol-dock-label', tag: 'mol-tagline',
    tag2: 'mol-tagline2', rings: 'mol-rings', caption: 'mol-caption',
    glow: 'mol-glow', burst: 'mol-burst',
  };
  Object.entries(ids).forEach(([key, id]) => { els[key] = document.getElementById(id); });
}

// Renderer + camera; canvas fills the fixed layer.
function createRenderer(THREE) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(1.5, window.devicePixelRatio || 1));
  renderer.setSize(state.vw, state.vh, false);
  if ('outputEncoding' in renderer && THREE.sRGBEncoding !== undefined) renderer.outputEncoding = THREE.sRGBEncoding;
  const canvas = renderer.domElement;
  canvas.style.display = 'block';
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); teardown(); });
  els.layer.appendChild(canvas);
  gl.renderer = renderer;
  gl.camera = new THREE.PerspectiveCamera(35, state.vw / state.vh, 0.1, 100);
  gl.camera.position.z = 10;
}

// Procedural gradient environment map so the lacquer and steel have something to reflect.
function createEnvMap(THREE) {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 0, 128);
  grad.addColorStop(0, '#43474e');
  grad.addColorStop(0.42, '#101114');
  grad.addColorStop(0.55, '#5a606a');
  grad.addColorStop(1, '#0a0b0d');
  g.fillStyle = grad; g.fillRect(0, 0, 256, 128);
  g.fillStyle = 'rgba(255,255,255,0.85)'; g.fillRect(0, 46, 256, 5);
  const env = new THREE.CanvasTexture(c);
  env.mapping = THREE.EquirectangularReflectionMapping;
  return env;
}

// Scene, fog and the three-light rig (key, cool steel rim, crimson rim).
function createScene(THREE, accent) {
  const scene = new THREE.Scene();
  scene.environment = createEnvMap(THREE);
  scene.fog = new THREE.FogExp2(0x0d0e10, 0.05);
  scene.add(new THREE.AmbientLight(0xffffff, 0.38));
  const key = new THREE.DirectionalLight(0xffffff, 1.05);
  key.position.set(3, 4, 5);
  scene.add(key);
  const coolRim = new THREE.DirectionalLight(0x9fb8d8, 0.9);
  coolRim.position.set(-5, 3, -4);
  scene.add(coolRim);
  gl.rimLight = new THREE.PointLight(accent, 1.25);
  gl.rimLight.position.set(4.5, -2.5, -3.5);
  scene.add(gl.rimLight);
  gl.grp = new THREE.Group();
  scene.add(gl.grp);
  gl.scene = scene;
}

// CO2 ball-and-stick: graphite carbon, crimson oxygens, double steel bonds.
function buildCO2(THREE, accent, sph) {
  const co2 = new THREE.Group();
  const carbonMat = new THREE.MeshPhysicalMaterial({ color: 0x1b1d21, roughness: 0.34, metalness: 0.5, clearcoat: 1, clearcoatRoughness: 0.22 });
  const oxyMat = new THREE.MeshPhysicalMaterial({ color: accent, roughness: 0.26, metalness: 0.05, clearcoat: 1, clearcoatRoughness: 0.14, emissive: accent, emissiveIntensity: 0.05, envMapIntensity: 0.45 });
  const bondMat = new THREE.MeshStandardMaterial({ color: 0xc7cad1, roughness: 0.28, metalness: 0.85 });
  const bondGeo = new THREE.CylinderGeometry(0.05, 0.05, 0.85, 20);
  const carbon = new THREE.Mesh(sph, carbonMat);
  carbon.scale.setScalar(0.52);
  co2.add(carbon);
  [-1, 1].forEach((s) => {
    const o = new THREE.Mesh(sph, oxyMat);
    o.scale.setScalar(0.46);
    o.position.x = 1.3 * s;
    co2.add(o);
    [-1, 1].forEach((off) => {
      const b = new THREE.Mesh(bondGeo, bondMat);
      b.rotation.z = Math.PI / 2;
      b.position.set(0.65 * s, 0.09 * off, 0);
      co2.add(b);
    });
  });
  gl.oxyMat = oxyMat;
  gl.co2 = co2;
  return co2;
}

// Layered-sine displacement used to rough up the rock.
function rockNoise(x, y, z) {
  return Math.sin(x * 2.3 + y * 1.7) * Math.cos(z * 2.1 + x * 1.1) * 0.5 +
    Math.sin(x * 5.1 + z * 4.3) * Math.cos(y * 4.7 + z * 0.7) * 0.28 +
    Math.sin(x * 11.3 + y * 9.1 + z * 8.7) * 0.12;
}

// Porous basalt: displaced icosahedron with a crater at the top pole.
function buildRockGeometry(THREE) {
  const geo = new THREE.IcosahedronGeometry(1, 5);
  const pos = geo.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i).normalize();
    let r = 1 + 0.24 * rockNoise(v.x * 1.5, v.y * 1.5, v.z * 1.5);
    const cap = Math.max(0, (v.y - 0.45) / 0.55);
    r -= cap * cap * 0.46;
    v.multiplyScalar(r * 1.95);
    pos.setXYZ(i, v.x, v.y, v.z);
  }
  geo.computeVertexNormals();
  return geo;
}

// Rock material with a vertex ripple injected via onBeforeCompile.
function buildRock(THREE, accent) {
  const rockMat = new THREE.MeshStandardMaterial({ color: 0x232528, roughness: 0.93, metalness: 0.07, flatShading: true, transparent: true, opacity: 1, emissive: accent, emissiveIntensity: 0.035 });
  const uniforms = { uTime: { value: 0 }, uAmp: { value: 0 } };
  rockMat.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = uniforms.uTime;
    shader.uniforms.uAmp = uniforms.uAmp;
    shader.vertexShader = 'uniform float uTime; uniform float uAmp;\n' + shader.vertexShader.replace(
      '#include <begin_vertex>',
      '#include <begin_vertex>\n' +
      'float dRip = distance(position, vec3(0.0, 1.4, 0.0));\n' +
      'transformed += normalize(position) * sin(dRip * 6.0 - uTime * 4.2) * exp(-dRip * 0.85) * uAmp;'
    );
  };
  const rock = new THREE.Mesh(buildRockGeometry(THREE), rockMat);
  rock.position.y = -8.5;
  gl.rockMat = rockMat;
  gl.rippleUniforms = uniforms;
  gl.rock = rock;
  return rock;
}

function buildMolecule(THREE, accent) {
  const sph = new THREE.SphereGeometry(1, 48, 32);
  gl.grp.add(buildCO2(THREE, accent, sph));
  gl.grp.add(buildRock(THREE, accent));
}

/* ---------- pointer: drag with inertia, hover x-ray ---------- */

function setupDrag() {
  const hit = els.hit;
  if (!hit) return;
  let lx = 0, ly = 0, lt = 0;

  hit.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    state.dragging = true;
    state.flung = false;
    try { hit.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
    hit.style.cursor = 'grabbing';
    lx = e.clientX; ly = e.clientY; lt = performance.now();
    state.vel.x = 0; state.vel.y = 0;
    e.preventDefault();
    wake();
  });

  hit.addEventListener('pointermove', (e) => {
    if (!state.dragging) return;
    const now = performance.now();
    const dtm = Math.max(8, now - lt);
    const dx = e.clientX - lx, dy = e.clientY - ly;
    state.rot.y += dx * 0.008;
    state.rot.x += dy * 0.005;
    state.vel.y = clamp(dx / dtm * 8, -6, 6);
    state.vel.x = clamp(dy / dtm * 5, -4, 4);
    state.rippleKick = Math.min(0.45, state.rippleKick + (Math.abs(dx) + Math.abs(dy)) * 0.0016);
    lx = e.clientX; ly = e.clientY; lt = now;
    wake();
  });

  hit.addEventListener('pointerenter', (e) => {
    if (e.pointerType && e.pointerType !== 'mouse') return;
    state.hoverOn = true;
    state.rippleKick = Math.min(0.4, state.rippleKick + 0.14);
    wake();
  });
  hit.addEventListener('pointerleave', () => { state.hoverOn = false; wake(); });

  const end = () => {
    if (!state.dragging) return;
    state.dragging = false;
    state.flung = true;
    hit.style.cursor = 'grab';
    wake();
  };
  hit.addEventListener('pointerup', end);
  hit.addEventListener('pointercancel', end);
  hit.addEventListener('lostpointercapture', end);
}

/* ---------- per-frame pieces ---------- */

// All layout reads happen first, before any style writes this frame.
function readLayout() {
  const { vw, vh } = state;
  const introH = els.intro.offsetHeight;
  const dockVisible = !!(els.dock && els.dock.offsetWidth > 0);
  return {
    vw, vh, introH,
    pinEnd: Math.max(1, introH - vh),
    dockRect: dockVisible ? els.dock.getBoundingClientRect() : null,
    heroGone: els.top ? els.top.getBoundingClientRect().bottom < 0 : false,
  };
}

// Inertial smoothing: the scene eases toward the real scroll position.
function smoothScroll(dt) {
  const rawY = state.rawY;
  state.sy = state.sy === undefined ? rawY : state.sy + (rawY - state.sy) * (1 - Math.exp(-dt * 14));
  if (Math.abs(rawY - state.sy) < 0.4) state.sy = rawY;
  return state.sy;
}

// Overlay copy and decor driven by pin progress p.
function applyOverlay(p) {
  const d = seg(p, 0.05, 0.2);
  setStyle(els.tag, 'opacity', (1 - d).toFixed(3));
  setStyle(els.tag, 'transform', `scale(${(1 + d * 0.08).toFixed(4)})`);

  const in2 = seg(p, 0.7, 0.78);
  setStyle(els.tag2, 'opacity', (in2 * (1 - seg(p, 0.88, 0.97))).toFixed(3));
  setStyle(els.tag2, 'transform', `translateY(${((1 - in2) * 14).toFixed(1)}px)`);

  setStyle(els.rings, 'opacity', (1 - seg(p, 0.18, 0.36)).toFixed(3));
  const fade = (1 - seg(p, 0.8, 0.96)).toFixed(3);
  setStyle(els.caption, 'opacity', fade);
  setStyle(els.glow, 'opacity', fade);
}

// Capture choreography: lift -> rock rises -> descend -> absorb.
function placeModels(lift, rockRise, desc, L) {
  const topY = 3.0, seatY = 0.12;
  gl.co2.position.y = lift * topY + desc * (seatY - topY);
  gl.co2.scale.setScalar(1 - 0.1 * lift - 0.42 * desc);
  // Rest the rock fully below the viewport whatever its shape: the group is
  // scaled so 5.5 world units span `size` px, so convert the needed pixel
  // offset (just over half the viewport plus the rock radius) back to units.
  const size = Math.min(L.vw, L.vh) * 0.5;
  const restY = Math.max(8.5, (L.vh * 0.55 + size * 0.55) * 5.5 / size);
  gl.rock.position.y = -restY + rockRise * restY;
}

// Contact burst ring (Web Animations) and the rock surface ripple.
function updateRipple(now, dt, rockRise, desc) {
  if (desc >= 0.999 && !state.bursted) {
    state.bursted = true;
    state.rippleKick = 0.55;
    const burst = els.burst;
    if (burst && burst.animate) {
      burst.animate(
        [{ opacity: 0.85, transform: 'translate(-50%, -50%) scale(0.35)' }, { opacity: 0, transform: 'translate(-50%, -50%) scale(1.6)' }],
        { duration: 950, easing: 'cubic-bezier(0.1, 0.5, 0.3, 1)' }
      );
    }
  }
  if (desc < 0.9) state.bursted = false;
  state.rippleKick *= Math.exp(-dt * 2.2);
  const u = gl.rippleUniforms;
  u.uTime.value = now * 0.001;
  const ambient = 0.016 + 0.012 * (0.5 + 0.5 * Math.sin(now * 0.0011));
  u.uAmp.value = (ambient * rockRise + state.rippleKick) * RIPPLE;
}

// Hover x-ray: once seated, the rock turns translucent and the stored CO2 glows.
function updateXray(dt, desc) {
  state.xrayTarget = (state.hoverOn && desc > 0.98) ? 1 : 0;
  state.xray += (state.xrayTarget - state.xray) * Math.min(1, dt * 6);
  gl.rockMat.opacity = 1 - 0.87 * state.xray;
  gl.rockMat.depthWrite = state.xray < 0.5;
  gl.oxyMat.emissiveIntensity = 0.05 + 0.55 * state.xray;
}

// Screen placement: center of the stage -> #mol-dock beside the name.
function computePlacement(L, y) {
  const cx = L.vw / 2, cy = L.vh / 2;
  const baseSize = Math.min(L.vw, L.vh) * 0.5;
  const travelStart = L.pinEnd * 0.82, travelEnd = L.introH;
  let px = cx, py = cy, size = baseSize, t = 0, opacity = 1;
  if (L.dockRect) {
    const dr = L.dockRect;
    t = ease(seg(y, travelStart, travelEnd));
    px = cx + (dr.left + dr.width / 2 - cx) * t;
    py = cy + (dr.top + dr.height / 2 - cy) * t;
    size = baseSize + (dr.width * 0.95 - baseSize) * t;
  } else {
    opacity = 1 - seg(y, travelStart, L.pinEnd);
  }
  return { cx, cy, px, py, size, t, opacity };
}

// Layer visibility and the drag hit-area follow the molecule.
function applyPlacement(P, offscreen) {
  setStyle(els.layer, 'opacity', P.opacity.toFixed(3));
  setStyle(els.layer, 'visibility', offscreen ? 'hidden' : 'visible');
  if (els.dock) setStyle(els.dockLabel, 'opacity', P.t >= 0.995 ? '1' : '0');
  const hit = els.hit;
  if (!hit) return;
  if (offscreen) {
    setStyle(hit, 'display', 'none');
    return;
  }
  const s = P.size.toFixed(0) + 'px';
  setStyle(hit, 'display', 'block');
  setStyle(hit, 'width', s);
  setStyle(hit, 'height', s);
  setStyle(hit, 'transform', `translate(${(P.px - P.size / 2).toFixed(1)}px, ${(P.py - P.size / 2).toFixed(1)}px)`);
}

// Playful rotation: scroll kick, breathing idle spin, fling decay, wobble.
function updateSpin(dt, ts, y, docked) {
  const dy = y - (state.lastScrollY !== undefined ? state.lastScrollY : y);
  state.lastScrollY = y;
  if (state.dragging) return;
  const { vel, rot } = state;
  vel.y = clamp(vel.y + dy * 0.0006, -6, 6);
  const playful = 0.32 + 0.5 * Math.pow(Math.sin(ts * 0.16), 2) + (docked ? 0.25 * Math.sin(ts * 0.47) : 0);
  vel.y += (playful - vel.y) * (state.flung ? 0.006 : 0.03);
  if (state.flung) {
    vel.y *= 0.988;
    vel.x *= 0.95;
    if (Math.abs(vel.y - 0.4) < 0.5 && Math.abs(vel.x) < 0.05) state.flung = false;
  } else {
    vel.x *= 0.9;
  }
  rot.y += vel.y * dt;
  rot.x += vel.x * dt;
  if (!state.flung) {
    const wobble = 0.16 * Math.sin(ts * 0.21) + 0.05 * Math.sin(ts * 0.9);
    rot.x += (wobble - rot.x) * 0.02;
  }
}

// Apply rotation and map screen px to world units, then draw.
function renderAt(P, ts, absorbed, vh) {
  const { rot } = state;
  // damp x-tilt while the molecule is high above the rock so the stack stays vertical
  gl.grp.rotation.set(rot.x * (0.15 + 0.85 * absorbed), rot.y, Math.sin(ts * 0.11) * 0.06);
  gl.co2.rotation.y = -rot.y * 0.45;
  const cam = gl.camera;
  const worldH = 2 * cam.position.z * Math.tan(cam.fov * Math.PI / 360);
  const wpp = worldH / vh;
  gl.grp.position.set((P.px - P.cx) * wpp, -(P.py - P.cy) * wpp, 0);
  gl.grp.scale.setScalar(Math.max(0.001, P.size * wpp / 5.5));
  gl.renderer.render(gl.scene, cam);
}

// One frame of the scene. Returns true when the molecule is on screen.
function tick(now) {
  const dt = Math.min(0.05, Math.max(0, (now - state.lastT) / 1000));
  state.lastT = now;
  const L = readLayout();
  const y = smoothScroll(dt);

  const p = clamp(y / L.pinEnd, 0, 1.2);
  const lift = ease(seg(p, 0.08, 0.28));
  const rockRise = ease(seg(p, 0.16, 0.42));
  const desc = ease(seg(p, 0.46, 0.68));
  const absorbed = seg(p, 0.62, 0.72);

  applyOverlay(p);
  placeModels(lift, rockRise, desc, L);
  updateRipple(now, dt, rockRise, desc);
  updateXray(dt, desc);

  const P = computePlacement(L, y);
  const offscreen = (P.py + P.size / 2) < -40 || P.opacity <= 0.01 || L.heroGone;
  state.offscreen = offscreen;
  applyPlacement(P, offscreen);
  if (offscreen) {
    state.lastScrollY = y; // no spin jolt when it comes back into view
    return false;
  }

  const ts = now * 0.001;
  updateSpin(dt, ts, y, P.t >= 0.995);
  renderAt(P, ts, absorbed, L.vh);
  return true;
}

/* ---------- frame scheduling ---------- */

// Full frame rate while something is moving; otherwise ~20 fps or asleep when offscreen.
function isActive(now) {
  return now - state.scrollT < ACTIVE_WINDOW_MS ||
    state.dragging || state.hoverOn || state.flung ||
    Math.abs(state.rawY - state.sy) > 0.4 ||
    state.rippleKick > 0.02 ||
    Math.abs(state.xrayTarget - state.xray) > 0.005;
}

function frame(now) {
  state.raf = 0;
  if (!state.running) return;
  let visible = true;
  try { visible = tick(now); } catch (err) { /* keep the loop alive */ }
  if (!state.running) return;
  if (isActive(now)) {
    wake();
  } else if (visible) {
    const spent = performance.now() - now;
    state.idleTimer = setTimeout(wake, Math.max(0, IDLE_FRAME_MS - 16 - spent));
  }
  // offscreen and idle: sleep until a scroll, resize or pointer event wakes the loop
}

function wake() {
  if (!state.running || state.raf) return;
  clearTimeout(state.idleTimer);
  state.idleTimer = 0;
  state.raf = requestAnimationFrame(frame);
}

/* ---------- window listeners ---------- */

// Passive: record position and time only; all work happens in the rAF.
function onScroll() {
  state.rawY = window.scrollY;
  state.scrollT = performance.now();
  wake();
}

function applyResize() {
  state.vw = window.innerWidth;
  state.vh = window.innerHeight;
  if (!gl.renderer) return;
  gl.renderer.setPixelRatio(Math.min(1.5, window.devicePixelRatio || 1));
  gl.renderer.setSize(state.vw, state.vh, false);
  gl.camera.aspect = state.vw / state.vh;
  gl.camera.updateProjectionMatrix();
  state.rawY = window.scrollY;
  wake();
}

function onResize() {
  clearTimeout(state.resizeTimer);
  state.resizeTimer = setTimeout(applyResize, RESIZE_DEBOUNCE_MS);
}

function onReducedChange() {
  if (prefersReduced()) teardown();
}

/* ---------- lifecycle ---------- */

// Stop everything and hand the intro back to the static CSS presentation.
function teardown() {
  state.running = false;
  cancelAnimationFrame(state.raf);
  clearTimeout(state.idleTimer);
  clearTimeout(state.resizeTimer);
  state.raf = 0;
  window.removeEventListener('scroll', onScroll);
  window.removeEventListener('resize', onResize);
  document.documentElement.classList.remove('mol-on');
  clearStyles(els.layer, ['opacity', 'visibility']);
  els.layer.style.visibility = 'hidden';
  clearStyles(els.hit, ['display', 'width', 'height', 'transform', 'cursor']);
  clearStyles(els.tag, ['opacity', 'transform']);
  clearStyles(els.tag2, ['opacity', 'transform']);
  clearStyles(els.rings, ['opacity']);
  clearStyles(els.caption, ['opacity']);
  clearStyles(els.glow, ['opacity']);
  clearStyles(els.dockLabel, ['opacity']);
}

function start(THREE) {
  if (state.running || gl.renderer || prefersReduced()) return;
  cacheElements();
  if (!els.layer || !els.intro) return;
  if (!hasWebGL()) return;
  gl.THREE = THREE;
  const accent = new THREE.Color(readAccent());
  try {
    createRenderer(THREE);
    createScene(THREE, accent);
    buildMolecule(THREE, accent);
  } catch (err) {
    if (gl.renderer && gl.renderer.domElement.parentNode) gl.renderer.domElement.remove();
    gl.renderer = null;
    return;
  }
  setupDrag();

  state.running = true;
  state.rawY = window.scrollY;
  state.lastT = performance.now();
  document.documentElement.classList.add('mol-on');
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onResize, { passive: true });
  if (reducedQuery) {
    if (reducedQuery.addEventListener) reducedQuery.addEventListener('change', onReducedChange);
    else if (reducedQuery.addListener) reducedQuery.addListener(onReducedChange);
  }
  window.__mol = { renderer: gl.renderer, scene: gl.scene, camera: gl.camera, state };
  wake();
}

if (!prefersReduced()) waitForThree(start);
