#!/usr/bin/env node
// gauntlet.cjs - visual test loop for the Hero Lab.
//
//   node hero-lab/tools/gauntlet.cjs [A B ...] [--mobile] [--w 1440 --h 900] [--out DIR] [--times 0,4.5,9]
//
// For each prototype: captures deterministic frames at every phase (via
// ?ui=0&t=...), writes a labelled contact sheet, measures the loop seam
// (difference between the first and last frame), checks the settled name's
// alignment from the DOM, and samples frame cost while playing. Needs
// Playwright (preinstalled in the cloud container; otherwise `npm i -D playwright`).
// Note: headless Chromium renders WebGL in software (SwiftShader), so fps is
// only a relative signal; compare prototypes against each other, not 60.

const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { chromium } = require('playwright');

const ROOT = path.resolve(__dirname, '../..');
const args = process.argv.slice(2);
const flag = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const ids = args.filter((a, i) => /^[A-F]$/.test(a) && !String(args[i - 1] || '').startsWith('--'));
const mobile = args.includes('--mobile');
const W = Number(flag('--w', mobile ? 390 : 1440));
const H = Number(flag('--h', mobile ? 844 : 900));
const OUT = path.resolve(flag('--out', path.join(os.tmpdir(), 'hero-lab-gauntlet')));
const TIMES = flag('--times', null);
const noPerf = args.includes('--no-perf');
fs.mkdirSync(OUT, { recursive: true });

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.woff2': 'font/woff2', '.svg': 'image/svg+xml', '.webp': 'image/webp', '.jpg': 'image/jpeg', '.png': 'image/png', '.json': 'application/json' };
const server = http.createServer((req, res) => {
  let p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (p.endsWith('/')) p += 'index.html';
  const f = path.join(ROOT, p);
  if (!f.startsWith(ROOT) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); res.end('404'); return; }
  res.writeHead(200, { 'Content-Type': MIME[path.extname(f)] || 'application/octet-stream' });
  fs.createReadStream(f).pipe(res);
});

(async () => {
  await new Promise((r) => server.listen(0, r));
  const port = server.address().port;
  const browser = await chromium.launch({ args: ['--ignore-gpu-blocklist', '--enable-unsafe-swiftshader'] });
  const list = ids.length ? ids : ['A', 'B', 'C', 'D', 'E', 'F'];
  const summary = {};

  for (const id of list) {
    const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
    const errors = [];
    page.on('console', (m) => { if ((m.type() === 'error' || m.type() === 'warning') && !/GPU stall due to ReadPixels/.test(m.text())) errors.push(`${m.type()}: ${m.text()}`); });
    page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
    const t0 = Date.now();
    await page.goto(`http://localhost:${port}/hero-lab/?p=${id}&ui=0&pause=1&t=0`, { waitUntil: 'load' });
    await page.waitForFunction(() => window.heroLab && window.heroLab.ready, null, { timeout: 30000 });
    await page.evaluate(() => window.heroLab.ready);
    const loadMs = Date.now() - t0;

    const info = await page.evaluate(() => {
      const v = window.heroLab.viewports()[0];
      if (!v || !v.def) return null;
      return { name: v.def.name, duration: v.def.duration, nameAt: v.def.nameAt, phases: v.def.phases || [], timing: v.timing, stats: v.inst?.stats || '' };
    });
    if (!info) {
      const err = await page.evaluate(() => document.querySelector('.vp-error')?.textContent || 'no viewport');
      summary[id] = { error: err, errors };
      console.log(`\n[${id}] FAILED: ${err}`);
      errors.forEach((e) => console.log('   ', e));
      await page.close();
      continue;
    }

    const T = info.timing;
    let times;
    if (TIMES) times = TIMES.split(',').map(Number);
    else {
      const s = new Set([0]);
      for (const p of info.phases) { s.add(p.start + (p.end - p.start) * 0.35); s.add(p.start + (p.end - p.start) * 0.8); }
      [T.nameAt - 0.6, T.nameAt, T.nameAt + 0.5, (T.split[0] + T.split[1]) / 2, T.yijin[1], (T.read[0] + T.read[1]) / 2,
        (T.out[0] + T.out[1]) / 2, info.duration - 0.25, info.duration - 0.02].forEach((x) => s.add(x));
      times = [...s].filter((x) => x >= 0 && x < info.duration).sort((a, b) => a - b);
    }

    const shots = [];
    for (const t of times) {
      await page.evaluate((tt) => window.heroLab.renderAt(tt), t);
      const buf = await page.screenshot({ type: 'png' });
      const file = path.join(OUT, `${id}${mobile ? '-m' : ''}-t${t.toFixed(2).padStart(5, '0')}.png`);
      fs.writeFileSync(file, buf);
      shots.push({ t, file, b64: buf.toString('base64') });
    }

    // loop seam: mean abs pixel difference between the first and last frame (0..255)
    const first = shots.find((s) => s.t === 0);
    const lastShot = shots[shots.length - 1];
    const seam = first && lastShot !== first ? await page.evaluate(async ([a, b]) => {
      const load = (s) => new Promise((r) => { const i = new Image(); i.onload = () => r(i); i.src = 'data:image/png;base64,' + s; });
      const [ia, ib] = await Promise.all([load(a), load(b)]);
      const w = 240, h = Math.round((ia.height / ia.width) * 240);
      const px = (img) => { const c = document.createElement('canvas'); c.width = w; c.height = h; const x = c.getContext('2d'); x.drawImage(img, 0, 0, w, h); return x.getImageData(0, 0, w, h).data; };
      const da = px(ia), db = px(ib);
      let sum = 0; for (let i = 0; i < da.length; i += 4) sum += (Math.abs(da[i] - db[i]) + Math.abs(da[i + 1] - db[i + 1]) + Math.abs(da[i + 2] - db[i + 2])) / 3;
      return +(sum / (w * h)).toFixed(2);
    }, [first.b64, lastShot.b64]) : null;

    // settled-name alignment, read from the DOM during the read phase
    await page.evaluate((tt) => window.heroLab.renderAt(tt), (T.read[0] + T.read[1]) / 2);
    const align = await page.evaluate(() => {
      const box = (s) => document.querySelector(s).getBoundingClientRect();
      const d = box('.hl-daryl'), y = box('.hl-yijin'), t = box('.hl-tan'), sub = box('.hl-sub'), vp = document.querySelector('.vp-host').getBoundingClientRect();
      const left = d.left, right = t.right;
      return {
        centerOffsetPx: +(((left + right) / 2) - (vp.left + vp.width / 2)).toFixed(1),
        gaps: [+(y.left - d.right).toFixed(1), +(t.left - y.right).toFixed(1)],
        nameWidthPct: +(((right - left) / vp.width) * 100).toFixed(1),
        sameTop: Math.abs(d.top - t.top) < 0.5 && Math.abs(d.top - y.top) < 0.5,
        subtitleInside: sub.left >= vp.left - 0.5 && sub.right <= vp.right + 0.5,
        opacity: getComputedStyle(document.querySelector('.hl-daryl')).opacity,
      };
    });

    // perf: play in real time for 3 s
    let perf = null;
    if (!noPerf) {
      perf = await page.evaluate(async () => {
        const L = window.heroLab;
        L.state.elapsed = 0; L.state.playing = true;
        let n = 0; const t0 = performance.now();
        await new Promise((r) => { const f = () => { n++; if (performance.now() - t0 < 3000) requestAnimationFrame(f); else r(); }; requestAnimationFrame(f); });
        L.state.playing = false;
        return { fps: +(n / ((performance.now() - t0) / 1000)).toFixed(1), costMs: +L.viewports()[0].cost.toFixed(2) };
      });
    }

    // contact sheet
    const cols = mobile ? 6 : 4;
    const thumbW = mobile ? 260 : 440;
    const thumbH = Math.round((H / W) * thumbW);
    const sheet = await browser.newPage({ viewport: { width: cols * (thumbW + 8) + 8, height: 200 } });
    const label = (t) => {
      const ph = info.phases.find((p) => t >= p.start && t < p.end);
      if (ph) return ph.label;
      const R = [['Daryl Tan', T.nameAt, T.split[0]], ['Yi Jin', T.split[0], T.subtitle[0]], ['Subtitle', T.subtitle[0], T.read[0]], ['Read', T.read[0], T.out[0]], ['Return', T.out[0], T.duration]];
      const r = R.find(([, a, b]) => t >= a && t < b);
      return r ? r[0] : '';
    };
    await sheet.setContent(`<body style="margin:0;background:#222;font:12px monospace;color:#ddd;padding:8px">
      <div style="margin:0 0 8px">${id} · ${info.name} · ${W}x${H} · cycle ${info.duration}s · name at ${info.nameAt}s · seam ${seam}</div>
      <div style="display:grid;grid-template-columns:repeat(${cols},${thumbW}px);gap:8px">
      ${shots.map((s) => `<figure style="margin:0"><img src="data:image/png;base64,${s.b64}" style="width:${thumbW}px;height:${thumbH}px;display:block;object-fit:cover"><figcaption style="padding:3px 0">t=${s.t.toFixed(2)}s · ${label(s.t)}</figcaption></figure>`).join('')}
      </div></body>`);
    const contact = path.join(OUT, `${id}${mobile ? '-m' : ''}-contact.png`);
    await sheet.screenshot({ path: contact, fullPage: true });
    await sheet.close();

    summary[id] = { name: info.name, loadMs, seam, align, perf, stats: info.stats, frames: shots.length, contact, errors: [...new Set(errors)].slice(0, 12) };
    console.log(`\n[${id}] ${info.name}  load ${loadMs}ms  ${info.stats}`);
    console.log(`   contact: ${contact}`);
    console.log(`   seam (first vs last frame, 0-255): ${seam}   ${seam != null && seam > 6 ? '<-- visible jump at loop point?' : 'ok'}`);
    console.log(`   name: center offset ${align.centerOffsetPx}px, gaps ${align.gaps.join(' / ')}px, width ${align.nameWidthPct}%, aligned ${align.sameTop}, subtitle inside ${align.subtitleInside}`);
    if (perf) console.log(`   perf (software GL, relative): ${perf.fps} fps, ${perf.costMs} ms update`);
    if (errors.length) { console.log('   console:'); [...new Set(errors)].slice(0, 12).forEach((e) => console.log('     ', e)); }
    await page.close();
  }

  fs.writeFileSync(path.join(OUT, `summary${mobile ? '-m' : ''}.json`), JSON.stringify(summary, null, 2));
  await browser.close();
  server.close();
})().catch((e) => { console.error(e); server.close(); process.exit(1); });
