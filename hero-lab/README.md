# Hero Lab

A standalone playground for the homepage hero, served at `/hero-lab/`. The live
homepage (`/index.html`) is untouched; nothing here is linked from it.

Concept for all six prototypes: **living reservoir → computational grid → data
points → network → name reveal**, ending on "Daryl Yi Jin Tan" and the subtitle
"Petroleum Engineering · Economics · Machine Learning · Data Analytics".

## Running it

Any static server from the repo root, e.g.

```bash
python3 -m http.server 8000
```

then open <http://localhost:8000/hero-lab/>.

Keys: `Space` play/pause · `R` replay · `1`–`6` select · `←`/`→` seek 0.5s
(`Shift` 0.05s) · `[` `]` speed · `C` side by side · `G` all six · `M` mobile frame.

URL parameters: `p=A..F`, `t=<seconds>` (implies paused), `pause=1`, `speed=`,
`mode=single|compare|grid`, `frame=fill|mobile`, `q=high|low`, `ui=0` (bare
viewport, for capture).

Scores, shortlist stars and notes are saved in the browser (localStorage).

## Layout

```
hero-lab/
  index.html, lab.css, lab.js   gallery shell, clock, timeline, scorecard
  core/text.js                  name layout, glyph raster + point sampling, contours, distance field
  core/reveal.js                shared HTML name reveal (GSAP timeline, seeked by the clock)
  core/stage.js                 renderer + pixel<->world helpers, easing, GLSL snippets
  prototypes/registry.js        the six prototypes
  prototypes/[a-f]-*.js         one module per prototype
  prototypes/_reference.js      smallest complete prototype (documentation)
  prototypes/review.js          Claude's evaluation shown in the leaderboard
  tools/gauntlet.cjs            Playwright visual test loop
  vendor/                       three r186, three addons, gsap 3.15 (minified ESM)
```

## Prototype contract

A prototype module default-exports:

```js
export default {
  name: 'Cinematic Reservoir',
  blurb: 'One or two sentences.',
  tech: 'Three.js · custom shaders · 24k points',
  duration: 18,          // seconds per loop
  nameAt: 9.2,           // "Daryl Tan" fully formed by the prototype's material
  nameWeight: 700,       // Archivo weight for both the material and the HTML (400/600/700/800)
  nameTracking: -0.028,  // optional letter-spacing in em (default -0.028)
  phases: [              // the prototype's own narrative, contiguous from 0 to nameAt
    { label: 'Strata', start: 0, end: 2.4 }, ...
  ],
  reveal: { colors: { name, yijin, sub, accent, rule, glow },     // optional CSS colors
            gradient: { name: [top, bottom], yijin: [top, bottom] } }, // optional lit gradient type
  posterTime: 13,        // optional: frame shown under prefers-reduced-motion (must fall in timing.read)
  create(ctx) { ...; return { update(t, dt), resize(ctx), dispose(), stats } },
};
```

`ctx` = `{ container, width, height, quality: 'high'|'low', reducedMotion, layout, timing }`.

- `layout` (core/text.js `computeLayout`) holds the exact glyph positions of the
  name in container pixels. Use `sampleName(layout, count, { which: 'short' })`
  for particle targets, `rasterizeName` / `distanceField` / `traceContours` for
  line or voxel work, `layout.words.*` for per-word positions.
- `timing` (core/reveal.js `revealTiming`) is the shared reveal schedule:
  `htmlIn`, `hold`, `split`, `yijin`, `subtitle`, `read`, `out`, `duration`.
  The HTML name crossfades in over `htmlIn`; Daryl/Tan slide apart over `split`.

### Rules

1. **Pure function of `t`.** `update(t)` must render the same frame for the same
   `t`, whatever came before. No accumulation buffers, no runtime `Math.random`
   (use `rng(seed)` from core), precompute in `create`/`resize`. This is what
   makes pause, scrub, speed, replay and the gauntlet work.
2. **Form the name on the shared glyphs** by `timing.nameAt`, using the `'short'`
   ("Daryl Tan") layout.
3. **Clean handoff.** The HTML name fades in over `timing.htmlIn`. By
   `timing.split[0]` the prototype's own "Daryl Tan" must be faded out, or be
   moving with the words. A ghost of "Daryl Tan" left behind would overlap "Yi Jin".
4. **Quiet read phase.** During `timing.read`, keep the scene alive, but keep it dim
   and low-contrast behind the text block.
5. **Seamless loop.** The frame at `t → duration` must match `t = 0`. The
   `out` phase returns the scene to its opening state. The gauntlet reports this
   as `seam` and it should stay below ~3.
6. **Performance.** One renderer per instance, no per-frame allocation, large
   counts animated in shaders. `quality: 'low'` (used by the six-up grid) cuts
   counts roughly 3× and skips post-processing. `dispose()` frees everything.
7. **Resize** rebuilds anything derived from `layout`.

## Visual testing (the gauntlet)

```bash
node hero-lab/tools/gauntlet.cjs A            # desktop 1440x900
node hero-lab/tools/gauntlet.cjs A --mobile   # 390x844
node hero-lab/tools/gauntlet.cjs A --times 0,3.5,9.2 --no-perf
```

The gauntlet writes one PNG per sampled time plus `<id>-contact.png` (a labelled
contact sheet of the whole cycle) to `$TMPDIR/hero-lab-gauntlet` (or `--out`).
It also prints the loop seam, the settled name's alignment, console errors and a
relative frame-cost figure (headless WebGL is software-rendered).
