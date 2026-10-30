// reveal.js - the shared name reveal, in real HTML typography.
//
// A prototype forms "Daryl Tan" out of its own material by `timing.nameAt`,
// exactly on the glyphs computed in text.js. This overlay then crossfades the
// crisp HTML name in over it, holds, slides "Daryl" left and "Tan" right,
// opens "Yi Jin" between them, reveals the subtitle, holds for reading and
// fades out before the loop point. Choreography is a paused GSAP timeline that
// the lab clock seeks every frame, so scrubbing, speed and replay are exact.

import { gsap } from 'gsap';
import { SUBTITLE_ITEMS, FULL_NAME, FONT_FAMILY, MONO_FAMILY } from './text.js';

/** Shared reveal schedule, derived from when the name is formed and the cycle length. */
export function revealTiming(nameAt, duration) {
  return {
    nameAt,
    htmlIn: [nameAt - 0.15, nameAt + 0.65],   // HTML name crossfades over the formed material
    hold: [nameAt + 0.65, nameAt + 1.25],
    split: [nameAt + 1.25, nameAt + 2.3],     // Daryl <- -> Tan
    yijin: [nameAt + 1.75, nameAt + 2.75],    // Yi Jin opens between them
    subtitle: [nameAt + 2.55, nameAt + 3.45],
    read: [nameAt + 3.45, duration - 2.0],
    out: [duration - 2.0, duration - 1.05],   // name dissolves; the last ~1 s belongs to the scene returning to t=0
    duration,
  };
}

/** Phases appended to every prototype's own narrative phases for the timeline UI. */
export function revealPhases(t) {
  return [
    { label: 'Daryl Tan', start: t.nameAt, end: t.split[0] },
    { label: 'Yi Jin', start: t.split[0], end: t.subtitle[0] },
    { label: 'Subtitle', start: t.subtitle[0], end: t.read[0] },
    { label: 'Read', start: t.read[0], end: t.out[0] },
    { label: 'Return', start: t.out[0], end: t.duration },
  ];
}

const STYLE_ID = 'hl-reveal-style';
function injectStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const s = document.createElement('style');
  s.id = STYLE_ID;
  s.textContent = `
.hl-reveal { position: absolute; inset: 0; pointer-events: none; z-index: 5; color: var(--hl-name, #F4F4F2); contain: layout paint; }
.hl-reveal .hl-word { position: absolute; left: 0; top: 0; line-height: 1; white-space: pre; font-family: ${FONT_FAMILY};
  font-kerning: normal; font-feature-settings: "kern" 1; -webkit-font-smoothing: antialiased; will-change: transform, opacity; }
.hl-reveal .hl-word i.bl { display: inline-block; width: 0; height: 0; vertical-align: baseline; }
.hl-reveal .hl-yijin { color: var(--hl-yijin, rgba(244,244,242,0.62)); }
.hl-reveal.hl-gradient .hl-word { color: transparent; -webkit-background-clip: text; background-clip: text;
  background-image: linear-gradient(180deg, var(--hl-g0) 8%, var(--hl-g1) 92%); }
.hl-reveal.hl-gradient .hl-yijin { background-image: linear-gradient(180deg, var(--hl-y0) 8%, var(--hl-y1) 92%); }
.hl-reveal .hl-sub { position: absolute; left: 50%; top: 0; transform: translateX(-50%); width: min(92%, 980px);
  text-align: center; font-family: ${MONO_FAMILY}; text-transform: uppercase; color: var(--hl-sub, rgba(244,244,242,0.72));
  line-height: 1.9; text-wrap: balance; }
.hl-reveal .hl-sub .it { white-space: nowrap; }
.hl-reveal .hl-sub .dot { color: var(--hl-accent, #C8364C); padding: 0 0.7em; }
.hl-reveal .hl-rule { position: absolute; left: 50%; height: 1px; background: var(--hl-rule, rgba(244,244,242,0.28)); transform-origin: 50% 50%; }
.hl-reveal .hl-sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
`;
  document.head.appendChild(s);
}

/**
 * createReveal(container, opts)
 *   opts.colors: { name, yijin, sub, accent, rule, glow }   (CSS colors)
 * Returns { setLayout(layout, timing), update(t), el, dispose() }
 */
export function createReveal(container, opts = {}) {
  injectStyle();
  const c = Object.assign({
    name: '#F4F4F2', yijin: 'rgba(244,244,242,0.62)', sub: 'rgba(244,244,242,0.72)',
    accent: '#C8364C', rule: 'rgba(244,244,242,0.28)', glow: 'rgba(120,200,255,0.55)',
  }, opts.colors || {});

  const el = document.createElement('div');
  el.className = 'hl-reveal';
  // opts.gradient: { name: [top, bottom], yijin: [top, bottom] } renders the name as light-lit
  // gradient type (background-clip: text). Glow is dropped there: a text-shadow would show
  // through the transparent fill.
  const grad = opts.gradient;
  if (grad) {
    el.classList.add('hl-gradient');
    el.style.setProperty('--hl-g0', grad.name[0]); el.style.setProperty('--hl-g1', grad.name[1]);
    const yj = grad.yijin || grad.name;
    el.style.setProperty('--hl-y0', yj[0]); el.style.setProperty('--hl-y1', yj[1]);
  }
  el.style.setProperty('--hl-name', c.name);
  el.style.setProperty('--hl-yijin', c.yijin);
  el.style.setProperty('--hl-sub', c.sub);
  el.style.setProperty('--hl-accent', c.accent);
  el.style.setProperty('--hl-rule', c.rule);
  el.innerHTML = `
    <h1 class="hl-sr">${FULL_NAME}</h1>
    <div class="hl-word hl-daryl" aria-hidden="true">Daryl<i class="bl"></i></div>
    <div class="hl-word hl-yijin" aria-hidden="true">Yi Jin<i class="bl"></i></div>
    <div class="hl-word hl-tan" aria-hidden="true">Tan<i class="bl"></i></div>
    <div class="hl-rule" aria-hidden="true"></div>
    <p class="hl-sub"></p>`;
  container.appendChild(el);

  const daryl = el.querySelector('.hl-daryl');
  const yijin = el.querySelector('.hl-yijin');
  const tan = el.querySelector('.hl-tan');
  const rule = el.querySelector('.hl-rule');
  const sub = el.querySelector('.hl-sub');
  const words = [daryl, yijin, tan];

  let tl = null;
  let lastT = 0;

  function setLayout(L, T) {
    for (const w of words) {
      w.style.font = `${L.weight} ${L.fontSize}px ${FONT_FAMILY}`;
      w.style.lineHeight = '1';
      w.style.letterSpacing = `${L.tracking}px`;
    }
    // where does the baseline sit inside each line box? (probe, once per layout)
    gsap.set(words, { clearProps: 'transform' });
    const top = daryl.getBoundingClientRect().top;
    const blOffset = daryl.querySelector('i.bl').getBoundingClientRect().top - top;
    const y = L.baseline - blOffset;

    // Narrow viewports get a fixed, balanced two-line subtitle so no separator dot is orphaned at a line edge.
    const dot = '<span class="dot" aria-hidden="true">·</span>';
    const items = SUBTITLE_ITEMS.map((x) => `<span class="it">${x}</span>`);
    sub.innerHTML = L.subtitle.narrow
      ? `${items.slice(0, 2).join(dot)}<br>${items.slice(2).join(dot)}`
      : items.join(dot);
    const subFs = L.subtitle.fontSize;
    sub.style.fontSize = `${subFs}px`;
    sub.style.letterSpacing = L.subtitle.narrow ? '0.1em' : '0.16em';
    const ruleW = Math.min(L.fullBox.w * 0.42, 260);
    rule.style.width = `${ruleW}px`;
    rule.style.marginLeft = `${-ruleW / 2}px`;
    const ruleTop = L.subtitle.top;
    rule.style.top = `${ruleTop}px`;
    const subTop = ruleTop + Math.max(14, subFs * 1.3);

    if (tl) tl.kill();
    const W = L.words;
    const glowOn = grad ? 'none' : `0 0 ${Math.round(L.fontSize * 0.35)}px ${c.glow}`;
    const glowOff = grad ? 'none' : `0 0 ${Math.round(L.fontSize * 0.35)}px rgba(0,0,0,0)`;
    const f3 = { force3D: false };

    const hidden = { opacity: 0, filter: 'blur(10px)', textShadow: glowOn };
    const shown = { opacity: 1, filter: 'blur(0px)', textShadow: glowOn };
    gsap.set(daryl, { x: W.daryl.x0, y, ...hidden, ...f3 });
    gsap.set(tan, { x: W.tan.x0, y, ...hidden, ...f3 });
    gsap.set(yijin, { x: W.yijin.x1, y: y + L.fontSize * 0.06, opacity: 0, clipPath: 'inset(-25% 50% -35% 50%)', ...f3 });
    gsap.set(rule, { scaleX: 0, opacity: 0 });
    gsap.set(sub, { y: subTop + 10, opacity: 0, ...f3 });

    // Every tween is an explicit fromTo so any seek order renders the same frame.
    const ft = (target, from, to, at) => tl.fromTo(target, from, { ...to, immediateRender: false }, at);
    tl = gsap.timeline({ paused: true, defaults: { ease: 'power2.out', ...f3 } });

    const [i0, i1] = T.htmlIn;
    ft([daryl, tan], hidden, { ...shown, duration: i1 - i0 }, i0);
    ft([daryl, tan], { textShadow: glowOn }, { textShadow: glowOff, duration: 1.4, ease: 'power1.out' }, i1 - 0.2);

    const [s0, s1] = T.split;
    ft(daryl, { x: W.daryl.x0 }, { x: W.daryl.x1, duration: s1 - s0, ease: 'power3.inOut' }, s0);
    ft(tan, { x: W.tan.x0 }, { x: W.tan.x1, duration: s1 - s0, ease: 'power3.inOut' }, s0);

    const [y0, y1] = T.yijin;
    // the clip opens from the centre; opacity trails it so the first sliver never reads as a speck in the gap
    ft(yijin, { y: y + L.fontSize * 0.06, clipPath: 'inset(-25% 50% -35% 50%)' },
      { y, clipPath: 'inset(-25% 0% -35% 0%)', duration: y1 - y0, ease: 'power3.out' }, y0);
    ft(yijin, { opacity: 0 }, { opacity: 1, duration: (y1 - y0) * 0.75, ease: 'power2.inOut' }, y0 + (y1 - y0) * 0.2);

    const [b0, b1] = T.subtitle;
    ft(rule, { scaleX: 0, opacity: 0 }, { scaleX: 1, opacity: 1, duration: (b1 - b0) * 1.1, ease: 'power3.inOut' }, b0 - 0.15);
    ft(sub, { y: subTop + 10, opacity: 0 }, { y: subTop, opacity: 1, duration: b1 - b0 }, b0);

    const [o0, o1] = T.out;
    ft([sub, rule], { opacity: 1 }, { opacity: 0, duration: (o1 - o0) * 0.6, ease: 'power1.in' }, o0);
    ft([daryl, yijin, tan], { opacity: 1, filter: 'blur(0px)', y }, { opacity: 0, filter: 'blur(8px)', y: y - L.fontSize * 0.05, duration: o1 - o0 - 0.22, ease: 'power2.in', stagger: 0.06 }, o0 + 0.1);   // last word ends exactly at o1
    // pad to full cycle so seeking near the end stays settled
    tl.set({}, {}, T.duration);
    tl.time(lastT, true);
  }

  function update(t) {
    lastT = t;
    if (tl) tl.time(t, true);
  }

  function dispose() {
    if (tl) tl.kill();
    el.remove();
  }

  return { el, setLayout, update, dispose };
}
