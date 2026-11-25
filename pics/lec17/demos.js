// Live demos for lec17.qmd. Each <div class="l17-demo" data-demo="NAME"> in the deck holds the static
// layout (so the slide text and maths are in the .qmd); init[NAME] wires up its canvases and controls.
// Data (VAE round trip, one generation step by step, cross-attention maps and a guidance sweep) come
// from demos-data.js, made by pics/lec17_figs.py from Stable Diffusion v1.5.
// Testing aid: window.__L17[NAME] exposes each demo's state and setters.
(function () {
  'use strict';
  const D = window.L17DATA;
  const ORANGE = '#d9480f', INK = '#222', FONT = '"IBM Plex Sans", sans-serif';
  const CLASS = ['#d9480f', '#2f7d4a', '#a23aa6'];

  // The DDPM schedule of lectures 15 and 16, for the toy: betas linspace(1e-4, 0.02, 1000).
  const T = 1000, DDPM = new Float64Array(T);
  for (let t = 0, p = 1; t < T; t++) { p *= 1 - (1e-4 + (0.02 - 1e-4) * t / (T - 1)); DDPM[t] = p; }

  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussians(n, seed) {
    const r = rng(seed), out = new Float32Array(n);
    for (let i = 0; i < n; i += 2) {
      const u = 1 - r(), v = r(), m = Math.sqrt(-2 * Math.log(u));
      out[i] = m * Math.cos(2 * Math.PI * v);
      if (i + 1 < n) out[i + 1] = m * Math.sin(2 * Math.PI * v);
    }
    return out;
  }

  const css = `
.l17-demo { font-size: 0.6em; line-height: 1.3; }
.l17-demo canvas { display: block; }
.l17-demo .l17-row { display: flex; gap: 24px; align-items: flex-start; justify-content: center; }
.l17-demo .l17-col { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.l17-demo .l17-lab { font-weight: 600; }
.l17-demo .l17-pix { border: 1px solid #555; }
.l17-demo .l17-blocky { image-rendering: pixelated; }
.l17-demo .l17-ctl { display: flex; gap: 10px; align-items: center; justify-content: center; margin-top: 12px; flex-wrap: wrap; }
.l17-demo input[type=range] { width: 380px; accent-color: ${ORANGE}; }
.l17-demo button { font: inherit; font-size: 0.95em; padding: 3px 11px; border: 1px solid #888; border-radius: 4px;
  background: #fff; color: #000; cursor: pointer; }
.l17-demo button:hover { background: #f3f3f3; }
.l17-demo button.on { background: ${ORANGE}; border-color: ${ORANGE}; color: #fff; }
.l17-demo .l17-out { min-width: 6.5em; font-variant-numeric: tabular-nums; }
.l17-demo .l17-words { display: flex; flex-wrap: wrap; gap: 6px; max-width: 420px; }
.l17-demo .l17-words button { padding: 2px 8px; }
.reveal .slides section .l17-num { font-variant-numeric: tabular-nums; }
`;
  document.head.appendChild(Object.assign(document.createElement('style'), {textContent: css}));

  const q = (root, role) => root.querySelector(`[data-role="${role}"]`);
  const section = root => root.closest('section') || root;
  const setAll = (root, cls, s) => section(root).querySelectorAll('.' + cls).forEach(e => { e.textContent = s; });
  const fmtT = t => t.toLocaleString('en-GB');

  // Mouse use shouldn't leave focus on a control, or the presenter's next keypress moves the slider
  // instead of the slide. Keyboard users who tab to a slider can still use the arrow keys on it.
  function control(e) {
    e.addEventListener('pointerup', () => setTimeout(() => e.blur(), 0));
    if (e.tagName === 'BUTTON') e.addEventListener('click', () => e.blur());
    e.addEventListener('keydown', ev => { if (e.type === 'range' && ev.key.startsWith('Arrow')) ev.stopPropagation(); });
    return e;
  }
  function hiDPI(c, w, h) {
    const r = window.devicePixelRatio > 1 ? 2 : 1;
    c.width = w * r; c.height = h * r; c.style.width = w + 'px'; c.style.height = h + 'px';
    const g = c.getContext('2d'); g.setTransform(r, 0, 0, r, 0, 0);
    return g;
  }
  // Pointer position on a canvas in its own CSS units, whatever reveal's scaling.
  function local(c, ev, w, h) {
    const b = c.getBoundingClientRect();
    return [(ev.clientX - b.left) * w / b.width, (ev.clientY - b.top) * h / b.height];
  }
  async function image(src) { const img = new Image(); img.src = src; await img.decode(); return img; }
  function posterior(xt, data, n, ab, w) {
    const s = Math.sqrt(ab), v2 = 2 * (1 - ab);
    let mx = -Infinity;
    for (let j = 0; j < n; j++) {
      const a = xt[0] - s * data[2 * j], b = xt[1] - s * data[2 * j + 1];
      w[j] = -(a * a + b * b) / v2; if (w[j] > mx) mx = w[j];
    }
    let z = 0;
    for (let j = 0; j < n; j++) { w[j] = Math.exp(w[j] - mx); z += w[j]; }
    let hx = 0, hy = 0;
    for (let j = 0; j < n; j++) { hx += w[j] * data[2 * j]; hy += w[j] * data[2 * j + 1]; }
    return [hx / z, hy / z];
  }

  const init = {};
  window.__L17 = {};

  // ---- 1. VAE round trip, with a magnifier -------------------------------------------------------
  init.vae = async function (root) {
    const [a, b] = await Promise.all([image(D.vae.in), image(D.vae.out)]);
    const S = 300, Z = 210, src = a.width;                   // display sizes; source is 512 x 512
    const gf = hiDPI(q(root, 'full'), S, S), gi = hiDPI(q(root, 'zin'), Z, Z), go = hiDPI(q(root, 'zout'), Z, Z);
    let cx = 0.41, cy = 0.22, win = 72;                       // lens centre (fraction) and size (source px)
    function draw() {
      const half = win / 2;
      const x0 = Math.min(Math.max(cx * src - half, 0), src - win), y0 = Math.min(Math.max(cy * src - half, 0), src - win);
      gf.drawImage(a, 0, 0, S, S);
      gf.strokeStyle = ORANGE; gf.lineWidth = 2.5; gf.strokeRect(x0 * S / src, y0 * S / src, win * S / src, win * S / src);
      for (const [g, im] of [[gi, a], [go, b]]) { g.imageSmoothingEnabled = false; g.drawImage(im, x0, y0, win, win, 0, 0, Z, Z); }
    }
    const c = q(root, 'full');
    let drag = false;
    const at = ev => { const [x, y] = local(c, ev, S, S); cx = x / S; cy = y / S; draw(); };
    c.addEventListener('pointerdown', ev => { drag = true; c.setPointerCapture(ev.pointerId); at(ev); });
    c.addEventListener('pointermove', ev => { if (drag) at(ev); });
    c.addEventListener('pointerup', () => { drag = false; });
    c.style.cursor = 'crosshair'; c.style.touchAction = 'none';
    root.querySelectorAll('[data-win]').forEach(btn => control(btn).addEventListener('click', () => {
      win = +btn.dataset.win; root.querySelectorAll('[data-win]').forEach(x => x.classList.toggle('on', x === btn)); draw();
    }));
    draw();
    return {stop() {}, api: {set(o) { if ('cx' in o) cx = o.cx; if ('cy' in o) cy = o.cy; if ('win' in o) win = o.win; draw(); }}};
  };

  // ---- 2. One generation, step by step -----------------------------------------------------------
  init.traj = async function (root) {
    const F = D.traj, n = F.length, S = 230;
    const imgs = await Promise.all(F.map(f => Promise.all([image(f.z), image(f.xt), image(f.x0)])));
    const gs = ['z', 'xt', 'x0'].map(r => { const c = q(root, r); const g = hiDPI(c, S, S); return g; });
    const slider = control(q(root, 'step')), play = control(q(root, 'play'));
    slider.max = n - 1;
    let k = +slider.value, timer = 0;
    function update() {
      const f = F[k];
      imgs[k].forEach((im, j) => { gs[j].imageSmoothingEnabled = j > 0; gs[j].drawImage(im, 0, 0, S, S); });
      q(root, 'tval').textContent = f.i >= D.traj_steps ? 'finished' : `step ${f.i + 1} of ${D.traj_steps}, t = ${fmtT(f.t)}`;
    }
    const stop = () => { clearInterval(timer); timer = 0; play.textContent = 'Play'; };
    slider.addEventListener('input', () => { stop(); k = +slider.value; update(); });
    play.addEventListener('click', () => {
      if (timer) return stop();
      if (k >= n - 1) k = 0;
      play.textContent = 'Pause';
      timer = setInterval(() => { k++; slider.value = k; update(); if (k >= n - 1) stop(); }, 450);
      update();
    });
    update();
    return {stop, api: {set(i) { k = i; slider.value = i; update(); }, n}};
  };

  // ---- 3. Cross-attention: where each word looks --------------------------------------------------
  init.xattn = async function (root) {
    const X = D.xattn, img = await image(X.image), S = 380;
    const g = hiDPI(q(root, 'img'), S, S);
    const small = document.createElement('canvas'); small.width = small.height = 16;
    const sg = small.getContext('2d'), id = sg.createImageData(16, 16);
    const box = q(root, 'words');
    let cur = -1;
    const words = X.words.map((w, j) => {
      const label = w === '<|startoftext|>' ? 'start' : w === '<|endoftext|>' ? 'end' : w;
      const b = control(Object.assign(document.createElement('button'), {textContent: label}));
      if (j === 0 || j === X.words.length - 1) b.style.fontStyle = 'italic';
      b.addEventListener('click', () => select(j === cur ? -1 : j));
      box.appendChild(b);
      return b;
    });
    function select(j) {
      cur = j;
      words.forEach((b, i) => b.classList.toggle('on', i === j));
      g.globalAlpha = 1; g.drawImage(img, 0, 0, S, S);
      if (j < 0) { q(root, 'share').textContent = ''; return; }
      const m = X.maps[j], mx = Math.max(...m), mn = Math.min(...m);
      for (let i = 0; i < 256; i++) {
        const v = (m[i] - mn) / (mx - mn);                // each word stretched from its own minimum to maximum
        id.data[4 * i] = 217; id.data[4 * i + 1] = 72; id.data[4 * i + 2] = 15; id.data[4 * i + 3] = Math.round(255 * v);
      }
      sg.putImageData(id, 0, 0);
      g.fillStyle = 'rgba(0,0,0,0.45)'; g.fillRect(0, 0, S, S);   // dim the picture, then light up the map
      g.imageSmoothingEnabled = true; g.globalAlpha = 0.85; g.drawImage(small, 0, 0, S, S); g.globalAlpha = 1;
      const share = m.reduce((s, v) => s + v, 0) / 256;
      q(root, 'share').textContent = `${(100 * share).toFixed(share < 0.01 ? 1 : 0)}% of all attention, on average`;
    }
    const first = X.words.findIndex(w => w === 'lighthouse</w>' || w === 'lighthouse');
    select(first >= 0 ? first : 1);
    return {stop() {}, api: {select, words: X.words}};
  };

  // ---- 4. Guidance on three rings (after Ho and Salimans' figure 2, which uses three Gaussians) ----
  init.toy = async function (root) {
    const R = 1.0, D0 = 0.75, NR = 300, M = 200, K = 100, S = 420, LIM = 2.1;
    // Three overlapping rings in a triangle, each one class. The unconditional data is all three.
    const r = rng(3), sets = [0, 1, 2].map(c => {
      const a = new Float32Array(2 * NR), cx = D0 * Math.cos(Math.PI / 2 + c * 2 * Math.PI / 3),
        cy = D0 * Math.sin(Math.PI / 2 + c * 2 * Math.PI / 3), e = gaussians(NR, 30 + c);
      for (let j = 0; j < NR; j++) {
        const th = 2 * Math.PI * r(), rr = R + 0.04 * e[j];
        a[2 * j] = cx + rr * Math.cos(th); a[2 * j + 1] = cy + rr * Math.sin(th);
      }
      return a;
    });
    const P = new Float32Array(6 * NR); sets.forEach((a, c) => P.set(a, 2 * NR * c));
    const N = 3 * NR, wa = new Float64Array(N), wc = new Float64Array(NR);
    const steps = Array.from({length: K}, (_, k) => Math.round(T - k * T / K) - 1);
    const px = v => S / 2 + v * S / (2 * LIM), py = v => S / 2 - v * S / (2 * LIM);
    // DDPM sampling from the exact best guesses, with x0-hat guided: u + s (c - u). Since the noise
    // prediction is a fixed linear function of x0-hat at each step, this is the same as guiding it.
    function sample(cls, s, seed = 7) {
      const x = gaussians(2 * M, seed), data = sets[cls];
      steps.forEach((t, k) => {
        const tp = k + 1 < K ? steps[k + 1] : -1, ab = DDPM[t], abp = tp >= 0 ? DDPM[tp] : 1, a = ab / abp, b = 1 - a;
        const c0 = Math.sqrt(abp) * b / (1 - ab), ct = Math.sqrt(a) * (1 - abp) / (1 - ab), sd = Math.sqrt(b * (1 - abp) / (1 - ab));
        const z = gaussians(2 * M, seed * 1000 + k + 1);
        for (let i = 0; i < M; i++) {
          const xi = x.subarray(2 * i, 2 * i + 2);
          const u = posterior(xi, P, N, ab, wa), cc = posterior(xi, data, NR, ab, wc);
          const hx = u[0] + s * (cc[0] - u[0]), hy = u[1] + s * (cc[1] - u[1]);
          xi[0] = c0 * hx + ct * xi[0] + sd * z[2 * i];
          xi[1] = c0 * hy + ct * xi[1] + sd * z[2 * i + 1];
        }
      });
      return x;
    }
    const sec = section(root), slider = control(q(sec, 's')), g = hiDPI(q(root, 'plot'), S, S);
    // Samples for each scale, all three classes. Each class is 200 points x 100 steps (about 0.2 s),
    // so work through the slider's scales in the background, one class at a time, and cache them:
    // once a scale is done, dragging the slider over it redraws at once.
    const cache = new Map(), idle = () => new Promise(res => setTimeout(res, 0));
    let s = +slider.value, busy = Promise.resolve();
    function compute(v) {
      if (cache.has(v)) return Promise.resolve(cache.get(v));
      busy = busy.then(async () => {
        if (cache.has(v)) return;
        const out = [];
        for (let c = 0; c < 3; c++) { await idle(); out.push(sample(c, v)); }
        cache.set(v, out);
      });
      return busy.then(() => cache.get(v));
    }
    function draw(v) {
      g.fillStyle = '#fff'; g.fillRect(0, 0, S, S);
      g.strokeStyle = '#ccc'; g.lineWidth = 1; g.strokeRect(0.5, 0.5, S - 1, S - 1);
      g.fillStyle = '#ddd';
      for (let j = 0; j < N; j++) { g.beginPath(); g.arc(px(P[2 * j]), py(P[2 * j + 1]), 2, 0, 2 * Math.PI); g.fill(); }
      const pts = cache.get(v);
      if (!pts) return;
      g.globalAlpha = 0.85;
      pts.forEach((x, c) => {
        g.fillStyle = CLASS[c];
        for (let i = 0; i < M; i++) { g.beginPath(); g.arc(px(x[2 * i]), py(x[2 * i + 1]), 3, 0, 2 * Math.PI); g.fill(); }
      });
      g.globalAlpha = 1;
    }
    async function show(v) {
      q(sec, 'sval').textContent = `s = ${v.toFixed(1)}`;
      draw(v);
      if (!cache.has(v)) { await compute(v); if (v === s) draw(v); }
    }
    slider.addEventListener('input', () => { s = +slider.value; show(s); });
    show(s).then(() => {
      for (let v = +slider.min; v <= +slider.max; v += +slider.step) compute(v);
    });
    return {stop() {}, api: {sample, sets, set(o) { if ('s' in o) { s = o.s; slider.value = s; } return show(s); }}};
  };

  // ---- 5. Guidance on Stable Diffusion --------------------------------------------------------------
  init.cfg = async function (root) {
    const C = D.cfg, S = 340;
    const imgs = {};
    await Promise.all(Object.entries(C.img).map(async ([k, v]) => { imgs[k] = await image(v); }));
    const g = hiDPI(q(root, 'img'), S, S), slider = control(q(root, 's'));
    slider.max = C.scales.length - 1;
    let k = +slider.value, seed = C.seeds[0];
    function update() {
      g.drawImage(imgs[`${seed}-${C.scales[k]}`], 0, 0, S, S);
      q(root, 'sval').textContent = `s = ${C.scales[k]}`;
    }
    slider.addEventListener('input', () => { k = +slider.value; update(); });
    const seeds = [...root.querySelectorAll('[data-seed]')].map(control);
    seeds.forEach(b => { if (+b.dataset.seed >= C.seeds.length) b.parentElement.style.display = 'none'; });
    seeds.forEach(b => b.addEventListener('click', () => {
      seed = C.seeds[+b.dataset.seed]; seeds.forEach(x => x.classList.toggle('on', x === b)); update();
    }));
    update();
    return {stop() {}, api: {set(o) { if ('k' in o) { k = o.k; slider.value = k; } if ('seed' in o) seed = C.seeds[o.seed]; update(); }}};
  };

  const running = [];
  async function start() {
    for (const root of document.querySelectorAll('.l17-demo[data-demo]')) {
      const name = root.dataset.demo;
      try {
        const d = await init[name](root);
        running.push(d); window.__L17[name] = d.api;
      } catch (e) { console.error('lec17 demo', name, e); }
    }
    window.__L17.ready = true;
    if (window.Reveal && Reveal.on) Reveal.on('slidechanged', () => running.forEach(d => d.stop()));
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
