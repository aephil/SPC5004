// Live demos for lec15.qmd. Each <div class="l15-demo" data-demo="NAME"> in the deck holds the static
// layout (so the slide text and maths are in the .qmd); init[NAME] wires up its canvases and controls.
// Data (photo, CIFAR-10 cats, spiral) come from demos-data.js, made by pics/lec15_demos.py.
// Testing aid: window.__L15[NAME] exposes each demo's state and setters.
(function () {
  'use strict';
  const D = window.L15DATA;
  const ORANGE = '#d9480f', GREY = '#7a7a7a', INK = '#222', FONT = '"IBM Plex Sans", sans-serif';

  // The DDPM schedule, as on the slides: betas linspace(1e-4, 0.02, 1000); AB[t] for t = 0..999.
  const T = 1000, AB = new Float64Array(T);
  for (let t = 0, p = 1; t < T; t++) { p *= 1 - (1e-4 + (0.02 - 1e-4) * t / (T - 1)); AB[t] = p; }

  // Seeded random numbers, so each demo looks the same every time the deck loads.
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
.l15-demo { font-size: 0.6em; line-height: 1.3; }
.l15-demo canvas { display: block; }
.l15-demo .l15-row { display: flex; gap: 28px; align-items: flex-start; justify-content: center; }
.l15-demo .l15-col { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.l15-demo .l15-lab { font-weight: 600; }
.l15-demo .l15-pix { image-rendering: pixelated; border: 1px solid #555; }
.l15-demo .l15-ctl { display: flex; gap: 14px; align-items: center; justify-content: center; margin-top: 14px; flex-wrap: wrap; }
.l15-demo input[type=range] { width: 420px; accent-color: ${ORANGE}; }
.l15-demo button { font: inherit; font-size: 0.95em; padding: 3px 12px; border: 1px solid #888; border-radius: 4px;
  background: #fff; color: #000; cursor: pointer; }
.l15-demo button:hover { background: #f3f3f3; }
.l15-demo button.on { background: ${ORANGE}; border-color: ${ORANGE}; color: #fff; }
.l15-demo .l15-out { min-width: 7.5em; font-variant-numeric: tabular-nums; }
.l15-demo .l15-top { display: grid; grid-template-columns: auto 120px 3.2em; gap: 6px 10px; align-items: center; }
.l15-demo .l15-bar { height: 14px; background: ${ORANGE}; }
.l15-demo .l15-pct { text-align: right; font-variant-numeric: tabular-nums; }
.reveal .slides section .l15-num { font-variant-numeric: tabular-nums; }
`;
  document.head.appendChild(Object.assign(document.createElement('style'), {textContent: css}));

  const q = (root, role) => root.querySelector(`[data-role="${role}"]`);
  // Write to every element with this class on the demo's slide (readouts in the slide text).
  const setAll = (root, cls, s) => (root.closest('section') || root).querySelectorAll('.' + cls)
    .forEach(e => { e.textContent = s; });
  const fmt = (x, d = 2) => x.toFixed(d);
  const fmtT = t => t.toLocaleString('en-GB');

  // Mouse use shouldn't leave focus on a control, or the presenter's next keypress moves the slider
  // instead of the slide. Keyboard users who tab to a slider can still use the arrow keys on it.
  function control(e) {
    e.addEventListener('pointerup', () => setTimeout(() => e.blur(), 0));
    if (e.tagName === 'BUTTON') e.addEventListener('click', () => e.blur());
    e.addEventListener('keydown', ev => { if (e.type === 'range' && ev.key.startsWith('Arrow')) ev.stopPropagation(); });
    return e;
  }

  // A canvas sized in CSS pixels, drawn at device resolution.
  function hiDPI(c, w, h) {
    const r = window.devicePixelRatio > 1 ? 2 : 1;
    c.width = w * r; c.height = h * r; c.style.width = w + 'px'; c.style.height = h + 'px';
    const g = c.getContext('2d'); g.setTransform(r, 0, 0, r, 0, 0);
    return g;
  }

  // Pixel position on a canvas in its own CSS units, whatever reveal's scaling.
  function localX(c, ev, w) { const b = c.getBoundingClientRect(); return (ev.clientX - b.left) * w / b.width; }

  async function pixels(src) {
    const img = new Image(); img.src = src; await img.decode();
    const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
    const g = c.getContext('2d'); g.drawImage(img, 0, 0);
    return g.getImageData(0, 0, img.width, img.height);
  }

  // Draw a [-1, 1] float RGB image (length 3 w h) onto a w x h canvas.
  function show(c, x, w, h) {
    const g = c.getContext('2d'), im = g.createImageData(w, h);
    for (let i = 0, j = 0; i < w * h; i++, j += 3) {
      im.data[4 * i] = (x[j] + 1) * 127.5; im.data[4 * i + 1] = (x[j + 1] + 1) * 127.5;
      im.data[4 * i + 2] = (x[j + 2] + 1) * 127.5; im.data[4 * i + 3] = 255;
    }
    g.putImageData(im, 0, 0);
  }
  function toFloat(im, x0, y0, w, h, out, off) {
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const s = 4 * ((y0 + y) * im.width + x0 + x), d = off + 3 * (y * w + x);
      out[d] = im.data[s] / 127.5 - 1; out[d + 1] = im.data[s + 1] / 127.5 - 1; out[d + 2] = im.data[s + 2] / 127.5 - 1;
    }
  }

  // Exact best guess of x0 for a finite data set: a softmax-weighted average of the data, each item
  // weighted by how likely it is to have produced xt. Returns the weights (data: n items of length m).
  function posterior(xt, data, n, m, ab, w) {
    const s = Math.sqrt(ab), v2 = 2 * (1 - ab);
    let mx = -Infinity;
    for (let j = 0; j < n; j++) {
      let d = 0;
      for (let k = 0, o = j * m; k < m; k++) { const e = xt[k] - s * data[o + k]; d += e * e; }
      w[j] = -d / v2; if (w[j] > mx) mx = w[j];
    }
    let z = 0;
    for (let j = 0; j < n; j++) { w[j] = Math.exp(w[j] - mx); z += w[j]; }
    for (let j = 0; j < n; j++) w[j] /= z;
    return w;
  }
  function weighted(data, n, m, w, out) {
    out.fill(0);
    for (let j = 0; j < n; j++) {
      if (w[j] < 1e-7) continue;
      for (let k = 0, o = j * m; k < m; k++) out[k] += w[j] * data[o + k];
    }
    return out;
  }

  const init = {};
  window.__L15 = {};

  // ---- A. Noising a photo, with the signal and noise curves --------------------------------------
  init.noise = async function (root) {
    const RES = 192, M = 3 * RES * RES;
    const x0 = new Float32Array(M);
    toFloat(await pixels(D.chelsea), 0, 0, RES, RES, x0, 0);
    const img = q(root, 'img'), chart = q(root, 'chart'), slider = control(q(root, 't'));
    img.width = img.height = RES;
    const CW = 600, CH = 300, L = 60, R = 20, TOP = 16, B = 254;
    const g = hiDPI(chart, CW, CH);
    const X = t => L + (CW - L - R) * t / (T - 1), Y = v => B - (B - TOP) * v;
    let seed = 1, eps = gaussians(M, seed), t = +slider.value, anim = 0;
    const xt = new Float32Array(M);

    function drawChart() {
      g.clearRect(0, 0, CW, CH);
      g.font = `500 15px ${FONT}`; g.fillStyle = '#444'; g.strokeStyle = '#ddd'; g.lineWidth = 1;
      g.textAlign = 'right'; g.textBaseline = 'middle';
      for (const v of [0, 0.5, 1]) {
        g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(CW - R, Y(v)); g.stroke(); g.fillText(v, L - 10, Y(v));
      }
      g.textAlign = 'center'; g.textBaseline = 'top';
      for (const v of [0, 250, 500, 750, 999]) g.fillText(v, X(v), B + 8);
      g.font = `600 16px ${FONT}`; g.fillStyle = '#000'; g.fillText('timestep t', (L + CW - R) / 2, B + 28);
      g.strokeStyle = INK; g.lineWidth = 1.6; g.beginPath(); g.moveTo(L, TOP); g.lineTo(L, B); g.lineTo(CW - R, B); g.stroke();
      for (const [f, col] of [[k => Math.sqrt(AB[k]), ORANGE], [k => Math.sqrt(1 - AB[k]), GREY]]) {
        g.strokeStyle = col; g.lineWidth = 3; g.beginPath();
        for (let k = 0; k < T; k += 3) g[k ? 'lineTo' : 'moveTo'](X(k), Y(f(k)));
        g.stroke();
      }
      g.font = `700 17px ${FONT}`; g.textAlign = 'left'; g.textBaseline = 'alphabetic';
      g.fillStyle = ORANGE; g.fillText('how much image is left', X(560), Y(0.3));
      g.fillStyle = GREY; g.fillText('how much noise is in', X(560), Y(0.86));
      // marker at t
      g.strokeStyle = '#000'; g.lineWidth = 1.5; g.setLineDash([5, 4]);
      g.beginPath(); g.moveTo(X(t), TOP); g.lineTo(X(t), B); g.stroke(); g.setLineDash([]);
      for (const [v, col] of [[Math.sqrt(AB[t]), ORANGE], [Math.sqrt(1 - AB[t]), GREY]]) {
        g.fillStyle = col; g.strokeStyle = '#fff'; g.lineWidth = 2;
        g.beginPath(); g.arc(X(t), Y(v), 7, 0, 2 * Math.PI); g.fill(); g.stroke();
      }
    }
    function update() {
      const s = Math.sqrt(AB[t]), n = Math.sqrt(1 - AB[t]);
      for (let k = 0; k < M; k++) xt[k] = Math.max(-1, Math.min(1, s * x0[k] + n * eps[k]));
      show(img, xt, RES, RES);
      drawChart();
      slider.value = t; q(root, 'tval').textContent = t;
      setAll(root, 'l15-sig', fmt(s)); setAll(root, 'l15-noi', fmt(n));
    }
    const set = v => { t = Math.max(0, Math.min(T - 1, Math.round(v))); update(); };
    const stop = () => { cancelAnimationFrame(anim); anim = 0; q(root, 'play').classList.remove('on'); };
    slider.addEventListener('input', () => { stop(); set(+slider.value); });
    let drag = false;
    chart.addEventListener('pointerdown', ev => { drag = true; stop(); chart.setPointerCapture(ev.pointerId); set((localX(chart, ev, CW) - L) / (CW - L - R) * (T - 1)); });
    chart.addEventListener('pointermove', ev => { if (drag) set((localX(chart, ev, CW) - L) / (CW - L - R) * (T - 1)); });
    chart.addEventListener('pointerup', () => { drag = false; });
    chart.style.cursor = 'ew-resize'; chart.style.touchAction = 'none';
    control(q(root, 'new')).addEventListener('click', () => { eps = gaussians(M, ++seed); update(); });
    control(q(root, 'play')).addEventListener('click', ev => {
      if (anim) return stop();
      ev.currentTarget.classList.add('on');
      if (t >= T - 1) t = 0;
      let last = performance.now();
      const tick = now => {
        set(t + (now - last) * T / 5000); last = now;   // 0 to 999 in 5 s
        if (t >= T - 1) stop(); else anim = requestAnimationFrame(tick);
      };
      anim = requestAnimationFrame(tick);
    });
    update();
    document.fonts.ready.then(drawChart);
    return {stop, api: {set, get t() { return t; }}};
  };

  // ---- C. The best guess is an average ------------------------------------------------------------
  init.average = async function (root) {
    const {n: N, cols} = D.cats, M = 3 * 32 * 32, TOPN = 5;
    const sprite = await pixels(D.cats.src), data = new Float32Array(N * M);
    for (let j = 0; j < N; j++) toFloat(sprite, 32 * (j % cols), 32 * Math.floor(j / cols), 32, 32, data, j * M);
    const slider = control(q(root, 't')), cx0 = q(root, 'x0'), cxt = q(root, 'xt'), chat = q(root, 'xhat');
    for (const c of [cx0, cxt, chat]) { c.width = c.height = 32; }
    // the five most likely sources, with weight bars
    const top = q(root, 'top'), rows = [];
    for (let r = 0; r < TOPN; r++) {
      const c = document.createElement('canvas'); c.width = c.height = 32; c.className = 'l15-pix';
      c.style.width = c.style.height = '44px';
      const bw = document.createElement('div'), bar = document.createElement('div'), pct = document.createElement('div');
      bar.className = 'l15-bar'; bw.appendChild(bar); pct.className = 'l15-pct';
      top.append(c, bw, pct); rows.push({c, bar, pct});
    }
    let idx = 7, seed = 11, eps = gaussians(M, seed), t = +slider.value;
    const xt = new Float32Array(M), xhat = new Float32Array(M), w = new Float64Array(N);
    function update() {
      const ab = AB[t], s = Math.sqrt(ab), n = Math.sqrt(1 - ab), x0 = data.subarray(idx * M, (idx + 1) * M);
      for (let k = 0; k < M; k++) xt[k] = s * x0[k] + n * eps[k];
      posterior(xt, data, N, M, ab, w);
      weighted(data, N, M, w, xhat);
      show(cx0, x0, 32, 32);
      show(cxt, xt.map(v => Math.max(-1, Math.min(1, v))), 32, 32);
      show(chat, xhat, 32, 32);
      const order = Array.from(w.keys()).sort((a, b) => w[b] - w[a]);
      rows.forEach((r, k) => {
        const j = order[k];
        show(r.c, data.subarray(j * M, (j + 1) * M), 32, 32);
        r.c.style.outline = j === idx ? `3px solid ${ORANGE}` : 'none';
        r.bar.style.width = (100 * w[j]).toFixed(1) + '%';
        r.pct.textContent = w[j] >= 0.995 ? '100%' : w[j] >= 0.01 ? Math.round(100 * w[j]) + '%' : '<1%';
      });
      let s2 = 0; for (let j = 0; j < N; j++) s2 += w[j] * w[j];
      slider.value = t; q(root, 'tval').textContent = t;
      setAll(root, 'l15-neff', fmtT(Math.round(1 / s2)));
      api.neff = 1 / s2; api.wmax = w[order[0]];
    }
    const api = {set: v => { t = v; update(); }, get t() { return t; }, data, xhat, N, M};
    slider.addEventListener('input', () => { t = +slider.value; update(); });
    control(q(root, 'new')).addEventListener('click', () => { idx = (idx * 7919 + 13) % N; eps = gaussians(M, ++seed); update(); });
    update();
    return {stop() {}, api};
  };

  // ---- D. Small steps, in two dimensions ----------------------------------------------------------
  init.spiral = async function (root) {
    const P = Float32Array.from(D.spiral), N = P.length / 2, S = 460, LIM = 3.2;
    const sec = root.closest('section') || root;           // the controls are in the other column
    const c = q(root, 'plot'), g = hiDPI(c, S, S), slider = control(q(sec, 't'));
    const px = v => S / 2 + v * S / (2 * LIM), py = v => S / 2 - v * S / (2 * LIM);
    const eps = gaussians(2 * N, 5), w = new Float64Array(N), hat = new Float64Array(2);
    let pts = new Float32Array(2 * N), anim = 0, runs = 0;

    function draw(points, label) {
      g.fillStyle = '#fff'; g.fillRect(0, 0, S, S);
      g.strokeStyle = '#ccc'; g.lineWidth = 1; g.strokeRect(0.5, 0.5, S - 1, S - 1);
      g.fillStyle = '#d4d4d4';
      for (let i = 0; i < N; i++) { g.beginPath(); g.arc(px(P[2 * i]), py(P[2 * i + 1]), 2.2, 0, 2 * Math.PI); g.fill(); }
      g.fillStyle = ORANGE; g.globalAlpha = 0.8;
      for (let i = 0; i < N; i++) { g.beginPath(); g.arc(px(points[2 * i]), py(points[2 * i + 1]), 3, 0, 2 * Math.PI); g.fill(); }
      g.globalAlpha = 1;
      q(root, 'status').textContent = label;
    }
    function forward(t) {
      const s = Math.sqrt(AB[t]), n = Math.sqrt(1 - AB[t]);
      for (let k = 0; k < 2 * N; k++) pts[k] = s * P[k] + n * eps[k];
      draw(pts, t === 0 ? 'the data: 500 points' : `the data, noised to step t = ${fmtT(t)}`);
    }
    // Timesteps for a K-step sampler, spaced evenly and ending at 999 (diffusers' "trailing" spacing).
    const steps = K => Array.from({length: K}, (_, k) => Math.round(T - k * T / K) - 1);
    // One DDPM step from t to tp (tp = -1 means clean), with the exact best guess for this point set.
    function step(x, t, tp, z) {
      const ab = AB[t], abp = tp >= 0 ? AB[tp] : 1, a = ab / abp, b = 1 - a;
      const c0 = Math.sqrt(abp) * b / (1 - ab), ct = Math.sqrt(a) * (1 - abp) / (1 - ab), sd = Math.sqrt(b * (1 - abp) / (1 - ab));
      for (let i = 0; i < N; i++) {
        const xi = x.subarray(2 * i, 2 * i + 2);
        posterior(xi, P, N, 2, ab, w);
        weighted(P, N, 2, w, hat);
        xi[0] = c0 * hat[0] + ct * xi[0] + sd * z[2 * i];
        xi[1] = c0 * hat[1] + ct * xi[1] + sd * z[2 * i + 1];
      }
    }
    // Run the whole sampler at once (for tests and print); returns the final points.
    function sample(K, seed) {
      const x = gaussians(2 * N, seed), ts = steps(K);
      ts.forEach((t, k) => step(x, t, k + 1 < K ? ts[k + 1] : -1, gaussians(2 * N, seed * 1000 + k + 1)));
      return x;
    }
    const buttons = [...sec.querySelectorAll('[data-steps]')].map(control);
    const stop = () => { cancelAnimationFrame(anim); anim = 0; buttons.forEach(b => b.classList.remove('on')); };
    function generate(K, btn) {
      stop(); btn.classList.add('on');
      const seed = 100 + ++runs, ts = steps(K);
      pts = gaussians(2 * N, seed);
      draw(pts, `generating in ${fmtT(K)} step${K > 1 ? 's' : ''}: pure noise`);
      const per = Math.max(1, Math.ceil(K / 120));          // steps per frame: about 2 s for 1,000 steps
      const hold = Math.min(500, 2500 / K);                 // ms per frame when there are few steps
      let k = 0, last = performance.now() + 400;            // show the starting noise first
      const tick = now => {
        if (now - last >= hold) {
          last = now;
          for (let j = 0; j < per && k < K; j++, k++) step(pts, ts[k], k + 1 < K ? ts[k + 1] : -1, gaussians(2 * N, seed * 1000 + k + 1));
          draw(pts, k < K ? `generating in ${fmtT(K)} steps: now at t = ${fmtT(ts[k])}`
            : `generated in ${fmtT(K)} step${K > 1 ? 's' : ''}`);
        }
        if (k < K) anim = requestAnimationFrame(tick); else { anim = 0; btn.classList.remove('on'); }
      };
      anim = requestAnimationFrame(tick);
    }
    buttons.forEach(b => b.addEventListener('click', () => generate(+b.dataset.steps, b)));
    slider.addEventListener('input', () => { stop(); forward(+slider.value); });
    forward(+slider.value);
    return {stop, api: {forward, sample, steps, draw, P, N}};
  };

  // Start every demo on the page; stop any animation when the slide changes.
  const running = [];
  async function start() {
    for (const root of document.querySelectorAll('.l15-demo[data-demo]')) {
      const name = root.dataset.demo;
      try {
        const d = await init[name](root);
        running.push(d); window.__L15[name] = d.api;
      } catch (e) { console.error('lec15 demo', name, e); }
    }
    window.__L15.ready = true;
    if (window.Reveal && Reveal.on) Reveal.on('slidechanged', () => running.forEach(d => d.stop()));
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
