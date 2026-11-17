// Live demos for lec16.qmd. Each <div class="l16-demo" data-demo="NAME"> in the deck holds the static
// layout (so the slide text and maths are in the .qmd); init[NAME] wires up its canvases and controls.
// Data come from demos-data.js, made by pics/lec16_demos.py. The helpers are copied from
// pics/lec15/demos.js. Testing aid: window.__L16[NAME] exposes each demo's state and setters.
(function () {
  'use strict';
  const D = window.L16DATA;
  const ORANGE = '#d9480f', GREY = '#7a7a7a', INK = '#222', FONT = '"IBM Plex Sans", sans-serif';
  const CLASS = ['#d9480f', '#2f7d4a', '#a23aa6'];          // ring, line, blobs

  // Schedules, t = 0..999: the lab's linear one (betas 0.001 to 0.02) and the cosine one
  // (diffusers' squaredcos_cap_v2, betas capped at 0.999). pics/lec16_demos.py checks both.
  const T = 1000;
  const LIN = new Float64Array(T), COS = new Float64Array(T), DDPM = new Float64Array(T);
  {
    const f = s => Math.cos((s / T + 0.008) / 1.008 * Math.PI / 2) ** 2;
    let pl = 1, pc = 1, pd = 1;
    for (let t = 0; t < T; t++) {
      pl *= 1 - (1e-3 + (0.02 - 1e-3) * t / (T - 1)); LIN[t] = pl;
      pc *= 1 - Math.min(1 - f(t + 1) / f(t), 0.999); COS[t] = pc;
      pd *= 1 - (1e-4 + (0.02 - 1e-4) * t / (T - 1)); DDPM[t] = pd;
    }
  }

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
.l16-demo { font-size: 0.6em; line-height: 1.3; }
.l16-demo canvas { display: block; }
.l16-demo .l16-row { display: flex; gap: 24px; align-items: flex-start; justify-content: center; }
.l16-demo .l16-col { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.l16-demo .l16-lab { font-weight: 600; }
.l16-demo .l16-pix { image-rendering: pixelated; border: 1px solid #555; }
.l16-demo .l16-ctl { display: flex; gap: 10px; align-items: center; justify-content: center; margin-top: 12px; flex-wrap: wrap; }
.l16-demo input[type=range] { width: 380px; accent-color: ${ORANGE}; }
.l16-demo button { font: inherit; font-size: 0.95em; padding: 3px 11px; border: 1px solid #888; border-radius: 4px;
  background: #fff; color: #000; cursor: pointer; }
.l16-demo button:hover { background: #f3f3f3; }
.l16-demo button.on { background: ${ORANGE}; border-color: ${ORANGE}; color: #fff; }
.l16-demo .l16-out { min-width: 6.5em; font-variant-numeric: tabular-nums; }
.l16-demo .l16-strip { display: flex; gap: 6px; justify-content: center; margin-top: 10px; }
.l16-demo .l16-strip canvas { width: 54px; height: 54px; image-rendering: pixelated; border: 1px solid #555; cursor: pointer; }
.l16-demo .l16-strip canvas.on { outline: 3px solid ${ORANGE}; outline-offset: 1px; }
.reveal .slides section .l16-num { font-variant-numeric: tabular-nums; }
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
  function localX(c, ev, w) { const b = c.getBoundingClientRect(); return (ev.clientX - b.left) * w / b.width; }
  async function pixels(src) {
    const img = new Image(); img.src = src; await img.decode();
    const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
    const g = c.getContext('2d'); g.drawImage(img, 0, 0);
    return g.getImageData(0, 0, img.width, img.height);
  }
  async function image(src) { const img = new Image(); img.src = src; await img.decode(); return img; }
  function show(c, x, w, h) {
    const g = c.getContext('2d'), im = g.createImageData(w, h);
    for (let i = 0, j = 0; i < w * h; i++, j += 3) {
      im.data[4 * i] = (x[j] + 1) * 127.5; im.data[4 * i + 1] = (x[j + 1] + 1) * 127.5;
      im.data[4 * i + 2] = (x[j + 2] + 1) * 127.5; im.data[4 * i + 3] = 255;
    }
    g.putImageData(im, 0, 0);
  }
  function toFloat(im, out) {
    for (let i = 0; i < im.width * im.height; i++)
      for (let c = 0; c < 3; c++) out[3 * i + c] = im.data[4 * i + c] / 127.5 - 1;
    return out;
  }
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
  // A t-axis line chart: axes, ticks and grid; returns coordinate maps.
  function axes(g, CW, CH, L, R, TOP, B, ylab) {
    const X = t => L + (CW - L - R) * t / (T - 1), Y = v => B - (B - TOP) * v;
    g.clearRect(0, 0, CW, CH);
    g.font = `500 15px ${FONT}`; g.fillStyle = '#444'; g.strokeStyle = '#ddd'; g.lineWidth = 1;
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const v of ylab) { g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(CW - R, Y(v)); g.stroke(); g.fillText(v, L - 10, Y(v)); }
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (const v of [0, 250, 500, 750, 999]) g.fillText(v, X(v), B + 8);
    g.font = `600 16px ${FONT}`; g.fillStyle = '#000'; g.fillText('timestep t', (L + CW - R) / 2, B + 28);
    g.strokeStyle = INK; g.lineWidth = 1.6; g.beginPath(); g.moveTo(L, TOP); g.lineTo(L, B); g.lineTo(CW - R, B); g.stroke();
    return {X, Y};
  }
  function curve(g, X, Y, f, col, width, alpha = 1) {
    g.globalAlpha = alpha; g.strokeStyle = col; g.lineWidth = width; g.beginPath();
    for (let k = 0; k < T; k += 2) g[k ? 'lineTo' : 'moveTo'](X(k), Y(f(k)));
    g.lineTo(X(T - 1), Y(f(T - 1))); g.stroke(); g.globalAlpha = 1;
  }
  function marker(g, X, t, TOP, B) {
    g.strokeStyle = '#000'; g.lineWidth = 1.5; g.setLineDash([5, 4]);
    g.beginPath(); g.moveTo(X(t), TOP); g.lineTo(X(t), B); g.stroke(); g.setLineDash([]);
  }
  function dot(g, x, y, col) {
    g.fillStyle = col; g.strokeStyle = '#fff'; g.lineWidth = 2;
    g.beginPath(); g.arc(x, y, 7, 0, 2 * Math.PI); g.fill(); g.stroke();
  }
  // Dragging on a t-chart sets t.
  function dragT(c, CW, L, R, set, stop) {
    let drag = false;
    const at = ev => set((localX(c, ev, CW) - L) / (CW - L - R) * (T - 1));
    c.addEventListener('pointerdown', ev => { drag = true; stop && stop(); c.setPointerCapture(ev.pointerId); at(ev); });
    c.addEventListener('pointermove', ev => { if (drag) at(ev); });
    c.addEventListener('pointerup', () => { drag = false; });
    c.style.cursor = 'ew-resize'; c.style.touchAction = 'none';
  }

  const init = {};
  window.__L16 = {};

  // ---- 1. Schedule x resolution ----------------------------------------------------------------
  init.sizes = async function (root) {
    const big = toFloat(await pixels(D.photo256), new Float32Array(3 * 256 * 256));
    const small = toFloat(await pixels(D.photo32), new Float32Array(3 * 32 * 32));
    const eb = gaussians(big.length, 1), es = gaussians(small.length, 2);
    const cb = q(root, 'big'), cs = q(root, 'small'), chart = q(root, 'chart'), slider = control(q(root, 't'));
    cb.width = cb.height = 256; cs.width = cs.height = 32;
    const bufB = new Float32Array(big.length), bufS = new Float32Array(small.length);
    const CW = 470, CH = 290, L = 56, R = 16, TOP = 14, B = 240;
    const g = hiDPI(chart, CW, CH);
    const btns = [...root.querySelectorAll('[data-sched]')].map(control);
    let t = +slider.value, sched = 'lin';
    const AB = () => sched === 'lin' ? LIN : COS;
    function drawChart() {
      const {X, Y} = axes(g, CW, CH, L, R, TOP, B, [0, 0.5, 1]);
      const on = AB();
      curve(g, X, Y, k => Math.sqrt(LIN[k]), ORANGE, sched === 'lin' ? 3.2 : 1.6, sched === 'lin' ? 1 : 0.35);
      curve(g, X, Y, k => Math.sqrt(COS[k]), GREY, sched === 'cos' ? 3.2 : 1.6, sched === 'cos' ? 1 : 0.35);
      g.font = `700 16px ${FONT}`; g.textAlign = 'left'; g.textBaseline = 'alphabetic';
      g.globalAlpha = sched === 'lin' ? 1 : 0.5; g.fillStyle = ORANGE; g.fillText('linear (lab)', X(255), Y(0.22));
      g.globalAlpha = sched === 'cos' ? 1 : 0.5; g.fillStyle = GREY; g.fillText('cosine', X(560), Y(0.8)); g.globalAlpha = 1;
      marker(g, X, t, TOP, B);
      dot(g, X(t), Y(Math.sqrt(on[t])), sched === 'lin' ? ORANGE : GREY);
    }
    function update() {
      const ab = AB()[t], s = Math.sqrt(ab), n = Math.sqrt(1 - ab);
      for (let k = 0; k < big.length; k++) bufB[k] = Math.max(-1, Math.min(1, s * big[k] + n * eb[k]));
      for (let k = 0; k < small.length; k++) bufS[k] = Math.max(-1, Math.min(1, s * small[k] + n * es[k]));
      show(cb, bufB, 256, 256); show(cs, bufS, 32, 32);
      drawChart();
      slider.value = t; q(root, 'tval').textContent = t;
      btns.forEach(b => b.classList.toggle('on', b.dataset.sched === sched));
      setAll(root, 'l16-sig', s.toFixed(2));
    }
    const set = v => { t = Math.max(0, Math.min(T - 1, Math.round(v))); update(); };
    slider.addEventListener('input', () => set(+slider.value));
    dragT(chart, CW, L, R, set);
    btns.forEach(b => b.addEventListener('click', () => { sched = b.dataset.sched; update(); }));
    update();
    document.fonts.ready.then(drawChart);
    return {stop() {}, api: {set, setSched(s) { sched = s; update(); }, get t() { return t; },
      noiseRMS() { // how far each noisy image is from its clean version, per pixel
        const r = (a, b) => { let e = 0; for (let k = 0; k < a.length; k++) e += (a[k] - b[k]) ** 2; return Math.sqrt(e / a.length); };
        return [r(bufB, big), r(bufS, small)];
      }}};
  };

  // ---- 2. Timestep embedding -------------------------------------------------------------------
  init.embed = async function (root) {
    const H = 16, OM = Float64Array.from({length: H}, (_, i) => Math.exp(-Math.log(10000) * i / H));
    const emb = t => { const e = new Float64Array(2 * H); for (let i = 0; i < H; i++) { e[i] = Math.cos(t * OM[i]); e[H + i] = Math.sin(t * OM[i]); } return e; };
    const sim = (t, s) => { let v = 0; for (let i = 0; i < H; i++) v += Math.cos((t - s) * OM[i]); return v / H; };
    const bar = q(root, 'bar'), chart = q(root, 'chart'), slider = control(q(root, 't'));
    const BW = 760, BH = 128, CELL = 38, X0 = 100;
    const gb = hiDPI(bar, BW, BH);
    // Every step's 32 numbers, one column per step (t across, the 32 numbers down), drawn once.
    const CW = 760, CH = 230, L = 100, R = 20, TOP = 10, B = 170;
    const g = hiDPI(chart, CW, CH);
    const map = document.createElement('canvas'); map.width = T; map.height = 2 * H;
    let t = +slider.value;
    // diverging scale: -1 grey, 0 white, +1 orange
    const mix = (a, b, f) => a.map((v, i) => Math.round(v + (b[i] - v) * f));
    const colour = v => { const c = v >= 0 ? mix([255, 255, 255], [217, 72, 15], v) : mix([255, 255, 255], [90, 90, 90], -v); return `rgb(${c})`; };
    function drawBar() {
      const e = emb(t);
      gb.clearRect(0, 0, BW, BH);
      gb.font = `600 16px ${FONT}`; gb.fillStyle = '#000'; gb.textAlign = 'right'; gb.textBaseline = 'middle';
      gb.fillText('cosines', X0 - 12, 20 + CELL / 2); gb.fillText('sines', X0 - 12, 20 + CELL * 1.5 + 6);
      for (let r = 0; r < 2; r++) for (let i = 0; i < H; i++) {
        gb.fillStyle = colour(e[r * H + i]); gb.fillRect(X0 + i * CELL, 20 + r * (CELL + 6), CELL - 3, CELL);
        gb.strokeStyle = '#bbb'; gb.lineWidth = 1; gb.strokeRect(X0 + i * CELL + 0.5, 20 + r * (CELL + 6) + 0.5, CELL - 4, CELL - 1);
      }
      gb.font = `500 15px ${FONT}`; gb.fillStyle = '#444'; gb.textBaseline = 'top';
      gb.textAlign = 'left'; gb.fillText('← fast: changes every step', X0, 20 + 2 * CELL + 14);
      gb.textAlign = 'right'; gb.fillText('slow: changes over hundreds of steps →', X0 + H * CELL - 3, 20 + 2 * CELL + 14);
    }
    {
      const mg = map.getContext('2d'), im = mg.createImageData(T, 2 * H);
      for (let s = 0; s < T; s++) {
        const e = emb(s);
        for (let r = 0; r < 2 * H; r++) {
          const v = e[r], o = 4 * (r * T + s);
          const c = v >= 0 ? mix([255, 255, 255], [217, 72, 15], v) : mix([255, 255, 255], [90, 90, 90], -v);
          im.data[o] = c[0]; im.data[o + 1] = c[1]; im.data[o + 2] = c[2]; im.data[o + 3] = 255;
        }
      }
      mg.putImageData(im, 0, 0);
    }
    function drawChart() {
      const X = s => L + (CW - L - R) * s / (T - 1);
      g.clearRect(0, 0, CW, CH);
      g.imageSmoothingEnabled = true;
      g.drawImage(map, L, TOP, CW - L - R, B - TOP);
      g.strokeStyle = '#bbb'; g.lineWidth = 1; g.strokeRect(L + 0.5, TOP + 0.5, CW - L - R - 1, B - TOP - 1);
      g.font = `600 16px ${FONT}`; g.fillStyle = '#000'; g.textAlign = 'right'; g.textBaseline = 'middle';
      g.fillText('cosines', L - 12, TOP + (B - TOP) / 4); g.fillText('sines', L - 12, TOP + 3 * (B - TOP) / 4);
      g.font = `500 15px ${FONT}`; g.fillStyle = '#444'; g.textAlign = 'center'; g.textBaseline = 'top';
      for (const v of [0, 250, 500, 750, 999]) g.fillText(v, X(v), B + 8);
      g.font = `600 16px ${FONT}`; g.fillStyle = '#000'; g.fillText('timestep t', (L + CW - R) / 2, B + 28);
      g.strokeStyle = '#000'; g.lineWidth = 3; g.strokeRect(X(t) - 3, TOP - 4, 6, B - TOP + 8);
    }
    function update() {
      drawBar(); drawChart();
      slider.value = t; q(root, 'tval').textContent = t;
    }
    const set = v => { t = Math.max(0, Math.min(T - 1, Math.round(v))); update(); };
    slider.addEventListener('input', () => set(+slider.value));
    dragT(chart, CW, L, R, set);
    update();
    document.fonts.ready.then(update);
    return {stop() {}, api: {set, emb, sim, get t() { return t; }}};
  };

  // ---- 3. Conditional toy in 2D ----------------------------------------------------------------
  init.classes = async function (root) {
    const sec = section(root);
    const P = Float32Array.from(D.classes.xy), LAB = D.classes.label, N = LAB.length, S = 440, LIM = 2.9, K = 100;
    const sets = [0, 1, 2].map(c => { const idx = LAB.flatMap((l, i) => l === c ? [i] : []);
      const a = new Float32Array(2 * idx.length); idx.forEach((i, j) => { a[2 * j] = P[2 * i]; a[2 * j + 1] = P[2 * i + 1]; }); return a; });
    const c = q(root, 'plot'), g = hiDPI(c, S, S), slider = control(q(sec, 't'));
    const px = v => S / 2 + v * S / (2 * LIM), py = v => S / 2 - v * S / (2 * LIM);
    const eps = gaussians(2 * N, 5), M = 300;
    let pts = new Float32Array(2 * M), cols = [], anim = 0, runs = 0;
    function draw(points, colours, label, faint = true) {
      g.fillStyle = '#fff'; g.fillRect(0, 0, S, S);
      g.strokeStyle = '#ccc'; g.lineWidth = 1; g.strokeRect(0.5, 0.5, S - 1, S - 1);
      if (faint) {
        g.globalAlpha = 0.18;
        for (let i = 0; i < N; i++) { g.fillStyle = CLASS[LAB[i]]; g.beginPath(); g.arc(px(P[2 * i]), py(P[2 * i + 1]), 2.2, 0, 2 * Math.PI); g.fill(); }
      }
      g.globalAlpha = 0.85;
      for (let i = 0; i < points.length / 2; i++) { g.fillStyle = colours[i]; g.beginPath(); g.arc(px(points[2 * i]), py(points[2 * i + 1]), 3, 0, 2 * Math.PI); g.fill(); }
      g.globalAlpha = 1;
      q(root, 'status').textContent = label;
    }
    function forward(t) {
      const s = Math.sqrt(DDPM[t]), n = Math.sqrt(1 - DDPM[t]), x = new Float32Array(2 * N);
      for (let k = 0; k < 2 * N; k++) x[k] = s * P[k] + n * eps[k];
      draw(x, LAB.map(l => CLASS[l]), t === 0 ? 'the data, coloured by label' : `the data, noised to step t = ${fmtT(t)}`, false);
    }
    const steps = Array.from({length: K}, (_, k) => Math.round(T - k * T / K) - 1);
    const w = new Float64Array(N), hat = new Float64Array(2);
    // One DDPM step from t to tp, with the exact best guess for the chosen data (one class, or all).
    function step(x, data, t, tp, z) {
      const n = data.length / 2, ab = DDPM[t], abp = tp >= 0 ? DDPM[tp] : 1, a = ab / abp, b = 1 - a;
      const c0 = Math.sqrt(abp) * b / (1 - ab), ct = Math.sqrt(a) * (1 - abp) / (1 - ab), sd = Math.sqrt(b * (1 - abp) / (1 - ab));
      for (let i = 0; i < x.length / 2; i++) {
        const xi = x.subarray(2 * i, 2 * i + 2);
        posterior(xi, data, n, 2, ab, w); weighted(data, n, 2, w, hat);
        xi[0] = c0 * hat[0] + ct * xi[0] + sd * z[2 * i];
        xi[1] = c0 * hat[1] + ct * xi[1] + sd * z[2 * i + 1];
      }
    }
    const nearest = (x, y) => { let best = 0, m = Infinity; for (let i = 0; i < N; i++) { const d = (x - P[2 * i]) ** 2 + (y - P[2 * i + 1]) ** 2; if (d < m) { m = d; best = LAB[i]; } } return best; };
    function sample(which, seed) {
      const x = gaussians(2 * M, seed), data = which === 'any' ? P : sets[+which];
      steps.forEach((t, k) => step(x, data, t, k + 1 < K ? steps[k + 1] : -1, gaussians(2 * M, seed * 1000 + k + 1)));
      return x;
    }
    const buttons = [...sec.querySelectorAll('[data-gen]')].map(control);
    const name = which => which === 'any' ? 'any shape' : ['the ring', 'the line', 'the blobs'][+which];
    const stop = () => { cancelAnimationFrame(anim); anim = 0; buttons.forEach(b => b.classList.remove('on')); };
    function generate(which, btn) {
      stop(); btn.classList.add('on');
      const seed = 100 + ++runs, data = which === 'any' ? P : sets[+which];
      pts = gaussians(2 * M, seed);
      const colour = () => Array.from({length: M}, (_, i) => which === 'any' ? INK : CLASS[+which]);
      cols = colour();
      draw(pts, cols, `generating ${name(which)}: pure noise`);
      let k = 0, last = performance.now() + 400;
      const tick = now => {
        if (now - last >= 20) {
          last = now;
          for (let j = 0; j < 2 && k < K; j++, k++) step(pts, data, steps[k], k + 1 < K ? steps[k + 1] : -1, gaussians(2 * M, seed * 1000 + k + 1));
          if (k >= K && which === 'any') cols = Array.from({length: M}, (_, i) => CLASS[nearest(pts[2 * i], pts[2 * i + 1])]);
          draw(pts, cols, k < K ? `generating ${name(which)}: now at t = ${fmtT(steps[k])}` : `generated: ${name(which)}`);
        }
        if (k < K) anim = requestAnimationFrame(tick); else { anim = 0; btn.classList.remove('on'); }
      };
      anim = requestAnimationFrame(tick);
    }
    buttons.forEach(b => b.addEventListener('click', () => generate(b.dataset.gen, b)));
    slider.addEventListener('input', () => { stop(); forward(+slider.value); });
    forward(+slider.value);
    return {stop, api: {sample, forward, nearest, sets, P, N}};
  };

  // ---- 4. Label x noise explorer (real model) -------------------------------------------------
  init.explorer = async function (root) {
    const E = D.explorer;
    if (!E) { q(root, 'status').textContent = '(model data not built)'; return {stop() {}, api: {}}; }
    const [xt, hat] = await Promise.all([image(E.xt), image(E.hat)]);
    const F = E.frames.length + 1, C = E.conds;            // last frame: the finished image
    const cx = q(root, 'xt'), ch = q(root, 'hat'), slider = control(q(root, 'step')), blend = control(q(root, 'blend'));
    cx.width = cx.height = ch.width = ch.height = 32;
    const digits = [...root.querySelectorAll('[data-digit]')].map(control);
    const seeds = [...root.querySelectorAll('[data-seed]')].map(control);
    const strip = q(root, 'strip'), thumbs = [];
    for (let d = 0; d < 10; d++) {
      const c = document.createElement('canvas'); c.width = c.height = 32; c.title = 'ask for ' + d;
      c.addEventListener('click', () => { cond = d; blend.value = d === 1 ? 0 : d === 7 ? 4 : blend.value; update(); });
      strip.appendChild(c); thumbs.push(c);
    }
    const BLEND = [1, 10, 11, 12, 7];                         // blend slider positions -> condition
    let cond = 7, seed = 0, f = +slider.value, anim = 0;
    const blit = (c, img, row, col) => { const g = c.getContext('2d'); g.imageSmoothingEnabled = false; g.drawImage(img, col * 32, row * 32, 32, 32, 0, 0, 32, 32); };
    function update() {
      const row = seed * C + cond;
      blit(cx, xt, row, f); blit(ch, hat, row, f);
      thumbs.forEach((c, d) => { blit(c, xt, seed * C + d, F - 1); c.classList.toggle('on', d === cond); });
      digits.forEach(b => b.classList.toggle('on', +b.dataset.digit === cond));
      seeds.forEach(b => b.classList.toggle('on', +b.dataset.seed === seed));
      slider.value = f;
      q(root, 'tval').textContent = f < F - 1 ? `t = ${E.frames[f]}` : 'finished';
      const bi = BLEND.indexOf(cond);
      q(root, 'bval').textContent = bi < 0 ? '' : `${bi * 25}% of the way to 7`;
      blend.style.opacity = bi < 0 ? 0.35 : 1;
      if (bi >= 0) blend.value = bi;
    }
    const stop = () => { clearInterval(anim); anim = 0; q(root, 'play').classList.remove('on'); };
    slider.addEventListener('input', () => { stop(); f = +slider.value; update(); });
    blend.addEventListener('input', () => { cond = BLEND[+blend.value]; update(); });
    digits.forEach(b => b.addEventListener('click', () => { cond = +b.dataset.digit; if (cond === 1) blend.value = 0; if (cond === 7) blend.value = 4; update(); }));
    seeds.forEach(b => b.addEventListener('click', () => { seed = +b.dataset.seed; update(); }));
    control(q(root, 'play')).addEventListener('click', ev => {
      if (anim) return stop();
      ev.currentTarget.classList.add('on');
      if (f >= F - 1) f = 0;
      anim = setInterval(() => { f++; update(); if (f >= F - 1) stop(); }, 160);
    });
    update();
    return {stop, api: {set(o) { if ('cond' in o) cond = o.cond; if ('seed' in o) seed = o.seed; if ('f' in o) f = o.f; update(); }, F, C}};
  };

  const running = [];
  async function start() {
    for (const root of document.querySelectorAll('.l16-demo[data-demo]')) {
      const name = root.dataset.demo;
      try {
        const d = await init[name](root);
        running.push(d); window.__L16[name] = d.api;
      } catch (e) { console.error('lec16 demo', name, e); }
    }
    window.__L16.ready = true;
    if (window.Reveal && Reveal.on) Reveal.on('slidechanged', () => running.forEach(d => d.stop()));
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
