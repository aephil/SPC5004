// Live demos for lec18.qmd. Each <div class="l18-demo" data-demo="NAME"> in the deck holds the static layout
// (so the slide text and maths are in the .qmd); init[NAME] wires up its canvases and controls.
// Data (the fine-tuning run's snapshots, losses, memory probe and dataset) come from demos-data.js, made by
// pics/lec18_figs.py from the GPU jobs in pics/lec18_gpu/. The EMA toy needs no data.
// Testing aid: window.__L18[NAME] exposes each demo's state and setters.
(function () {
  'use strict';
  const D = window.L18DATA;
  const ORANGE = '#d9480f', BLUE = '#2c6fb7', GREEN = '#2f7d4a', INK = '#222', GREY = '#888';
  const FONT = '"IBM Plex Sans", sans-serif';
  const GiB = 1.073741824;                       // probe.json is in GiB; the slides use GB

  const css = `
.l18-demo { font-size: 0.6em; line-height: 1.3; position: relative; }
.l18-demo canvas { display: block; }
.l18-demo .l18-row { display: flex; gap: 24px; align-items: flex-start; justify-content: center; }
.l18-demo .l18-col { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.l18-demo .l18-lab { font-weight: 600; }
.l18-demo .l18-pix { border: 1px solid #555; }
.l18-demo .l18-ctl { display: flex; gap: 10px; align-items: center; justify-content: center; margin-top: 10px; flex-wrap: wrap; }
.l18-demo input[type=range] { width: 380px; accent-color: ${ORANGE}; }
.l18-demo button { font: inherit; font-size: 0.95em; padding: 3px 11px; border: 1px solid #888; border-radius: 4px;
  background: #fff; color: #000; cursor: pointer; }
.l18-demo button:hover { background: #f3f3f3; }
.l18-demo button.on { background: ${ORANGE}; border-color: ${ORANGE}; color: #fff; }
.l18-demo .l18-out { min-width: 6.5em; font-variant-numeric: tabular-nums; }
.l18-demo .l18-prompt { font-style: italic; max-width: 760px; text-align: center; min-height: 2.6em; }
.l18-demo .l18-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px 16px; }
.l18-demo .l18-grid figure { margin: 0; display: flex; flex-direction: column; align-items: center; gap: 4px; }
.l18-demo .l18-grid img { width: 190px; height: 150px; object-fit: contain; background: #f4f1ea; border: 1px solid #ccc; margin: 0; }
.l18-demo .l18-grid figcaption { font-size: 0.85em; font-style: italic; text-align: center; max-width: 200px; }
.l18-demo .l18-ph { position: absolute; top: -8px; right: 0; background: #ffe066; color: #000; font-weight: 700;
  padding: 2px 8px; border-radius: 4px; font-size: 0.9em; }
.l18-demo .l18-legend { display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; }
.l18-demo .l18-legend span::before { content: ''; display: inline-block; width: 12px; height: 12px; margin-right: 5px;
  vertical-align: -1px; background: var(--c); border: 1px solid #555; }
`;
  document.head.appendChild(Object.assign(document.createElement('style'), {textContent: css}));

  const q = (root, role) => root.querySelector(`[data-role="${role}"]`);
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
  async function image(src) { const img = new Image(); img.src = src; await img.decode(); return img; }
  function radio(root, attr, onPick) {
    const bs = [...root.querySelectorAll(`[${attr}]`)].map(control);
    bs.forEach(b => b.addEventListener('click', () => { bs.forEach(x => x.classList.toggle('on', x === b)); onPick(b.getAttribute(attr)); }));
    return bs;
  }
  function badge(root) {
    if (D && D.placeholder) root.appendChild(Object.assign(document.createElement('div'), {className: 'l18-ph', textContent: 'placeholder data'}));
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
  function gauss(r) { const u = 1 - r(), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  const fmt = (x, d = 1) => x.toLocaleString('en-GB', {minimumFractionDigits: d, maximumFractionDigits: d});

  const init = {};
  window.__L18 = {};

  // ---- 1. The training data: prints and their BLIP captions ----------------------------------------
  init.data = async function (root) {
    badge(root);
    const grid = q(root, 'grid');
    for (const e of D.data.examples) {
      const f = document.createElement('figure');
      const im = Object.assign(document.createElement('img'), {src: e.img, alt: e.title});
      im.title = `${e.title} — ${e.artist}, ${e.date}. Cleveland Museum of Art ${e.accession}, CC0.`;
      f.append(im, Object.assign(document.createElement('figcaption'), {textContent: `“${e.caption}”`}));
      grid.appendChild(f);
    }
    return {stop() {}, api: {}};
  };

  // ---- 2. The style arriving: snapshots through training ------------------------------------------
  init.progress = async function (root) {
    badge(root);
    const S = 300, T = 150;
    const gb = hiDPI(q(root, 'before'), S, S), ga = hiDPI(q(root, 'after'), S, S), gn = hiDPI(q(root, 'near'), T, T);
    const slider = control(q(root, 'step'));
    slider.max = D.stages.length - 1;
    let k = +slider.value, key = D.prompts[0].key, seed = D.seeds[0], variant = 'ema';
    const cache = {};
    const get = async src => cache[src] || (cache[src] = await image(src));
    const pbox = q(root, 'prompts');
    for (const p of D.prompts) {
      const b = Object.assign(document.createElement('button'), {textContent: p.key.startsWith('train') ? `training caption ${+p.key.slice(5) + 1}` : p.key});
      b.dataset.p = p.key;
      if (p.key === key) b.classList.add('on');
      pbox.appendChild(b);
    }
    radio(root, 'data-p', v => { key = v; update(); });
    radio(root, 'data-seed', v => { seed = D.seeds[+v]; update(); });
    radio(root, 'data-w', v => { variant = v; update(); });
    async function update() {
      const s = D.stages[k], p = D.prompts.find(x => x.key === key);
      const [b, a] = await Promise.all([get(D.img[`0-ema-${key}-${seed}`]), get(D.img[`${s}-${variant}-${key}-${seed}`])]);
      gb.drawImage(b, 0, 0, S, S); ga.drawImage(a, 0, 0, S, S);
      const n = D.nearest[`${s}-${key}-${seed}`];
      gn.fillStyle = '#fff'; gn.fillRect(0, 0, T, T);
      if (n && D.thumbs[n.file_name]) gn.drawImage(await get(D.thumbs[n.file_name]), 0, 0, T, T);
      q(root, 'cos').textContent = n ? `similarity ${n.cos.toFixed(2)}` : '';
      q(root, 'text').textContent = `“${p.text}”`;
      q(root, 'tval').textContent = s === 0 ? 'step 0: the original model' :
        `step ${s.toLocaleString('en-GB')} · ${fmt(s / D.steps_per_epoch)} epochs`;
    }
    slider.addEventListener('input', () => { k = +slider.value; update(); });
    await update();
    return {stop() {}, api: {set(o) { if ('k' in o) { k = o.k; slider.value = k; } if ('key' in o) key = o.key;
      if ('variant' in o) variant = o.variant; return update(); }}};
  };

  // ---- 3. Losses: the logged training loss, and the fixed-noise loss on training and held-out prints ----
  init.loss = async function (root) {
    badge(root);
    const W = 470, H = 300, P = {l: 52, r: 12, t: 14, b: 40};
    function axes(g, title, xlab, ylab) {
      g.fillStyle = '#fff'; g.fillRect(0, 0, W, H);
      g.strokeStyle = '#444'; g.lineWidth = 1;
      g.beginPath(); g.moveTo(P.l, P.t); g.lineTo(P.l, H - P.b); g.lineTo(W - P.r, H - P.b); g.stroke();
      g.fillStyle = INK; g.font = `14px ${FONT}`; g.textAlign = 'center';
      g.fillText(xlab, (P.l + W - P.r) / 2, H - 6);
      g.save(); g.translate(14, (P.t + H - P.b) / 2); g.rotate(-Math.PI / 2); g.fillText(ylab, 0, 0); g.restore();
    }
    // (a) per-step training loss, with a running mean
    const ga = hiDPI(q(root, 'train'), W, H);
    const L = D.loss, n = L.length, xmax = L[n - 1][0];
    const ys = L.map(r => r[1]).sort((a, b) => a - b), ymax = ys[Math.floor(0.995 * n)] * 1.05;
    const X = s => P.l + (W - P.l - P.r) * s / xmax, Y = v => H - P.b - (H - P.t - P.b) * Math.min(v, ymax) / ymax;
    axes(ga, '', 'training step', 'loss');
    ga.fillStyle = 'rgba(44, 111, 183, 0.25)';
    for (const [s, v] of L) ga.fillRect(X(s) - 0.7, Y(v) - 0.7, 1.4, 1.4);
    const win = Math.max(25, Math.round(n / 60));
    ga.strokeStyle = ORANGE; ga.lineWidth = 2.4; ga.beginPath();
    let acc = 0;
    for (let i = 0; i < n; i++) {
      acc += L[i][1]; if (i >= win) acc -= L[i - win][1];
      if (i >= win - 1) { const x = X(L[i][0]), y = Y(acc / win); i === win - 1 ? ga.moveTo(x, y) : ga.lineTo(x, y); }
    }
    ga.stroke();
    ga.fillStyle = INK; ga.font = `12px ${FONT}`; ga.textAlign = 'right';
    for (const v of [0, ymax / 2]) ga.fillText(v.toFixed(2), P.l - 5, Y(v) + 4);
    ga.textAlign = 'center';
    for (const s of [0, xmax / 2, xmax]) {
      const v = Math.round(s / 100) * 100; ga.textAlign = s === xmax ? 'right' : 'center';
      ga.fillText(v.toLocaleString('en-GB'), Math.min(X(v), W - 2), H - P.b + 15);
    }

    // (b) fixed-noise loss against step, training vs held-out prints (log step axis; step 0 at the left)
    const gb = hiDPI(q(root, 'eval'), W, H);
    const st = D.stages.filter(s => D.eval[s]);
    const mean = (s, part) => { const e = D.eval[s][part]; const v = Object.values(e); return v.reduce((a, b) => a + b, 0) / v.length; };
    // Plotted as % change from step 0, since the two sets of prints start at slightly different losses.
    const rel = part => st.map(s => 100 * (mean(s, part) / mean(st[0], part) - 1));
    const ser = {train: rel('train'), heldout: rel('heldout')};
    const all = [...ser.train, ...ser.heldout], lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.15 || 0.01;
    const XB = i => P.l + 14 + (W - P.l - P.r - 28) * i / (st.length - 1), YB = v => H - P.b - (H - P.t - P.b) * (v - lo + pad) / (hi - lo + 2 * pad);
    axes(gb, '', 'training step', 'change since step 0 (%)');
    for (const [part, col] of [['train', ORANGE], ['heldout', BLUE]]) {
      gb.strokeStyle = col; gb.fillStyle = col; gb.lineWidth = 2.4; gb.beginPath();
      ser[part].forEach((v, i) => i ? gb.lineTo(XB(i), YB(v)) : gb.moveTo(XB(i), YB(v))); gb.stroke();
      ser[part].forEach((v, i) => { gb.beginPath(); gb.arc(XB(i), YB(v), 3.5, 0, 2 * Math.PI); gb.fill(); });
    }
    gb.fillStyle = INK; gb.font = `12px ${FONT}`; gb.textAlign = 'center';
    st.forEach((s, i) => gb.fillText(s.toLocaleString('en-GB'), XB(i), H - P.b + 15));
    gb.textAlign = 'right';
    gb.strokeStyle = '#ccc'; gb.lineWidth = 1; gb.beginPath(); gb.moveTo(P.l, YB(0)); gb.lineTo(W - P.r, YB(0)); gb.stroke();
    for (const v of [0, Math.ceil(lo)]) gb.fillText(v.toFixed(0), P.l - 5, YB(v) + 4);
    return {stop() {}, api: {ser}};
  };

  // ---- 3b. A static strip: chosen prompts at chosen snapshots (averaged weights, first noise) ---------
  // <div data-demo="strip" data-keys="kimono,bridge" data-stages="0,50,6400" data-seed="1">
  init.strip = async function (root) {
    badge(root);
    const keys = root.dataset.keys.split(','), stages = root.dataset.stages.split(',').map(Number);
    const seed = D.seeds[+(root.dataset.seed || 0)], S = +(root.dataset.size || 120);
    const t = document.createElement('table');
    t.style.cssText = 'border-collapse: collapse; margin: 0 auto;';
    const head = t.insertRow();
    head.insertCell().textContent = '';
    for (const s of stages) {
      const c = head.insertCell(); c.style.cssText = 'text-align: center; font-weight: 600; padding: 0 2px 4px;';
      c.textContent = s === 0 ? 'step 0' : s.toLocaleString('en-GB');
    }
    const print = root.dataset.print;            // optionally, one training print in a last column
    if (print) Object.assign(head.insertCell(), {textContent: 'a training print'}).style.cssText = 'text-align: center; font-weight: 600; padding: 0 2px 4px 14px;';
    for (const ks of keys) {
      const [k, si] = ks.split(':'), sd = si === undefined ? seed : D.seeds[+si];
      const r = t.insertRow(), p = D.prompts.find(x => x.key === k);
      const lab = r.insertCell();
      lab.style.cssText = 'font-style: italic; text-align: right; padding-right: 10px; max-width: 15em; vertical-align: middle;';
      lab.textContent = `“${p.text}”`;
      for (const s of stages) {
        const im = Object.assign(document.createElement('img'), {src: D.img[`${s}-ema-${k}-${sd}`], alt: `${k}, step ${s}`});
        im.style.cssText = `width: ${S}px; height: ${S}px; margin: 0; border: 1px solid #999; display: block;`;
        const c = r.insertCell(); c.style.padding = '2px'; c.appendChild(im);
      }
      if (print) {
        const im = Object.assign(document.createElement('img'), {src: D.thumbs[print], alt: 'training print'});
        im.style.cssText = `height: ${S}px; margin: 0; border: 1px solid #999; display: block;`;
        const c = r.insertCell(); c.style.padding = '2px 2px 2px 14px'; c.appendChild(im);
      }
    }
    root.appendChild(t);
    return {stop() {}, api: {}};
  };

  // ---- 4. Will it fit? Measured peak memory for real settings, split into its parts --------------
  // From probe.json: 10 training steps of the real UNet per setting. Each bar is the measured peak; the
  // parts are what was allocated at the optimiser step (frozen networks, weights, gradients, Adam's state,
  // EMA), and "other" is the rest of the peak: Adam's temporary workspace and activations.
  init.memory = async function (root) {
    badge(root);
    const ROWS = [
      ['fp32', '32-bit, Adam'],
      ['mixed', 'mixed precision'],
      ['mixed+ckpt', '+ gradient checkpointing'],
      ['mixed+ckpt+8bit', '+ 8-bit Adam'],
      ['mixed+ckpt+8bit+ema', '+ EMA copy'],
      ['mixed+ckpt+8bit+ema_cpu', '+ EMA copy in CPU memory'],
      ['mixed+ckpt+ema', 'mixed, checkpointing, EMA, Adam'],
    ];
    const W = 1000, ROWH = 42, TOP = 50, LABW = 300, RIGHT = 130, MAXGB = 24;
    const cfg = D.probe.configs;
    let bs = 1;
    const c = q(root, 'bar');
    const PARTS = [
      ['frozen text encoder + VAE', 'mem_frozen_gb', '#c9c9c9'],
      ['UNet weights', 'mem_weights_gb', '#9db7d8'],
      ['gradients', 'mem_grads_gb', '#f2b38c'],
      ["Adam's averages", 'mem_optimiser_gb', '#e8875a'],
      ['EMA copy', 'mem_ema_gb', '#b9d9a8'],
    ];
    const GPUS = [[8, 'lab GPU, 8 GB'], [12.4, 'Radeon 890M'], [24, '24 GB']];
    function rows() {
      return ROWS.map(([k, lab]) => [bs === 1 ? k : `${k}/bs4`, lab]).filter(([k]) => cfg[k] && cfg[k].fits);
    }
    function draw() {
      const rs = rows(), H = TOP + rs.length * ROWH + 34;
      const g = hiDPI(c, W, H);
      const X = gb => LABW + (W - LABW - RIGHT) * Math.min(gb, MAXGB) / MAXGB;
      g.fillStyle = '#fff'; g.fillRect(0, 0, W, H);
      rs.forEach(([k, lab], i) => {
        const r = cfg[k], y = TOP + i * ROWH + 4, h = ROWH - 10;
        g.fillStyle = INK; g.font = `600 17px ${FONT}`; g.textAlign = 'right'; g.fillText(lab, LABW - 10, y + h / 2 + 6);
        let x = LABW, sum = 0;
        const segs = PARTS.map(([, f, col]) => [r[f] * GiB, col]);
        const known = segs.reduce((a, [v]) => a + v, 0);
        segs.push([Math.max(0, r.peak_gb * GiB - known), '#d4c1e8']);
        for (const [v, col] of segs) {
          const w = X(sum + v) - X(sum); sum += v;
          if (w <= 0) continue;
          g.fillStyle = col; g.fillRect(x, y, w, h); g.strokeStyle = '#555'; g.lineWidth = 1; g.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
          x += w;
        }
        g.fillStyle = INK; g.font = `600 17px ${FONT}`; g.textAlign = 'left';
        g.fillText(`${fmt(r.peak_gb * GiB)} GB`, X(sum) + 6, y + h / 2 + 6);
        g.font = `15px ${FONT}`; g.fillStyle = '#555';
        g.fillText(`${fmt(r.sec_per_micro_batch * 1000 / r.batch, 0)} ms/image`, W - RIGHT + 22, y + h / 2 + 6);
      });
      GPUS.forEach(([gb, lab], i) => {
        const gx = X(gb), top = TOP - (i % 2 ? 8 : 28);
        g.strokeStyle = '#c92a2a'; g.lineWidth = 2; g.setLineDash([5, 4]);
        g.beginPath(); g.moveTo(gx, top + 4); g.lineTo(gx, TOP + rs.length * ROWH); g.stroke(); g.setLineDash([]);
        g.fillStyle = '#c92a2a'; g.font = `600 15px ${FONT}`; g.textAlign = 'center'; g.fillText(lab, gx, top);
      });
      g.fillStyle = INK; g.font = `14px ${FONT}`; g.textAlign = 'center';
      const yb = TOP + rs.length * ROWH + 16;
      for (let v = 0; v <= MAXGB; v += 4) g.fillText(`${v}`, X(v), yb);
      g.fillText('GB', X(MAXGB), yb + 15);
      q(root, 'legend').innerHTML = [...PARTS.map(([n, , col]) => [n, col]), ['other: workspace and activations', '#d4c1e8']]
        .map(([n, col]) => `<span style="--c:${col}">${n}</span>`).join('');
    }
    radio(root, 'data-bs', v => { bs = +v; draw(); });
    draw();
    return {stop() {}, api: {set(o) { if ('bs' in o) bs = o.bs; draw(); }, rows}};
  };

  // ---- 5. EMA toy: noisy gradient descent on a bowl, and the running average of the weights --------
  init.ema = async function (root) {
    const S = 440, LIM = 3, STEPS = 1500, LR = 0.03, NOISE = 2.2, A = 1, B = 5;
    const c = q(root, 'plot'), g = hiDPI(c, S, S);
    const px = v => S / 2 + v * S / (2 * LIM), py = v => S / 2 - v * S / (2 * LIM);
    let beta = 0.99, seed = 3;
    function run() {
      const r = rng(seed); let x = -2.4, y = 2.2;
      const raw = [[x, y]];
      for (let k = 0; k < STEPS; k++) {
        x -= LR * (A * x + NOISE * gauss(r)); y -= LR * (B * y + NOISE * gauss(r));
        raw.push([x, y]);
      }
      const avg = [raw[0].slice()];
      for (let k = 1; k <= STEPS; k++) {
        const [ax, ay] = avg[k - 1];
        avg.push([beta * ax + (1 - beta) * raw[k][0], beta * ay + (1 - beta) * raw[k][1]]);
      }
      return {raw, avg};
    }
    function draw() {
      const {raw, avg} = run();
      g.fillStyle = '#fff'; g.fillRect(0, 0, S, S);
      g.strokeStyle = '#ddd'; g.lineWidth = 1;
      for (const lev of [0.25, 1, 2.25, 4, 6.25, 9]) {       // contours of the loss 0.5 (A x^2 + B y^2)
        g.beginPath(); g.ellipse(px(0), py(0), Math.sqrt(2 * lev / A) * S / (2 * LIM), Math.sqrt(2 * lev / B) * S / (2 * LIM), 0, 0, 2 * Math.PI); g.stroke();
      }
      g.strokeStyle = 'rgba(80, 80, 80, 0.45)'; g.lineWidth = 1; g.beginPath();
      raw.forEach(([x, y], i) => i ? g.lineTo(px(x), py(y)) : g.moveTo(px(x), py(y))); g.stroke();
      g.strokeStyle = ORANGE; g.lineWidth = 3; g.beginPath();
      avg.forEach(([x, y], i) => i ? g.lineTo(px(x), py(y)) : g.moveTo(px(x), py(y))); g.stroke();
      g.fillStyle = INK; g.beginPath(); g.arc(px(0), py(0), 4, 0, 2 * Math.PI); g.fill();
      for (const [p, col] of [[raw[STEPS], '#555'], [avg[STEPS], ORANGE]]) { g.fillStyle = col; g.beginPath(); g.arc(px(p[0]), py(p[1]), 6, 0, 2 * Math.PI); g.fill(); }
      g.strokeStyle = '#ccc'; g.strokeRect(0.5, 0.5, S - 1, S - 1);
      const d = p => Math.hypot(p[0], p[1]);
      const tail = a => a.slice(STEPS / 2).reduce((s, p) => s + d(p), 0) / (STEPS / 2);
      q(root, 'bval').textContent = `β = ${beta}`;
      q(root, 'dist').innerHTML = `Average distance from the lowest point, second half of training:<br>` +
        `raw weights <b>${tail(raw).toFixed(2)}</b>, averaged weights <b style="color:${ORANGE}">${tail(avg).toFixed(2)}</b>`;
    }
    radio(root, 'data-b', v => { beta = +v; draw(); });
    control(q(root, 'again')).addEventListener('click', () => { seed += 1; draw(); });
    draw();
    return {stop() {}, api: {set(o) { if ('beta' in o) beta = o.beta; if ('seed' in o) seed = o.seed; draw(); }}};
  };

  const running = [];
  async function start() {
    for (const root of document.querySelectorAll('.l18-demo[data-demo]')) {
      const name = root.dataset.demo;
      try {
        const d = await init[name](root);
        running.push(d); window.__L18[name] = d.api;
      } catch (e) { console.error('lec18 demo', name, e); }
    }
    window.__L18.ready = true;
    if (window.Reveal && Reveal.on) Reveal.on('slidechanged', () => running.forEach(d => d.stop()));
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
