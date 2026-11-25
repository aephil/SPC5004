// Live demos for lec19.qmd ... lec22.qmd. Each <div class="dm-demo" data-demo="NAME"> in a deck holds the static
// layout (so the slide text is in the .qmd); init[NAME] wires up its images and controls. Every demo only switches
// between images made in advance by the GPU jobs in pics/lec19_22_gpu/; pics/lec19_22_figs.py packs them into
// pics/lecNN/demo/ and lists them in pics/lecNN/demos-data.js (window.LNNDATA).
// Testing aid: window.__DM[NAME] exposes each demo's setters; window.__DM.ready is set once all have started.
(function () {
  'use strict';
  const ORANGE = '#d9480f', BLUE = '#2c6fb7', INK = '#222';
  const FONT = '"IBM Plex Sans", sans-serif';

  const css = `
.dm-demo { font-size: 0.6em; line-height: 1.3; position: relative; }
.dm-demo .dm-row { display: flex; gap: 20px; align-items: flex-start; justify-content: center; }
.dm-demo .dm-col { display: flex; flex-direction: column; align-items: center; gap: 5px; }
.dm-demo .dm-lab { font-weight: 600; }
.dm-demo img.dm-pix { border: 1px solid #555; margin: 0; width: var(--w, 280px); height: var(--w, 280px); background: #eee; }
.dm-demo .dm-ctl { display: flex; gap: 10px; align-items: center; justify-content: center; margin-top: 9px; flex-wrap: wrap; }
.dm-demo input[type=range] { width: 300px; accent-color: ${ORANGE}; }
.dm-demo button { font: inherit; font-size: 0.95em; padding: 3px 11px; border: 1px solid #888; border-radius: 4px;
  background: #fff; color: #000; cursor: pointer; }
.dm-demo button:hover { background: #f3f3f3; }
.dm-demo button.on { background: ${ORANGE}; border-color: ${ORANGE}; color: #fff; }
.dm-demo button:disabled { opacity: 0.35; cursor: default; }
.dm-demo input[type=range]:disabled { opacity: 0.35; }
.dm-demo .dm-out { min-width: 6.5em; font-variant-numeric: tabular-nums; }
.dm-demo .dm-prompt { font-style: italic; text-align: center; min-height: 1.4em; margin-top: 6px; }
.dm-demo .dm-legend { display: flex; gap: 18px; flex-wrap: wrap; justify-content: center; }
.dm-demo .dm-legend span::before { content: ''; display: inline-block; width: 22px; height: 0; margin-right: 6px;
  vertical-align: 4px; border-top: 3px var(--s, solid) var(--c); }
`;
  document.head.appendChild(Object.assign(document.createElement('style'), {textContent: css}));

  const q = (root, role) => root.querySelector(`[data-role="${role}"]`);
  // Python's str() of a float, as used in the result file names: 1 -> "1.0", 0.25 -> "0.25"
  const py = x => Number.isInteger(x) ? x.toFixed(1) : String(x);
  const fmt = (x, d = 2) => x.toLocaleString('en-GB', {minimumFractionDigits: d, maximumFractionDigits: d});

  function control(e) {
    e.addEventListener('pointerup', () => setTimeout(() => e.blur(), 0));
    if (e.tagName === 'BUTTON') e.addEventListener('click', () => e.blur());
    e.addEventListener('keydown', ev => { if (e.type === 'range' && ev.key.startsWith('Arrow')) ev.stopPropagation(); });
    return e;
  }
  // A row of buttons, one per [value, label]; calls onPick(value). Returns a setter.
  function buttons(box, items, value, onPick) {
    const bs = items.map(([v, label]) => {
      const b = control(Object.assign(document.createElement('button'), {textContent: label}));
      b.dataset.v = v;
      b.addEventListener('click', () => { set(v); onPick(v); });
      box.appendChild(b);
      return b;
    });
    function set(v) { bs.forEach(b => b.classList.toggle('on', b.dataset.v === String(v))); }
    set(value);
    return {set, buttons: bs};
  }
  // A range input over the indices of `values`; calls onMove(value).
  function slider(e, values, value, onMove) {
    control(e);
    e.min = 0; e.max = values.length - 1; e.step = 1; e.value = Math.max(0, values.indexOf(value));
    e.addEventListener('input', () => { if (!e.disabled) onMove(values[+e.value]); });
    return {set(v) { e.value = values.indexOf(v); }, el: e};
  }
  function show(el, src, alt) {
    if (src) { el.src = src; el.style.visibility = 'visible'; } else el.style.visibility = 'hidden';
    if (alt) el.alt = alt;
  }
  function preload(D) {
    const go = () => Object.values(D.img).forEach(src => { const i = new Image(); i.src = src; });
    if (window.requestIdleCallback) window.requestIdleCallback(go, {timeout: 3000}); else setTimeout(go, 1500);
  }
  function hiDPI(c, w, h) {
    const r = window.devicePixelRatio > 1 ? 2 : 1;
    c.width = w * r; c.height = h * r; c.style.width = w + 'px'; c.style.height = h + 'px';
    const g = c.getContext('2d'); g.setTransform(r, 0, 0, r, 0, 0);
    return g;
  }

  const init = {};
  window.__DM = {};

  // ================================================================ lec19
  // Both runs side by side at the same training step, prompt and noise.
  init.db = function (root) {
    const D = window.L19DATA;
    let k = 0, key = 'subject', seed = 1;
    const st = slider(q(root, 'step'), D.steps, 0, v => { k = v; update(); });
    const pb = buttons(q(root, 'prompts'), D.keys.map(x => [x, x === 'subject' ? 'sks dog' : x]), key, v => { key = v; update(); });
    const sb = buttons(q(root, 'seeds'), [[1, 'A'], [2, 'B']], seed, v => { seed = +v; update(); });
    function update() {
      for (const run of ['prior', 'noprior']) {
        const id = `${run}-${k}-${key}-${seed}`;
        show(q(root, run), D.img[id], `“${D.prompts[key]}” after ${k} steps, ${run === 'prior' ? 'with' : 'without'} prior preservation`);
        const s = D.dino[id];
        q(root, run + '-sim').textContent = s == null ? '' : `similarity to the subject ${fmt(s)}`;
      }
      q(root, 'text').textContent = `“${D.prompts[key]}”`;
      q(root, 'sval').textContent = k === 0 ? 'step 0: the original model' : `after ${k.toLocaleString('en-GB')} steps`;
    }
    update(); preload(D);
    return {set(o) { if ('k' in o) { k = o.k; st.set(k); } if ('key' in o) { key = o.key; pb.set(key); }
      if ('seed' in o) { seed = o.seed; sb.set(seed); } update(); }};
  };

  // DINO similarity to the subject photos against training step, for both runs.
  init.dbcurve = function (root) {
    const D = window.L19DATA;
    const W = 480, H = 330, P = {l: 52, r: 84, t: 14, b: 44};
    const g = hiDPI(q(root, 'chart'), W, H);
    const xs = D.steps, xmax = 1600, ymin = 0.2, ymax = 0.9;
    const X = s => P.l + (W - P.l - P.r) * s / xmax, Y = v => H - P.b - (H - P.t - P.b) * (v - ymin) / (ymax - ymin);
    g.fillStyle = '#fff'; g.fillRect(0, 0, W, H);
    g.font = `13px ${FONT}`; g.fillStyle = INK; g.strokeStyle = '#ddd'; g.lineWidth = 1;
    g.textAlign = 'right';
    for (let v = 0.2; v <= 0.901; v += 0.1) { g.beginPath(); g.moveTo(P.l, Y(v)); g.lineTo(W - P.r, Y(v)); g.stroke(); g.fillText(fmt(v, 1), P.l - 6, Y(v) + 4); }
    g.textAlign = 'center';
    for (const s of [0, 400, 800, 1200, 1600]) g.fillText(s.toLocaleString('en-GB'), X(s), H - P.b + 18);
    g.font = `14px ${FONT}`;
    g.fillText('training step', (P.l + W - P.r) / 2, H - 6);
    g.save(); g.translate(16, (P.t + H - P.b) / 2); g.rotate(-Math.PI / 2); g.fillText('similarity to the subject', 0, 0); g.restore();
    // reference levels
    g.setLineDash([2, 4]); g.strokeStyle = '#777'; g.lineWidth = 1.5; g.textAlign = 'left'; g.fillStyle = '#555'; g.font = `13px ${FONT}`;
    for (const [v, lab] of [[D.ref.subject_self, 'the 5 photos'], [D.ref.class_to_subject, 'class images']]) {
      g.beginPath(); g.moveTo(P.l, Y(v)); g.lineTo(W - P.r, Y(v)); g.stroke(); g.fillText(lab, W - P.r + 6, Y(v) + 4);
    }
    g.setLineDash([]);
    for (const [run, col] of [['prior', BLUE], ['noprior', ORANGE]]) {
      for (const [what, dash] of [['subject', []], ['dog', [7, 5]]]) {
        const ys = D.curves[run][what];
        g.strokeStyle = col; g.lineWidth = 2.6; g.setLineDash(dash); g.beginPath();
        xs.forEach((s, i) => i ? g.lineTo(X(s), Y(ys[i])) : g.moveTo(X(s), Y(ys[i]))); g.stroke();
        g.setLineDash([]); g.fillStyle = col;
        xs.forEach((s, i) => { g.beginPath(); g.arc(X(s), Y(ys[i]), 3.2, 0, 2 * Math.PI); g.fill(); });
      }
    }
    return {set() {}};
  };

  // ================================================================ lec20
  init.i2i = function (root) {
    const D = window.L20DATA;
    let s = 0.6, src = 'cat', seed = 1;
    const st = slider(q(root, 'strength'), D.strengths, s, v => { s = v; update(); });
    const sb = buttons(q(root, 'sources'), [['cat', 'cat → fox'], ['sketch', 'sketch → cottage']], src, v => { src = v; update(); });
    buttons(q(root, 'seeds'), [[1, 'A'], [2, 'B']], seed, v => { seed = +v; update(); });
    function update() {
      const a = D.start[py(s)];
      show(q(root, 'input'), D.img[`${src}-input`]);
      show(q(root, 'start'), D.img[`${src}-start-s${py(s)}`]);
      show(q(root, 'output'), D.img[`${src}-s${py(s)}-${seed}`]);
      show(q(root, 't2i'), D.img[`${src}-t2i-${seed}`]);
      q(root, 'text').textContent = `“${D.sources[src]}”`;
      q(root, 'sval').textContent = `strength ${fmt(s, 1)}`;
      q(root, 'info').innerHTML = `starts at <i>t</i> = ${a.t}, runs the last ${a.steps_run} of ${D.steps} steps`;
      q(root, 'left').textContent = fmt(Math.sqrt(a.alphabar));
    }
    update(); preload(D);
    return {set(o) { if ('s' in o) { s = o.s; st.set(s); } if ('src' in o) { src = o.src; sb.set(src); } update(); }};
  };

  init.inpaint = function (root) {
    const D = window.L20DATA;
    let mask = 'tight', s = 1.0, seed = 1;
    buttons(q(root, 'masks'), [['tight', 'tight'], ['loose', 'loose']], mask, v => { mask = v; update(); });
    buttons(q(root, 'strengths'), [['0.75', '0.75'], ['1.0', '1']], py(s), v => { s = +v; update(); });
    buttons(q(root, 'seeds'), [[1, 'A'], [2, 'B']], seed, v => { seed = +v; update(); });
    function update() {
      show(q(root, 'photo'), D.img[`coffee-${mask}`]);
      for (const m of ['blend', 'model']) show(q(root, m), D.img[`inpaint-${m}-${mask}-s${py(s)}-${seed}`]);
    }
    q(root, 'text').textContent = `“${D.inpaint_prompt}”`;
    update(); preload(D);
    return {set(o) { if ('mask' in o) mask = o.mask; if ('s' in o) s = o.s; update(); }};
  };

  // ================================================================ lec21
  init.weight = function (root) {
    const D = window.L21DATA, W = D.weight;
    let w = 1.5, key = 'snow', seed = 3;
    const st = slider(q(root, 'w'), W.weights, w, v => { w = v; update(); });
    buttons(q(root, 'prompts'), [['snow', 'snowy cottage'], ['red', 'red car']], key, v => { key = v; update(); });
    buttons(q(root, 'seeds'), [[3, 'A'], [4, 'B']], seed, v => { seed = +v; update(); });
    function update() {
      show(q(root, 'ref'), D.img[`weight-${key}-w1.0-${seed}`]);
      show(q(root, 'out'), D.img[`weight-${key}-w${py(w)}-${seed}`]);
      q(root, 'text').textContent = '“' + W.prompts[key].replace(')w', `)${fmt(w, 1)}`) + '”';
      q(root, 'wval').textContent = `weight ${fmt(w, 1)}`;
    }
    update(); preload(D);
    return {set(o) { if ('w' in o) { w = o.w; st.set(w); } if ('key' in o) key = o.key; update(); }};
  };

  const SEGA_EDITS = [['autumn', '+ autumn'], ['snow', '+ snow'], ['cars', '− cars'], ['autumn+cars', '+ autumn − cars']];
  // Seeds and edit scales come from the data (the lec21 retune notebook records them); older data used 21/22 and 6.
  function segaSettings(D) {
    const S = D.sega, scale = S.scale || {autumn: 6, snow: 6, cars: 6};
    const gFor = edit => { const v = edit.split('+').map(x => scale[x]); return v.every(x => x === v[0]) ? `${v[0]}` : v.join('_'); };
    return {seeds: S.seeds || [21, 22], scale, gFor, sweep: S.autumn_sweep || [2, 4, 6, 8, 12]};
  }
  init.sega = function (root) {
    const D = window.L21DATA, Z = segaSettings(D);
    let edit = 'autumn', g = Z.scale.autumn, seed = Z.seeds[0];
    const st = slider(q(root, 'g'), Z.sweep, g, v => { g = v; update(); });
    buttons(q(root, 'edits'), SEGA_EDITS, edit, v => { edit = v; if (edit !== 'autumn') { g = Z.scale.autumn; st.set(g); } update(); });
    buttons(q(root, 'seeds'), [[Z.seeds[0], 'A'], [Z.seeds[1], 'B']], seed, v => { seed = +v; update(); });
    function update() {
      st.el.disabled = edit !== 'autumn';
      const gs = edit === 'autumn' ? `${g}` : Z.gFor(edit);
      const id = `${edit}-g${gs}-th0.95-${seed}`;
      show(q(root, 'none'), D.img[`sega-none-g0-th0.95-${seed}`]);
      show(q(root, 'out'), D.img[`sega-${id}`]);
      show(q(root, 'map'), D.img[`sega-map-${id}`]);
      q(root, 'mapnote').textContent = D.img[`sega-map-${id}`] ? '' : '(recorded for noise A only)';
      q(root, 'gval').textContent = edit === 'autumn' ? `edit scale ${g}`
        : `edit scale ${gs.replace('_', ' and ')} (only autumn was run at other scales)`;
      const e = edit.split('+').map(x => (D.sega.edits[x][1] ? '− ' : '+ ') + `“${D.sega.edits[x][0]}”`).join('  ');
      q(root, 'text').textContent = `“${D.sega.prompt}”  ${e}`;
    }
    update(); preload(D);
    return {set(o) { if ('edit' in o) edit = o.edit; if ('g' in o) { g = o.g; st.set(g); } update(); }};
  };

  init.segath = function (root) {
    const D = window.L21DATA, Z = segaSettings(D), A = Z.seeds[0];
    let th = 0.95;
    const ths = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99];
    const st = slider(q(root, 'th'), ths, th, v => { th = v; update(); });
    function update() {
      const id = `autumn-g${Z.scale.autumn}-th${py(th)}-${A}`;
      show(q(root, 'out'), D.img[`sega-${id}`]);
      show(q(root, 'map'), D.img[`sega-map-${id}`]);
      q(root, 'thval').textContent = `threshold ${th}: ` + (th === 0 ? 'every element' : `the largest ${Math.round((1 - th) * 100)}% at each step`);
    }
    show(q(root, 'none'), D.img[`sega-none-g0-th0.95-${A}`]);
    update();
    return {set(o) { if ('th' in o) { th = o.th; st.set(th); } update(); }};
  };

  init.p2p = function (root) {
    const D = window.L21DATA, P = D.pix2pix;
    let task = 'cat', si = 1.5, stv = 7.5;
    const a = slider(q(root, 'si'), P.s_I, si, v => { si = v; update(); });
    const b = slider(q(root, 'st'), P.s_T, stv, v => { stv = v; update(); });
    buttons(q(root, 'tasks'), Object.keys(P.tasks).map(k => [k, k]), task, v => { task = v; update(); });
    function update() {
      show(q(root, 'input'), D.img[`p2p-${task}-input`]);
      show(q(root, 'out'), D.img[`p2p-${task}-sI${py(si)}-sT${py(stv)}`]);
      q(root, 'text').textContent = `“${P.tasks[task]}”`;
      q(root, 'sival').innerHTML = `<i>s</i><sub>I</sub> = ${si}`;
      q(root, 'stval').innerHTML = `<i>s</i><sub>T</sub> = ${stv}`;
    }
    update(); preload(D);
    return {set(o) { if ('task' in o) task = o.task; if ('si' in o) { si = o.si; a.set(si); } if ('st' in o) { stv = o.st; b.set(stv); } update(); }};
  };

  // ================================================================ lec22
  init.ip = function (root) {
    const D = window.L22DATA, I = D.ip;
    let s = 0.4, prompt = 'bicycle', ref = 'hiroshige';
    const st = slider(q(root, 'scale'), I.scales, s, v => { s = v; update(); });
    buttons(q(root, 'prompts'), Object.keys(I.prompts).map(k => [k, k]), prompt, v => { prompt = v; update(); });
    buttons(q(root, 'refs'), [['hiroshige', 'woodblock print'], ['vangogh', 'van Gogh']], ref, v => { ref = v; update(); });
    function update() {
      show(q(root, 'ref'), D.img[`ref-${ref}`]);
      show(q(root, 'out'), D.img[`ip-${prompt}-${ref}-s${py(s)}-11`]);
      show(q(root, 'words'), D.img[`ip-${prompt}-text${ref}-11`]);
      q(root, 'text').textContent = `“${I.prompts[prompt]}”`;
      q(root, 'wtext').textContent = `“… , ${I.ref_words[ref]}”`;
      q(root, 'sval').textContent = `scale λ = ${fmt(s, 1)}`;
    }
    update(); preload(D);
    return {set(o) { if ('s' in o) { s = o.s; st.set(s); } if ('prompt' in o) prompt = o.prompt; if ('ref' in o) ref = o.ref; update(); }};
  };

  init.cn = function (root) {
    const D = window.L22DATA, C = D.cn;
    let map = 'depth', s = 1.0;
    const st = slider(q(root, 'scale'), C.scales, s, v => { s = v; update(); });
    buttons(q(root, 'maps'), [['canny', 'edges'], ['depth', 'depth'], ['pose', 'pose']], map, v => { map = v; update(); });
    function update() {
      show(q(root, 'map'), D.img[`cn-${map}`]);
      show(q(root, 'out'), D.img[`cn-${map}-knight-c${py(s)}`]);
      q(root, 'sval').textContent = `conditioning scale ${fmt(s, 1)}`;
    }
    q(root, 'text').textContent = `“${C.prompts.knight}”`;
    update(); preload(D);
    return {set(o) { if ('s' in o) { s = o.s; st.set(s); } if ('map' in o) map = o.map; update(); }};
  };

  function start() {
    for (const root of document.querySelectorAll('.dm-demo[data-demo]')) {
      const name = root.dataset.demo;
      try { window.__DM[name] = init[name](root); } catch (e) { console.error('demo', name, e); }
    }
    window.__DM.ready = true;
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
