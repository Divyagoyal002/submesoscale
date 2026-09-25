// Live super-resolution explorer. Runs the trained CNN in the browser with
// onnxruntime-web and derives currents / vorticity with physics.js.
import { makeGrid, derive, superResolve, smoothNoise, rmse, corr, maxAbsDiff } from "./physics.js";

const $ = (id) => document.getElementById(id);
const DATA = "data/";
const LAYERS = {
  adt: { label: "Sea level", cmap: "viridis", unit: "cm", k: 100, digits: 1 },
  speed: { label: "Current speed", cmap: "magma", unit: "m/s", k: 1, digits: 2 },
  ro: { label: "Eddy spin ζ/f", cmap: "RdBu_r", unit: "", k: 1, digits: 2 },
};

const state = { day: 0, layer: "adt", arrows: true, noiseCm: 0, hideSst: false, playing: false, hover: null };
let manifest, cmaps, grid, mask, ort, session, engineError = null;
const dayCache = new Map();
const aiCache = new Map();
let aiView = null; // derived fields currently shown in the AI panel
let renderToken = 0;

// ------------------------------------------------------------------ data

async function loadDay(t) {
  if (dayCache.has(t)) return dayCache.get(t);
  const buf = await (await fetch(`${DATA}day_${String(t).padStart(3, "0")}.bin`)).arrayBuffer();
  const n = grid.ny * grid.nx, q = new Int16Array(buf), out = {};
  manifest.fields.forEach((f, k) => {
    const a = new Float32Array(n);
    for (let p = 0; p < n; p++) {
      const v = q[k * n + p];
      a[p] = v === manifest.nodata ? NaN : f.offset + f.scale * v;
    }
    out[f.name] = a;
  });
  const day = { raw: out, lr: derive(out.adt_lr, grid), truth: out.adt_truth ? derive(out.adt_truth, grid) : null };
  dayCache.set(t, day);
  return day;
}

async function loadEngine() {
  try {
    ort = await import("../vendor/ort-1.30.0/ort.wasm.min.mjs");
    ort.env.wasm.wasmPaths = new URL("../vendor/ort-1.30.0/", import.meta.url).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    session = await ort.InferenceSession.create(manifest.model.file, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  } catch (err) {
    engineError = err;
    console.error("AI engine failed to load", err);
  }
}

// ------------------------------------------------------------------ AI

function inputFor(day, t) {
  if (!state.noiseCm) return day.raw.adt_lr;
  const noise = smoothNoise(grid, mask, state.noiseCm / 100, 1000 + t);
  return Float32Array.from(day.raw.adt_lr, (x, p) => x + noise[p]);
}

async function runAI(t, day, onProgress) {
  const key = `${t}|${state.noiseCm}|${state.hideSst}`;
  if (aiCache.has(key)) return aiCache.get(key);
  const adtIn = inputFor(day, t);
  const t0 = performance.now();
  const { adt, tiles } = await superResolve(ort, session, {
    adt: adtIn, sst: day.raw.sst_in, mask, g: grid, stats: manifest.stats,
    tile: manifest.model.tile, overlap: manifest.model.overlap, hideSst: state.hideSst, onProgress,
  });
  const ms = performance.now() - t0;
  const pristine = !state.noiseCm && !state.hideSst;
  const res = {
    view: derive(adt, grid), input: state.noiseCm ? derive(adtIn, grid) : day.lr, ms, tiles,
    diffMm: pristine ? maxAbsDiff(adt, day.raw.adt_ref) * 1000 : null,
  };
  aiCache.set(key, res);
  return res;
}

// ------------------------------------------------------------------ drawing

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function range(layer) {
  const d = manifest.display;
  if (layer === "adt") return [d.adt[0], d.adt[1]];
  if (layer === "speed") return [0, d.speed_max];
  return [-d.ro_abs, d.ro_abs];
}

function paintField(canvas, field, layer, lo, hi, cmapName) {
  const { nx, ny } = grid;
  canvas.width = nx; canvas.height = ny;
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(nx, ny);
  const lut = cmaps[cmapName], land = hexToRgb(cssVar("--land"));
  for (let i = 0; i < ny; i++) {
    for (let j = 0; j < nx; j++) {
      const v = field[i * nx + j], o = ((ny - 1 - i) * nx + j) * 4; // north up
      let c = land;
      if (Number.isFinite(v) && mask[i * nx + j]) c = lut[Math.max(0, Math.min(255, Math.round(((v - lo) / (hi - lo)) * 255)))];
      img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function hexToRgb(h) {
  const m = h.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(m.slice(i, i + 2), 16));
}

function refSpeed() {
  const p = manifest.display.speed_max * 0.5;
  return [0.05, 0.1, 0.2, 0.3, 0.5, 1].reduce((a, b) => (Math.abs(Math.log(b / p)) < Math.abs(Math.log(a / p)) ? b : a));
}

function paintOverlay(canvas, view) {
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const { nx, ny } = grid, sx = w / nx, sy = h / ny;
  if (state.arrows && view) {
    const every = Math.max(1, Math.round(nx / 26)), spacing = every * sx, ref = refSpeed();
    ctx.strokeStyle = state.layer === "speed" ? "rgba(255,255,255,.85)" : "rgba(20,20,19,.8)";
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = 1.1;
    for (let i = Math.floor(every / 2); i < ny; i += every) {
      for (let j = Math.floor(every / 2); j < nx; j += every) {
        const p = i * nx + j, u = view.u[p], v = view.v[p];
        if (!Number.isFinite(u) || !Number.isFinite(v)) continue;
        const x0 = (j + 0.5) * sx, y0 = (ny - 1 - i + 0.5) * sy;
        const L = Math.min(2.2, Math.hypot(u, v) / ref) * spacing * 0.9;
        if (L < 1.5) continue;
        const a = Math.atan2(-v, u), x1 = x0 + L * Math.cos(a), y1 = y0 + L * Math.sin(a), hl = Math.min(5, L * 0.4);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - hl * Math.cos(a - 0.45), y1 - hl * Math.sin(a - 0.45));
        ctx.lineTo(x1 - hl * Math.cos(a + 0.45), y1 - hl * Math.sin(a + 0.45));
        ctx.closePath(); ctx.fill();
      }
    }
  }
  if (state.hover) {
    const { i, j } = state.hover, x = (j + 0.5) * sx, y = (ny - 1 - i + 0.5) * sy;
    ctx.strokeStyle = "rgba(255,255,255,.95)"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.stroke();
    ctx.strokeStyle = "rgba(20,20,19,.95)"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.stroke();
  }
}

function paintColorbar(canvas, cmapName) {
  canvas.width = 256; canvas.height = 1;
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(256, 1);
  cmaps[cmapName].forEach((c, k) => img.data.set([c[0], c[1], c[2], 255], k * 4));
  ctx.putImageData(img, 0, 0);
}

const fmt = (v, layer) => {
  const L = LAYERS[layer];
  return Number.isFinite(v) ? `${(v * L.k).toFixed(L.digits)}${L.unit ? " " + L.unit : ""}` : "land";
};

// ------------------------------------------------------------------ panels

const panels = () => [
  { id: "lr", view: () => (aiCache.get(aiKey())?.input) || dayCache.get(state.day)?.lr },
  { id: "sr", view: () => aiView },
  { id: "third", view: () => thirdView() },
];

function aiKey() { return `${state.day}|${state.noiseCm}|${state.hideSst}`; }

function thirdView() {
  const day = dayCache.get(state.day);
  if (!day) return null;
  if (day.truth) return day.truth;
  if (!aiView) return null;
  // real data: show the detail the AI added (AI minus input) in the same units
  const diff = (a, b) => Float32Array.from(a, (x, p) => x - b[p]);
  return { adt: diff(aiView.adt, day.lr.adt), speed: diff(aiView.speed, day.lr.speed), ro: diff(aiView.ro, day.lr.ro), u: aiView.u, v: aiView.v, isDiff: true };
}

function drawPanels() {
  const layer = state.layer, [lo, hi] = range(layer), L = LAYERS[layer];
  for (const p of panels()) {
    const view = p.view(), field = $(`f-${p.id}`), over = $(`o-${p.id}`);
    if (view) {
      if (view.isDiff) {
        const m = layer === "adt" ? 0.03 : layer === "speed" ? manifest.display.speed_max / 3 : 0.3;
        paintField(field, view[layer], layer, -m, m, "RdBu_r");
      } else {
        paintField(field, view[layer], layer, lo, hi, L.cmap);
      }
    }
    paintOverlay(over, view && !view.isDiff ? view : null);
  }
  $("cb-label").textContent = `${L.label}${L.unit ? ` (${L.unit})` : ""}`;
  paintColorbar($("cb"), L.cmap);
  $("cb-lo").textContent = (lo * L.k).toFixed(L.digits);
  $("cb-hi").textContent = (hi * L.k).toFixed(L.digits);
  updateReadouts();
}

function updateReadouts() {
  const h = state.hover;
  for (const p of panels()) {
    const el = $(`r-${p.id}`), view = p.view();
    if (!h || !view) { el.textContent = " "; continue; }
    const k = h.i * grid.nx + h.j, v = view[state.layer][k];
    el.textContent = `${view.isDiff ? "Δ " : ""}${fmt(v, state.layer)} · ${fmt(view.speed[k], "speed")}`;
  }
  $("where").textContent = h ? `${grid.lat[h.i].toFixed(2)}°N, ${grid.lon[h.j].toFixed(2)}°E` : "Hover a map to compare the same point";
}

function updateStats(day, res) {
  const box = $("stats");
  if (!day.truth || !res) { box.hidden = !res; if (res) fillRealStats(day, res); return; }
  box.hidden = false;
  const eIn = rmse(res.input.speed, day.truth.speed) * 100, eAi = rmse(res.view.speed, day.truth.speed) * 100;
  const rIn = corr(res.input.ro, day.truth.ro), rAi = corr(res.view.ro, day.truth.ro);
  box.innerHTML = `
    <div class="card"><div class="v">${eIn.toFixed(1)} cm/s</div><div class="l">Speed error of the satellite input</div></div>
    <div class="card"><div class="v">${eAi.toFixed(1)} cm/s</div><div class="l">Speed error after the AI</div></div>
    ${changeCard(eAi, eIn)}
    <div class="card"><div class="v">${rIn.toFixed(2)} → ${rAi.toFixed(2)}</div><div class="l">Eddy-spin match with truth (correlation)</div></div>`;
}

function changeCard(after, before) {
  const pct = 100 * (1 - after / before), better = pct >= 0;
  return `<div class="card"><div class="v" style="color:var(${better ? "--good" : "--orange"})">${Math.abs(pct).toFixed(0)}% ${better ? "lower" : "higher"}</div>
    <div class="l">${better ? "Error reduction on this day" : "The AI made it worse on this day"}</div></div>`;
}

function fillRealStats(day, res) {
  const added = rmse(res.view.adt, day.lr.adt) * 100;
  $("stats").innerHTML = `
    <div class="card"><div class="v">${added.toFixed(1)} cm</div><div class="l">Fine-scale detail added (RMS)</div></div>
    <div class="card"><div class="v">${(maxOf(res.view.speed)).toFixed(2)} m/s</div><div class="l">Fastest current found</div></div>`;
}

const maxOf = (a) => a.reduce((m, x) => (Number.isFinite(x) && x > m ? x : m), 0);

// ------------------------------------------------------------------ update loop

async function update() {
  const token = ++renderToken, t = state.day;
  $("date").textContent = manifest.dates[t];
  $("slider").value = t;
  const day = await loadDay(t);
  if (token !== renderToken) return;
  const cached = aiCache.get(aiKey());
  aiView = cached ? cached.view : aiView;
  drawPanels();
  if (!session) {
    if (engineError) {
      aiView = derive(day.raw.adt_ref, grid);
      drawPanels();
      setStatus("The AI engine could not start in this browser, so the AI panel shows the precomputed result.", true);
    }
    return;
  }
  const busy = $("busy-sr");
  if (!cached) busy.classList.add("on");
  try {
    const res = await runAI(t, day, (f) => { busy.textContent = `AI running… ${Math.round(f * 100)}%`; });
    if (token !== renderToken) return;
    aiView = res.view;
    drawPanels();
    updateStats(day, res);
    const verify = res.diffMm === null ? "" : ` · matches the Python pipeline to ${res.diffMm < 0.01 ? "<0.01" : res.diffMm.toFixed(2)} mm`;
    setStatus(`AI ran in your browser in ${(res.ms / 1000).toFixed(1)} s on ${res.tiles} tiles${verify}.`);
    document.body.dataset.verify = JSON.stringify({ day: t, ms: Math.round(res.ms), tiles: res.tiles, diffMm: res.diffMm, threads: ort.env.wasm.numThreads, isolated: self.crossOriginIsolated });
  } catch (err) {
    console.error(err);
    setStatus(`AI run failed: ${err.message}`, true);
  } finally {
    busy.classList.remove("on");
    busy.textContent = "AI running…";
  }
}

function setStatus(text, isErr = false) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("err", isErr);
}

async function play() {
  state.playing = !state.playing;
  $("play").textContent = state.playing ? "❚❚ Pause" : "▶ Play";
  $("play").setAttribute("aria-label", state.playing ? "Pause" : "Play through the days");
  while (state.playing) {
    const start = performance.now();
    state.day = (state.day + 1) % manifest.dates.length;
    await update();
    await new Promise((r) => setTimeout(r, Math.max(0, 700 - (performance.now() - start))));
  }
}

// ------------------------------------------------------------------ wiring

function wire() {
  const n = manifest.dates.length;
  $("slider").max = n - 1;
  $("slider").addEventListener("input", (e) => { state.day = +e.target.value; update(); });
  $("prev").onclick = () => { state.day = (state.day - 1 + n) % n; update(); };
  $("next").onclick = () => { state.day = (state.day + 1) % n; update(); };
  $("play").onclick = play;
  document.querySelectorAll("#layers button").forEach((b) => {
    b.onclick = () => {
      state.layer = b.dataset.layer;
      document.querySelectorAll("#layers button").forEach((x) => x.setAttribute("aria-pressed", x === b));
      drawPanels();
    };
  });
  $("arrows").onchange = (e) => { state.arrows = e.target.checked; drawPanels(); };
  $("noise").oninput = (e) => { state.noiseCm = +e.target.value; $("noise-v").textContent = `${state.noiseCm.toFixed(1)} cm`; };
  $("noise").onchange = () => update();
  $("hide-sst").onchange = (e) => { state.hideSst = e.target.checked; update(); };
  for (const p of ["lr", "sr", "third"]) {
    const stage = $(`s-${p}`);
    stage.addEventListener("pointermove", (e) => {
      const r = stage.getBoundingClientRect();
      const j = Math.min(grid.nx - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * grid.nx)));
      const i = grid.ny - 1 - Math.min(grid.ny - 1, Math.max(0, Math.floor(((e.clientY - r.top) / r.height) * grid.ny)));
      state.hover = { i, j };
      for (const q of panels()) paintOverlay($(`o-${q.id}`), q.view() && !q.view().isDiff ? q.view() : null);
      updateReadouts();
    });
    stage.addEventListener("pointerleave", () => { state.hover = null; drawPanels(); });
  }
  new ResizeObserver(() => drawPanels()).observe($("s-sr"));
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => drawPanels());
}

/** Optional deep-link state: ?day=5&layer=ro&noise=2&nosst=1&arrows=0 */
function readUrlState() {
  const q = new URLSearchParams(location.search), n = manifest.dates.length;
  if (q.has("day")) state.day = Math.max(0, Math.min(n - 1, parseInt(q.get("day"), 10) || 0));
  if (LAYERS[q.get("layer")]) state.layer = q.get("layer");
  if (q.has("noise")) state.noiseCm = Math.max(0, Math.min(3, parseFloat(q.get("noise")) || 0));
  state.hideSst = q.get("nosst") === "1";
  if (q.get("arrows") === "0") state.arrows = false;
  document.querySelectorAll("#layers button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.layer === state.layer));
  $("noise").value = state.noiseCm;
  $("noise-v").textContent = `${state.noiseCm.toFixed(1)} cm`;
  $("hide-sst").checked = state.hideSst;
  $("arrows").checked = state.arrows;
}

function setupPanels() {
  const { nx, ny, lat, lon } = grid;
  const midLat = (lat[0] + lat[ny - 1]) / 2;
  const aspect = (nx * Math.abs(lon[1] - lon[0]) * Math.cos((midLat * Math.PI) / 180)) / (ny * Math.abs(lat[1] - lat[0]));
  document.querySelectorAll(".stage").forEach((s) => (s.style.aspectRatio = aspect.toFixed(3)));
  const real = manifest.label === "real";
  $("badge").textContent = real ? `Real satellite data · ${manifest.region}` : "Synthetic demo data";
  $("badge").classList.toggle("real", real);
  if (!manifest.has_truth) {
    $("t-third").textContent = "Detail added by the AI";
    $("d-third").textContent = "AI minus input (red = higher, blue = lower)";
  }
}

async function main() {
  try {
    [manifest, cmaps] = await Promise.all([
      fetch(`${DATA}manifest.json`).then((r) => r.json()),
      fetch(`${DATA}colormaps.json`).then((r) => r.json()),
    ]);
  } catch (err) {
    setStatus("Could not load the data files.", true);
    throw err;
  }
  grid = makeGrid(manifest.grid.lat, manifest.grid.lon);
  setupPanels();
  wire();
  readUrlState();
  const first = await loadDay(0);
  mask = Uint8Array.from(first.raw.adt_lr, (x) => (Number.isFinite(x) ? 1 : 0));
  setStatus("Loading the AI engine…");
  const engine = loadEngine();
  await update();
  await engine;
  await update();
}

main();
