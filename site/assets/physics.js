// Browser port of submeso.physics and submeso.inference.super_resolve_snapshot.
// Kept numerically identical to the Python code (same finite differences, tiling,
// per-tile normalisation and taper) so the live result can be checked against it.
// Fields are Float32Array in row-major (lat, lon) order, row 0 = southernmost.

export const G = 9.81;
export const OMEGA = 7.2921159e-5;
export const R_EARTH = 6371000.0;
const MIN_ABS_LAT = 2.0;
const DEG = Math.PI / 180;

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const diffs = (a) => a.slice(1).map((v, i) => v - a[i]);

/** Grid geometry shared by all derivative operators. */
export function makeGrid(lat, lon) {
  const ny = lat.length, nx = lon.length;
  const dlat = mean(diffs(lat)) * DEG, dlon = mean(diffs(lon)) * DEG;
  const dx = Float64Array.from(lat, (l) => R_EARTH * Math.cos(l * DEG) * dlon);
  const f = Float64Array.from(lat, (l) => (Math.abs(l) < MIN_ABS_LAT ? NaN : 2 * OMEGA * Math.sin(l * DEG)));
  return { ny, nx, lat, lon, dx, dy: R_EARTH * dlat, f };
}

// np.gradient semantics: central differences inside, one-sided at the edges.
function gradX(a, g) {
  const { ny, nx } = g, out = new Float32Array(ny * nx);
  for (let i = 0; i < ny; i++) {
    const r = i * nx;
    out[r] = a[r + 1] - a[r];
    out[r + nx - 1] = a[r + nx - 1] - a[r + nx - 2];
    for (let j = 1; j < nx - 1; j++) out[r + j] = (a[r + j + 1] - a[r + j - 1]) / 2;
  }
  return out;
}

function gradY(a, g) {
  const { ny, nx } = g, out = new Float32Array(ny * nx);
  for (let j = 0; j < nx; j++) {
    out[j] = a[nx + j] - a[j];
    out[(ny - 1) * nx + j] = a[(ny - 1) * nx + j] - a[(ny - 2) * nx + j];
    for (let i = 1; i < ny - 1; i++) out[i * nx + j] = (a[(i + 1) * nx + j] - a[(i - 1) * nx + j]) / 2;
  }
  return out;
}

/** Geostrophic velocity u = -(g/f) d(eta)/dy, v = (g/f) d(eta)/dx  [m/s]. */
export function geostrophic(adt, g) {
  const ex = gradX(adt, g), ey = gradY(adt, g);
  const u = new Float32Array(adt.length), v = new Float32Array(adt.length);
  for (let i = 0; i < g.ny; i++) {
    const k = G / g.f[i];
    for (let j = 0; j < g.nx; j++) {
      const p = i * g.nx + j;
      u[p] = -k * ey[p] / g.dy;
      v[p] = k * ex[p] / g.dx[i];
    }
  }
  return { u, v };
}

/** Rossby number zeta / f, with zeta = dv/dx - du/dy. */
export function rossby(u, v, g) {
  const vx = gradX(v, g), uy = gradY(u, g), out = new Float32Array(u.length);
  for (let i = 0; i < g.ny; i++) {
    for (let j = 0; j < g.nx; j++) {
      const p = i * g.nx + j;
      out[p] = (vx[p] / g.dx[i] - uy[p] / g.dy) / g.f[i];
    }
  }
  return out;
}

export function speed(u, v) {
  return Float32Array.from(u, (x, p) => Math.hypot(x, v[p]));
}

/** Everything the viewer shows for one ADT field. */
export function derive(adt, g) {
  const { u, v } = geostrophic(adt, g);
  return { adt, u, v, speed: speed(u, v), ro: rossby(u, v, g) };
}

// ------------------------------------------------------------------ inference

function starts(n, tile, step) {
  if (n <= tile) return [0];
  const s = [];
  for (let x = 0; x <= n - tile; x += step) s.push(x);
  if (s[s.length - 1] !== n - tile) s.push(n - tile);
  return s;
}

function taper(ty, tx) {
  // np.hanning(n + 2)[1:-1] outer product, + 1e-3 (as in inference._taper)
  const h = (n) => Float64Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 1)) / (n + 1)));
  const wy = h(ty), wx = h(tx), w = new Float64Array(ty * tx);
  for (let i = 0; i < ty; i++) for (let j = 0; j < tx; j++) w[i * tx + j] = wy[i] * wx[j] + 1e-3;
  return w;
}

/** Build the (3, ty, tx) network input for one tile (dataset.normalize_inputs). */
function tileInput(adt, sst, mask, g, y, x, ty, tx, stats, hideSst) {
  const n = ty * tx, input = new Float32Array(3 * n);
  let sa = 0, ss = 0, count = 0;
  const ocean = new Uint8Array(n);
  for (let i = 0; i < ty; i++) {
    for (let j = 0; j < tx; j++) {
      const p = (y + i) * g.nx + x + j;
      if (mask[p] && Number.isFinite(adt[p]) && Number.isFinite(sst[p])) {
        ocean[i * tx + j] = 1;
        sa += adt[p]; ss += sst[p]; count++;
      }
    }
  }
  if (!count) return null;
  const aOff = sa / count, sOff = ss / count;
  for (let i = 0; i < ty; i++) {
    for (let j = 0; j < tx; j++) {
      const q = i * tx + j, p = (y + i) * g.nx + x + j;
      if (!ocean[q]) continue;
      input[q] = (adt[p] - aOff) / stats.adt_scale;
      input[n + q] = hideSst ? 0 : (sst[p] - sOff) / stats.sst_scale;
      input[2 * n + q] = 1;
    }
  }
  return input;
}

/**
 * Super-resolve one snapshot with an onnxruntime-web session.
 * Returns { adt: Float32Array, tiles } with NaN over land.
 */
export async function superResolve(ort, session, { adt, sst, mask, g, stats, tile, overlap, hideSst = false, batch = 8, onProgress }) {
  const ty = Math.min(tile, g.ny), tx = Math.min(tile, g.nx);
  const w = taper(ty, tx);
  const tiles = [];
  for (const y of starts(g.ny, ty, Math.max(1, ty - overlap))) {
    for (const x of starts(g.nx, tx, Math.max(1, tx - overlap))) {
      const input = tileInput(adt, sst, mask, g, y, x, ty, tx, stats, hideSst);
      if (input) tiles.push({ y, x, input });
    }
  }
  const acc = new Float64Array(g.ny * g.nx), wsum = new Float64Array(g.ny * g.nx);
  const n = ty * tx;
  for (let b = 0; b < tiles.length; b += batch) {
    const chunk = tiles.slice(b, b + batch);
    const data = new Float32Array(chunk.length * 3 * n);
    chunk.forEach((t, k) => data.set(t.input, k * 3 * n));
    const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", data, [chunk.length, 3, ty, tx]) };
    const out = (await session.run(feeds))[session.outputNames[0]].data;
    chunk.forEach((t, k) => {
      for (let i = 0; i < ty; i++) {
        for (let j = 0; j < tx; j++) {
          const p = (t.y + i) * g.nx + t.x + j, q = i * tx + j;
          acc[p] += out[k * n + q] * stats.residual_scale * w[q];
          wsum[p] += w[q];
        }
      }
    });
    onProgress?.(Math.min(1, (b + batch) / tiles.length));
  }
  const result = new Float32Array(g.ny * g.nx);
  for (let p = 0; p < result.length; p++) {
    result[p] = mask[p] ? adt[p] + (wsum[p] > 0 ? acc[p] / wsum[p] : 0) : NaN;
  }
  return { adt: result, tiles: tiles.length };
}

// ------------------------------------------------------------------ experiments & metrics

/** Deterministic smooth noise (std = sigma over ocean), for the "satellite noise" experiment. */
export function smoothNoise(g, mask, sigma, seed) {
  let s = seed >>> 0 || 1;
  const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const n = g.ny * g.nx;
  let a = Float32Array.from({ length: n }, () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand()));
  const r = 3;
  for (let pass = 0; pass < 2; pass++) {
    const b = new Float32Array(n);
    for (let i = 0; i < g.ny; i++) for (let j = 0; j < g.nx; j++) {
      let sum = 0, c = 0;
      for (let dj = -r; dj <= r; dj++) { const jj = j + dj; if (jj >= 0 && jj < g.nx) { sum += a[i * g.nx + jj]; c++; } }
      b[i * g.nx + j] = sum / c;
    }
    const c2 = new Float32Array(n);
    for (let i = 0; i < g.ny; i++) for (let j = 0; j < g.nx; j++) {
      let sum = 0, c = 0;
      for (let di = -r; di <= r; di++) { const ii = i + di; if (ii >= 0 && ii < g.ny) { sum += b[ii * g.nx + j]; c++; } }
      c2[i * g.nx + j] = sum / c;
    }
    a = c2;
  }
  let ss = 0, cnt = 0;
  for (let p = 0; p < n; p++) if (mask[p]) { ss += a[p] * a[p]; cnt++; }
  const k = sigma / Math.sqrt(ss / Math.max(cnt, 1));
  return Float32Array.from(a, (x) => x * k);
}

export function rmse(a, b) {
  let s = 0, n = 0;
  for (let p = 0; p < a.length; p++) {
    const d = a[p] - b[p];
    if (Number.isFinite(d)) { s += d * d; n++; }
  }
  return n ? Math.sqrt(s / n) : NaN;
}

export function corr(a, b) {
  let n = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  for (let p = 0; p < a.length; p++) {
    const x = a[p], y = b[p];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    n++; sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
  }
  const cov = sab / n - (sa / n) * (sb / n);
  return cov / Math.sqrt((saa / n - (sa / n) ** 2) * (sbb / n - (sb / n) ** 2));
}

export function maxAbsDiff(a, b) {
  let m = 0;
  for (let p = 0; p < a.length; p++) {
    const d = Math.abs(a[p] - b[p]);
    if (Number.isFinite(d) && d > m) m = d;
  }
  return m;
}
