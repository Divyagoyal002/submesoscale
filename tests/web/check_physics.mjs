// Node check: JS geostrophy (site/assets/physics.js) vs Python reference values.
// Usage: node tests/web/check_physics.mjs <reference.json>
import { readFileSync } from "node:fs";
import { makeGrid, derive, maxAbsDiff } from "../../site/assets/physics.js";

const ref = JSON.parse(readFileSync(process.argv[2], "utf8"));
const g = makeGrid(ref.lat, ref.lon);
const nan = (a) => Float32Array.from(a, (x) => (x === null ? NaN : x));
const d = derive(nan(ref.adt), g);
const rel = (name) => maxAbsDiff(d[name], nan(ref[name])) / Math.max(...ref[name].filter((x) => x !== null).map(Math.abs));
const out = { u: rel("u"), v: rel("v"), ro: rel("ro") };
console.log(JSON.stringify(out));
process.exit(Object.values(out).every((e) => e < 1e-4) ? 0 : 1);
