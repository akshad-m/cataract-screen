// Parity test: web/vision.js + web/model/cataract.onnx against the laptop demo (Python + OpenCV).
// Resizing and the quality gate match OpenCV exactly. Alignment can differ by up to ~1 px because
// highlight removal is approximated (OpenCV uses TELEA inpainting); the test therefore checks the
// end results (composite, scores, calls) and the glare-separation behaviour itself (moving dot /
// fixed haze, as test_glare_separation.py does in Python).
// Needs web/test/_out/ from make_vision_reference.py, plus `npm install` in web/test
// (onnxruntime-node and pngjs, pinned in package.json).
//   cd web/test && npm install && node test_vision.mjs

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { PNG } from "pngjs";
import * as ort from "onnxruntime-node";
import * as V from "../vision.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "../..");
const OUT = path.join(HERE, "_out");
const ref = JSON.parse(fs.readFileSync(path.join(OUT, "reference.json"), "utf8"));
const config = JSON.parse(fs.readFileSync(path.join(ROOT, "web/model/config.json"), "utf8"));

function readPNG(rel) {
  const png = PNG.sync.read(fs.readFileSync(path.join(ROOT, rel)));
  return V.toRGB({ width: png.width, height: png.height, data: png.data });
}
function readBin(name) { return new Uint8Array(fs.readFileSync(path.join(OUT, name))); }
function meanAbs(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s / a.length; }
function maxAbs(a, b) { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; }
const fmt = (x, d = 3) => Number(x).toFixed(d);

let failures = 0;
function check(ok, msg) { if (!ok) { failures++; console.log("  FAIL:", msg); } }

console.log("== resize and quality gate ==");
for (const c of ref.resize_cases) {
  const img = readPNG(c.file);
  const area = V.resize(img, 180, 180, "area");
  const linear = V.resize(img, 224, 224, "linear");
  const ea = meanAbs(area.data, readBin(`resize_${c.key}_area.bin`));
  const el = meanAbs(linear.data, readBin(`resize_${c.key}_linear.bin`));
  const q = V.qualityCheck(img);
  console.log(`  ${c.key.padEnd(28)} ${img.width}x${img.height}  area diff ${fmt(ea, 2)}  linear diff ${fmt(el, 2)}  ` +
              `blur ${fmt(q.blur, 1)} vs ${fmt(c.quality.blur, 1)}  bright ${fmt(q.brightness, 1)} vs ${fmt(c.quality.brightness, 1)}`);
  check(ea < 1.0 && el < 1.0, `${c.key}: resize differs (mean abs ${ea}, ${el})`);
  check(q.passed === c.quality.passed, `${c.key}: quality gate verdict differs`);
  check(Math.abs(q.brightness - c.quality.brightness) < 0.01, `${c.key}: brightness differs`);
  check(Math.abs(q.blur - c.quality.blur) / Math.max(c.quality.blur, 1) < 0.01, `${c.key}: blur differs`);
}

console.log("\n== bursts: alignment, glare separation, classifier ==");
const session = await ort.InferenceSession.create(path.join(ROOT, "web/model/cataract.onnx"));
let sameCalls = 0, borderline = 0, worstScore = 0;
for (const b of ref.bursts) {
  const crops = b.frames.map(readPNG);
  const ir = Array(crops.length).fill(b.ratio), cr = Array(crops.length).fill(1.0);
  const t0 = performance.now();
  const res = V.glareSeparation(crops, ir, cr);
  const ms = performance.now() - t0;
  const shiftErr = Math.max(...res.shifts.map((s, i) => Math.hypot(s[0] - b.shifts[i][0], s[1] - b.shifts[i][1])));
  const medDiff = meanAbs(res.medianColor.data, readBin(`${b.name}_median.bin`));
  const input = V.classifierInput(res, ir, cr);
  const inDiff = meanAbs(input, readBin(`${b.name}_input.bin`));
  const out = await session.run({ image: new ort.Tensor("float32", input, [1, 224, 224, 3]) });
  const cat = out.cataract_score.data[0], mat = out.mature_score.data[0];
  const call = V.decide(cat, mat, config);
  sameCalls += call === b.output;
  // tiny score differences can flip an eye that sits right on the cut-off
  if (call !== b.output && Math.abs(b.cataract_score - config.screen_threshold) < 0.05) borderline++;
  worstScore = Math.max(worstScore, Math.abs(cat - b.cataract_score));
  const glare = res.glareMask.data.reduce((s, v) => s + (v > 0), 0);
  const opac = res.opacityMask.data.reduce((s, v) => s + (v > 0), 0);
  console.log(`  ${b.name}  shift err ${fmt(shiftErr, 2)}px  composite diff ${fmt(medDiff, 2)}  input diff ${fmt(inDiff, 2)}  ` +
              `opacity ${fmt(100 * res.opacityFraction, 1)}% vs ${fmt(100 * b.opacity_fraction, 1)}%  ` +
              `glare px ${glare} vs ${b.glare_pixels}  score ${fmt(cat)} vs ${fmt(b.cataract_score)}  ${call}/${b.output}  (${ms.toFixed(0)} ms)`);
  check(shiftErr < 1.5, `${b.name}: alignment shift differs by ${shiftErr}px`);
  check(medDiff < 1.0 && inDiff < 1.5, `${b.name}: composite or classifier input differs`);
  check(Math.abs(res.opacityFraction - b.opacity_fraction) < 0.02, `${b.name}: opacity fraction differs`);
  check(Math.abs(cat - b.cataract_score) < 0.05, `${b.name}: cataract score differs`);
}
console.log(`\nsame NORMAL/REFER/URGENT call on ${sameCalls}/${ref.bursts.length} bursts; worst cataract-score difference ${fmt(worstScore, 4)}`);
check(sameCalls + borderline >= ref.bursts.length, "a call differs on an eye that is not near the cut-off");

console.log("\n== synthetic glare test (moving dot should be glare, fixed haze should be opacity) ==");
function paintDisk(img, cx, cy, r, fn) {
  for (let y = Math.max(0, cy - r); y <= Math.min(img.height - 1, cy + r); y++)
    for (let x = Math.max(0, cx - r); x <= Math.min(img.width - 1, cx + r); x++)
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) fn(y * img.width + x);
}
function share(mask, region) { let n = 0, hit = 0; region.forEach((p) => { n++; if (mask.data[p]) hit++; }); return n ? hit / n : NaN; }
const cases = { 3: [], 6: [], haze: [] };
for (const b of ref.bursts) {
  const base = b.frames.map(readPNG);
  const ir = Array(base.length).fill(b.ratio), cr = Array(base.length).fill(1.0);
  for (const step of [3, 6]) {
    const region = new Set();
    const painted = base.map((f, k) => {
      const g = { ...f, data: Uint8Array.from(f.data) };
      const cx = Math.floor(f.width / 2) - 10 + step * k, cy = Math.floor(f.height / 2) - 8 + Math.floor((step * k) / 2);
      paintDisk(g, cx, cy, 4, (p) => { g.data[3 * p] = g.data[3 * p + 1] = g.data[3 * p + 2] = 255; region.add(p); });
      return g;
    });
    const res = V.glareSeparation(painted, ir, cr);
    cases[step].push([share(res.glareMask, region), share(res.opacityMask, region)]);
  }
  const region = new Set();
  const hazy = base.map((f) => {
    const g = { ...f, data: Uint8Array.from(f.data) };
    paintDisk(g, Math.floor(f.width / 2), Math.floor(f.height / 2), 12, (p) => {
      region.add(p);
      for (let k = 0; k < 3; k++) g.data[3 * p + k] = Math.floor(0.3 * g.data[3 * p + k] + 0.7 * 225);
    });
    return g;
  });
  const res = V.glareSeparation(hazy, ir, cr);
  cases.haze.push([share(res.opacityMask, region), share(res.glareMask, region)]);
}
const avg = (rows, i) => rows.reduce((s, r) => s + r[i], 0) / rows.length;
for (const step of [3, 6]) {
  console.log(`  dot moving ${step} px/frame: caught as glare ${fmt(100 * avg(cases[step], 0), 0)}%, wrongly opacity ${fmt(100 * avg(cases[step], 1), 0)}%`);
  check(avg(cases[step], 0) > 0.6 && avg(cases[step], 1) < 0.05, `moving dot (${step} px/frame) not separated`);
}
console.log(`  fixed haze: flagged as opacity ${fmt(100 * avg(cases.haze, 0), 0)}%, wrongly glare ${fmt(100 * avg(cases.haze, 1), 0)}%`);
check(avg(cases.haze, 0) > 0.85 && avg(cases.haze, 1) < 0.05, "fixed haze not kept as opacity");
console.log(failures ? `\n${failures} check(s) FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);
