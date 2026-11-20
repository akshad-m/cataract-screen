// Cataract screening web app - image processing.
// A JavaScript port of eye_quality_gate_demo.py: eye crop, quality gate, multi-frame glare
// separation and classifier input. No browser APIs here, so the same file runs in Node for
// the parity tests (web/test/test_vision.mjs).
//
// Images are {width, height, data} with data a Uint8ClampedArray in RGBA order (ImageData),
// or {width, height, channels, data} with tightly packed RGB / gray for internal buffers.

// ---- Settings, kept identical to eye_quality_gate_demo.py ----
export const BLUR_THRESHOLD = 60.0;
export const BRIGHTNESS_MIN = 40;
export const BRIGHTNESS_MAX = 220;
export const BURST_FRAMES = 8;
export const BURST_TIMEOUT_FRAMES = 150;
export const EYE_CROP_SIZE = 180;
export const GLARE_BRIGHT_THRESH = 235;
export const GLARE_STD_THRESH = 18.0;
export const OPACITY_BRIGHT_THRESH = 150;
export const PUPIL_MASK_SCALE = 0.8;
export const ALIGN_HIGHLIGHT_THRESH = 200;
export const MAX_ALIGN_SHIFT_FRACTION = 0.08;
export const CLASSIFIER_INPUT_SIZE = 224;
export const CLASSIFIER_CROP_HALF_IRIS = 2.2;
export const LEFT_IRIS = [474, 475, 476, 477];
export const RIGHT_IRIS = [469, 470, 471, 472];

// ---- Small image helpers ----

/** RGBA (ImageData) or RGB buffer -> packed RGB buffer. */
export function toRGB(img) {
  if (img.channels === 3) return img;
  const n = img.width * img.height;
  const out = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    out[3 * i] = img.data[4 * i];
    out[3 * i + 1] = img.data[4 * i + 1];
    out[3 * i + 2] = img.data[4 * i + 2];
  }
  return { width: img.width, height: img.height, channels: 3, data: out };
}

/** Gray exactly as OpenCV's RGB->GRAY (fixed-point 0.299 R + 0.587 G + 0.114 B). */
export function toGray(rgb) {
  const n = rgb.width * rgb.height;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const r = rgb.data[3 * i], g = rgb.data[3 * i + 1], b = rgb.data[3 * i + 2];
    out[i] = (r * 4899 + g * 9617 + b * 1868 + 8192) >> 14;
  }
  return { width: rgb.width, height: rgb.height, channels: 1, data: out };
}

/** Cut a rectangle (integer bounds, already inside the image) out of an RGB buffer. */
export function cropRGB(rgb, x1, y1, x2, y2) {
  const w = x2 - x1, h = y2 - y1;
  const out = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    const src = ((y1 + y) * rgb.width + x1) * 3;
    out.set(rgb.data.subarray(src, src + w * 3), y * w * 3);
  }
  return { width: w, height: h, channels: 3, data: out };
}

/** Resize like cv2.INTER_AREA when shrinking (exact pixel-area weights) and INTER_LINEAR when enlarging. */
export function resize(img, outW, outH, mode = "area") {
  const c = img.channels;
  const sx = img.width / outW, sy = img.height / outH;
  if (mode === "area" && sx >= 1 && sy >= 1) return resizeArea(img, outW, outH);
  const out = new Uint8Array(outW * outH * c);
  for (let y = 0; y < outH; y++) {
    let fy = (y + 0.5) * sy - 0.5;
    let y0 = Math.floor(fy);
    let wy = fy - y0;
    if (y0 < 0) { y0 = 0; wy = 0; }
    const y1 = Math.min(y0 + 1, img.height - 1);
    if (y0 >= img.height - 1) { y0 = img.height - 1; wy = 0; }
    for (let x = 0; x < outW; x++) {
      let fx = (x + 0.5) * sx - 0.5;
      let x0 = Math.floor(fx);
      let wx = fx - x0;
      if (x0 < 0) { x0 = 0; wx = 0; }
      const x1 = Math.min(x0 + 1, img.width - 1);
      if (x0 >= img.width - 1) { x0 = img.width - 1; wx = 0; }
      for (let k = 0; k < c; k++) {
        const a = img.data[(y0 * img.width + x0) * c + k], b = img.data[(y0 * img.width + x1) * c + k];
        const d = img.data[(y1 * img.width + x0) * c + k], e = img.data[(y1 * img.width + x1) * c + k];
        const v = (a * (1 - wx) + b * wx) * (1 - wy) + (d * (1 - wx) + e * wx) * wy;
        out[(y * outW + x) * c + k] = Math.round(v);
      }
    }
  }
  return { width: outW, height: outH, channels: c, data: out };
}

function areaWeights(inSize, outSize) {
  // For each output index: list of [inputIndex, weight], weights summing to 1.
  const scale = inSize / outSize;
  const table = [];
  for (let o = 0; o < outSize; o++) {
    const start = o * scale, end = start + scale;
    const cells = [];
    for (let i = Math.floor(start); i < Math.min(Math.ceil(end), inSize); i++) {
      const w = Math.min(end, i + 1) - Math.max(start, i);
      if (w > 1e-9) cells.push([i, w / scale]);
    }
    table.push(cells);
  }
  return table;
}

function resizeArea(img, outW, outH) {
  const c = img.channels;
  const wx = areaWeights(img.width, outW), wy = areaWeights(img.height, outH);
  const out = new Uint8Array(outW * outH * c);
  const acc = new Float64Array(c);
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      acc.fill(0);
      for (const [iy, ay] of wy[y]) {
        for (const [ix, ax] of wx[x]) {
          const base = (iy * img.width + ix) * c, w = ay * ax;
          for (let k = 0; k < c; k++) acc[k] += img.data[base + k] * w;
        }
      }
      for (let k = 0; k < c; k++) out[(y * outW + x) * c + k] = Math.round(acc[k]);
    }
  }
  return { width: outW, height: outH, channels: c, data: out };
}

// ---- Eye localisation (get_eye_crop) ----

/** Square box around one iris from normalised Face Mesh landmarks, as in get_eye_crop(). */
export function eyeBox(landmarks, irisIds, w, h, pad = 1.8) {
  const pts = irisIds.map((i) => [Math.trunc(landmarks[i].x * w), Math.trunc(landmarks[i].y * h)]);
  const cx = Math.trunc(pts.reduce((s, p) => s + p[0], 0) / pts.length);
  const cy = Math.trunc(pts.reduce((s, p) => s + p[1], 0) / pts.length);
  const irisRadius = Math.max(...pts.map((p) => Math.hypot(p[0] - cx, p[1] - cy)));
  const radius = Math.trunc(irisRadius * pad) + 15;
  const x1 = Math.max(cx - radius, 0), y1 = Math.max(cy - radius, 0);
  const x2 = Math.min(cx + radius, w), y2 = Math.min(cy + radius, h);
  return { x1, y1, x2, y2, irisRadius, cropRadius: radius, valid: x2 > x1 && y2 > y1 };
}

// ---- Close-up mode (one eye, aligned to an on-screen ring) ----
// The phone photos the model learned from (Sardjito) are close-ups of a single eye, cropped
// around the annotated iris box with half-width 1.1 x box size. Face Mesh cannot be used on
// such close-ups (it invents a face), so in close-up mode the user centres the iris in a ring
// and the crop is taken around the ring, framed like the training crops.
export const GUIDE_IRIS_FRACTION = 0.12;     // ring radius as a share of the frame's shorter side
export const CLOSEUP_CROP_HALF_IRIS = 2.4;   // crop half-width / ring radius (~ the Sardjito framing)

/** Ring and crop square for a w x h frame. */
export function guideBox(w, h) {
  const r = GUIDE_IRIS_FRACTION * Math.min(w, h);
  const half = Math.min(Math.trunc(CLOSEUP_CROP_HALF_IRIS * r), Math.floor(Math.min(w, h) / 2));
  const cx = Math.floor(w / 2), cy = Math.floor(h / 2);
  return { cx, cy, ringRadius: r, x1: cx - half, y1: cy - half, x2: cx + half, y2: cy + half,
           irisRadius: r, cropRadius: half, valid: true };
}

// ---- Face sanity check ----
// Face Mesh happily "finds a face" in a one-eye close-up and returns iris points in the wrong
// place, which gives a meaningless crop (seen on the Sardjito close-ups). As in crop_eyes.py, a
// real face has its irises about 3.5-8 iris-widths apart. A very small iris (face too far away)
// gives a crop too small to show anything.
// The landmark model also invents a whole face around a single-eye close-up, with plausible
// iris spacing; such a face mostly lies outside the picture, so most face points must be inside.
export const FACE_IPD_RATIO = [3.5, 8.0];
export const MIN_IRIS_RADIUS_PX = 10;
export const MIN_FACE_POINTS_INSIDE = 0.85;

export function faceCheck(landmarks, w, h) {
  const facePoints = landmarks.slice(0, 468);   // 468 face points; 468-477 are the irises
  const inside = facePoints.filter((p) => p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1).length / facePoints.length;
  const irises = [LEFT_IRIS, RIGHT_IRIS].map((ids) => {
    const pts = ids.map((i) => [landmarks[i].x * w, landmarks[i].y * h]);
    const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
    const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    return { cx, cy, r: Math.max(...pts.map((p) => Math.hypot(p[0] - cx, p[1] - cy))) };
  });
  const meanR = (irises[0].r + irises[1].r) / 2;
  const ratio = Math.hypot(irises[0].cx - irises[1].cx, irises[0].cy - irises[1].cy) / (2 * meanR || 1);
  const measured = { spacingRatio: ratio, irisRadius: meanR, faceInside: inside };
  if (inside < MIN_FACE_POINTS_INSIDE || !(ratio >= FACE_IPD_RATIO[0] && ratio <= FACE_IPD_RATIO[1])) {
    return { ok: false, reason: "show the whole face (eyes not found reliably)", ...measured };
  }
  if (meanR < MIN_IRIS_RADIUS_PX) return { ok: false, reason: "too far away: move closer", ...measured };
  return { ok: true, reason: "", ...measured };
}

// ---- Quality gate (quality_check) ----

export function qualityCheck(rgb) {
  const g = toGray(rgb);
  const { width: w, height: h, data } = g;
  const at = (x, y) => {
    // BORDER_REFLECT_101, as cv2.Laplacian uses
    if (x < 0) x = -x; else if (x >= w) x = 2 * w - 2 - x;
    if (y < 0) y = -y; else if (y >= h) y = 2 * h - 2 - y;
    return data[y * w + x];
  };
  let sum = 0, sumSq = 0, bright = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4 * data[y * w + x];
      sum += v; sumSq += v * v; bright += data[y * w + x];
    }
  }
  const n = w * h;
  const blur = sumSq / n - (sum / n) ** 2;
  const brightness = bright / n;
  let reason = "OK";
  if (blur < BLUR_THRESHOLD) reason = "TOO BLURRY";
  else if (brightness < BRIGHTNESS_MIN) reason = "TOO DARK";
  else if (brightness > BRIGHTNESS_MAX) reason = "OVEREXPOSED / GLARE";
  return { passed: reason === "OK", blur, brightness, reason };
}

// ---- Burst collector (EyeBurstCollector) ----

export class EyeBurstCollector {
  constructor(nFrames = BURST_FRAMES, timeoutFrames = BURST_TIMEOUT_FRAMES) {
    this.nFrames = nFrames;
    this.timeoutFrames = timeoutFrames;
    this.active = false;
    this.pending = false;
    this.reset();
  }
  reset() { this.framesWaited = 0; this.crops = []; this.irisRadii = []; this.cropRadii = []; }
  start() { this.active = true; this.pending = true; this.reset(); }
  cancel() { this.active = false; this.pending = false; }
  offer(crop, irisRadius, cropRadius, qualityPassed) {
    if (!this.active) return;
    this.framesWaited += 1;
    if (qualityPassed && crop && crop.width > 0 && crop.height > 0) {
      const ratio = crop.width / crop.height;
      if (ratio >= 0.85 && ratio <= 1.15) {
        this.crops.push(resize(crop, EYE_CROP_SIZE, EYE_CROP_SIZE, "area"));
        this.irisRadii.push(irisRadius);
        this.cropRadii.push(cropRadius);
      }
    }
    if (this.crops.length >= this.nFrames || this.framesWaited >= this.timeoutFrames) this.active = false;
  }
  ready() { return this.pending && !this.active && this.crops.length >= this.nFrames; }
  timedOut() { return this.pending && !this.active && this.crops.length < this.nFrames; }
}

// ---- Alignment (phase correlation, as cv2.phaseCorrelate with a Hanning window) ----

// Radix-2 FFT. Frames are zero-padded to the next power of two (180 -> 256); after the Hanning
// window the frame edges are ~0, so this changes the correlation peak only marginally while
// running ~5x faster than a 180-point mixed-radix transform (matters on phones).
function fft1d(re, im, inverse) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (2 * Math.PI / len) * (inverse ? 1 : -1);
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

function fft2d(re, im, n, inverse) {
  const rr = new Float64Array(n), ri = new Float64Array(n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) { rr[x] = re[y * n + x]; ri[x] = im[y * n + x]; }
    fft1d(rr, ri, inverse);
    for (let x = 0; x < n; x++) { re[y * n + x] = rr[x]; im[y * n + x] = ri[x]; }
  }
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) { rr[y] = re[y * n + x]; ri[y] = im[y * n + x]; }
    fft1d(rr, ri, inverse);
    for (let y = 0; y < n; y++) { re[y * n + x] = rr[y]; im[y * n + x] = ri[y]; }
  }
}

function hanning(w, h) {
  // cv2.createHanningWindow
  const wx = new Float64Array(w), wy = new Float64Array(h);
  for (let i = 0; i < w; i++) wx[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (w - 1)));
  for (let i = 0; i < h; i++) wy[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (h - 1)));
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = wy[y] * wx[x];
  return out;
}

/** Copy of a gray frame with near-white highlights painted over (for alignment only). */
export function highlightsRemoved(gray) {
  const { width: w, height: h, data } = gray;
  const mask = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (data[y * w + x] < ALIGN_HIGHLIGHT_THRESH) continue;
      for (let dy = -2; dy <= 2; dy++) {           // 5x5 dilation
        for (let dx = -2; dx <= 2; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx >= 0 && xx < w && yy >= 0 && yy < h) mask[yy * w + xx] = 1;
        }
      }
    }
  }
  const out = Float64Array.from(data);
  // fill the masked area from its edge inwards with the mean of already-known neighbours
  let remaining = mask.reduce((s, v) => s + v, 0);
  while (remaining > 0) {
    const fill = [];
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (!mask[y * w + x]) continue;
        let s = 0, n = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx, yy = y + dy;
            if (xx >= 0 && xx < w && yy >= 0 && yy < h && !mask[yy * w + xx]) { s += out[yy * w + xx]; n++; }
          }
        }
        if (n) fill.push([y * w + x, s / n]);
      }
    }
    if (!fill.length) break;                        // whole image masked: leave as is
    for (const [i, v] of fill) { out[i] = v; mask[i] = 0; }
    remaining -= fill.length;
  }
  return out;
}

/** Shift (dx, dy) of frame `b` relative to frame `a` (both Float64Array w*h), like cv2.phaseCorrelate(a, b, window). */
export function phaseCorrelate(a, b, w, h, window) {
  let n = 1;
  while (n < Math.max(w, h)) n <<= 1;
  const ar = new Float64Array(n * n), ai = new Float64Array(n * n);
  const br = new Float64Array(n * n), bi = new Float64Array(n * n);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      ar[y * n + x] = a[y * w + x] * window[y * w + x];
      br[y * n + x] = b[y * w + x] * window[y * w + x];
    }
  }
  fft2d(ar, ai, n, false);
  fft2d(br, bi, n, false);
  // normalised cross-power spectrum  B * conj(A) / |B * conj(A)|
  for (let i = 0; i < n * n; i++) {
    const re = br[i] * ar[i] + bi[i] * ai[i];
    const im = bi[i] * ar[i] - br[i] * ai[i];
    const mag = Math.hypot(re, im) || 1e-12;
    ar[i] = re / mag; ai[i] = im / mag;
  }
  fft2d(ar, ai, n, true);
  let best = -Infinity, px = 0, py = 0;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) if (ar[y * n + x] > best) { best = ar[y * n + x]; px = x; py = y; }
  // sub-pixel: weighted centroid of the 5x5 neighbourhood (as OpenCV does)
  let sw = 0, sx = 0, sy = 0;
  for (let dy = -2; dy <= 2; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      const yy = (py + dy + n) % n, xx = (px + dx + n) % n;
      const v = ar[yy * n + xx];
      sw += v; sx += v * (px + dx); sy += v * (py + dy);
    }
  }
  let dx = sw ? sx / sw : px, dy = sw ? sy / sw : py;
  if (dx > n / 2) dx -= n;
  if (dy > n / 2) dy -= n;
  return [dx, dy];
}

/** Align gray frames to the first one; returns the (dx, dy) shift of each frame. */
export function alignmentShifts(grays) {
  const { width: w, height: h } = grays[0];
  const window = hanning(w, h);
  const ref = highlightsRemoved(grays[0]);
  const maxShift = MAX_ALIGN_SHIFT_FRACTION * Math.min(w, h);
  const shifts = [[0, 0]];
  for (const g of grays.slice(1)) {
    let [dx, dy] = phaseCorrelate(ref, highlightsRemoved(g), w, h, window);
    if (Math.hypot(dx, dy) > maxShift) { dx = 0; dy = 0; }
    shifts.push([dx, dy]);
  }
  return shifts;
}

/** dst(x, y) = src(x + dx, y + dy), bilinear, edge pixels repeated (cv2.warpAffine, BORDER_REPLICATE). */
export function translate(img, dx, dy) {
  const { width: w, height: h, channels: c } = img;
  if (dx === 0 && dy === 0) return img;
  const out = new Uint8Array(w * h * c);
  const clampX = (v) => (v < 0 ? 0 : v > w - 1 ? w - 1 : v);
  const clampY = (v) => (v < 0 ? 0 : v > h - 1 ? h - 1 : v);
  for (let y = 0; y < h; y++) {
    const fy = y + dy, y0 = Math.floor(fy), wy = fy - y0;
    const ya = clampY(y0), yb = clampY(y0 + 1);
    for (let x = 0; x < w; x++) {
      const fx = x + dx, x0 = Math.floor(fx), wx = fx - x0;
      const xa = clampX(x0), xb = clampX(x0 + 1);
      for (let k = 0; k < c; k++) {
        const v = (img.data[(ya * w + xa) * c + k] * (1 - wx) + img.data[(ya * w + xb) * c + k] * wx) * (1 - wy) +
                  (img.data[(yb * w + xa) * c + k] * (1 - wx) + img.data[(yb * w + xb) * c + k] * wx) * wy;
        out[(y * w + x) * c + k] = Math.round(v);
      }
    }
  }
  return { width: w, height: h, channels: c, data: out };
}

// ---- Glare separation (glare_separation) ----

function median(values) {
  const s = values.slice().sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/**
 * crops: equal-size RGB buffers (the burst); irisRadii / cropRadii as recorded per frame.
 * Returns the de-glared composite, the moving-glare and stable-opacity masks, the pupil mask,
 * the opacity fraction and the alignment shifts.
 */
export function glareSeparation(crops, irisRadii, cropRadii) {
  const { width: w, height: h } = crops[0];
  const n = crops.length;
  const grays = crops.map(toGray);
  const shifts = alignmentShifts(grays);
  const alignedGray = grays.map((g, i) => translate(g, shifts[i][0], shifts[i][1]));
  const alignedColor = crops.map((c, i) => translate(c, shifts[i][0], shifts[i][1]));

  const medianGray = new Float64Array(w * h);
  const stdGray = new Float64Array(w * h);
  const medianColor = new Uint8Array(w * h * 3);
  const glareMask = new Uint8Array(w * h);
  const vals = new Array(n);
  for (let p = 0; p < w * h; p++) {
    let s = 0, everBright = false;
    for (let i = 0; i < n; i++) {
      const v = alignedGray[i].data[p];
      vals[i] = v; s += v;
      if (v > GLARE_BRIGHT_THRESH) everBright = true;
    }
    const mean = s / n;
    let ss = 0;
    for (let i = 0; i < n; i++) ss += (vals[i] - mean) ** 2;
    stdGray[p] = Math.sqrt(ss / n);
    medianGray[p] = median(vals);
    if (everBright && stdGray[p] > GLARE_STD_THRESH) glareMask[p] = 255;
    for (let k = 0; k < 3; k++) {
      for (let i = 0; i < n; i++) vals[i] = alignedColor[i].data[3 * p + k];
      medianColor[3 * p + k] = Math.floor(median(vals));
    }
  }

  const ratios = irisRadii.map((r, i) => (cropRadii[i] > 0 ? r / cropRadii[i] : null)).filter((r) => r !== null);
  const avgRatio = ratios.length ? ratios.reduce((a, b) => a + b, 0) / ratios.length : 0.3;
  const pupilR = Math.max(Math.trunc(avgRatio * PUPIL_MASK_SCALE * (w / 2)), 5);
  const cx = Math.trunc(w / 2), cy = Math.trunc(h / 2);
  const pupilMask = new Uint8Array(w * h);
  const opacityMask = new Uint8Array(w * h);
  let inPupil = 0, stable = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = y * w + x;
      if ((x - cx) ** 2 + (y - cy) ** 2 > pupilR * pupilR) continue;
      pupilMask[p] = 255; inPupil++;
      if (medianGray[p] > OPACITY_BRIGHT_THRESH && stdGray[p] < GLARE_STD_THRESH) { opacityMask[p] = 255; stable++; }
    }
  }
  return {
    medianColor: { width: w, height: h, channels: 3, data: medianColor },
    glareMask: { width: w, height: h, channels: 1, data: glareMask },
    opacityMask: { width: w, height: h, channels: 1, data: opacityMask },
    pupilMask: { width: w, height: h, channels: 1, data: pupilMask },
    opacityFraction: inPupil ? stable / inPupil : 0,
    shifts,
  };
}

// ---- Classifier input (classify_eye) ----

/** 224x224 RGB float32 (0-255, NHWC) for the model, cut from the de-glared composite. */
export function classifierInput(result, irisRadii, cropRadii) {
  const comp = result.medianColor;
  const size = comp.width;
  const ratios = irisRadii.map((r, i) => (cropRadii[i] > 0 ? r / cropRadii[i] : null)).filter((r) => r !== null);
  const irisPx = (ratios.length ? ratios.reduce((a, b) => a + b, 0) / ratios.length : 0.3) * size / 2;
  const half = Math.trunc(Math.min(Math.max(irisPx * CLASSIFIER_CROP_HALF_IRIS, 8), Math.floor(size / 2)));
  const c = Math.floor(size / 2);
  const crop = cropRGB(comp, c - half, c - half, c + half, c + half);
  const resized = resize(crop, CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE, "linear");
  return Float32Array.from(resized.data);
}

/** NORMAL / REFER / URGENT from the two model scores and the thresholds in model/config.json. */
export function decide(cataractScore, matureScore, config) {
  if (cataractScore < config.screen_threshold) return "normal";
  return matureScore >= config.severity_threshold ? "urgent" : "refer";
}
