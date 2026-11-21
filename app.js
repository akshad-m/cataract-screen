// Cataract screening web app - camera, UI and model running.
// Processing (quality gate, glare separation, classifier input) lives in vision.js and is
// tested against the Python laptop demo by web/test/test_vision.mjs.
//
// Two capture modes:
//   close-up (default): one eye fills the frame, the user centres the iris in a ring. This matches
//                       the one-eye phone photos the model learned from, and gives the most detail.
//   face:               the whole face in view, both eyes found with MediaPipe Face Landmarker
//                       (as in the laptop demo).

import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";
import * as V from "./vision.js?v=7";   // bump ?v= (here and in index.html) on every release: Pages caches ~10 min

const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const FACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const BURST_WALL_TIMEOUT_MS = 10000;   // give up on a burst after this long, even if no face is seen
const MAX_PHOTO_SIDE = 2048;            // large photos are scaled down to this before analysis
const FACE_EYES = [["L", V.LEFT_IRIS, "Left eye"], ["R", V.RIGHT_IRIS, "Right eye"]];
const OUTPUT_TEXT = {
  normal: ["NORMAL", "No cataract-like signs found by the prototype."],
  refer: ["REFER", "Cataract-like appearance. A real screening app would advise an eye check-up."],
  urgent: ["URGENT", "Looks like an advanced (mature) cataract to the prototype. A real screening app would advise an early eye check-up."],
};
const COLORS = { pass: "#2fbf55", fail: "#ff4d4d", busy: "#4c9cff" };
const AUTO_HOLD_MS = 1200;              // auto-capture once the picture has stayed good this long
const ALIGNED_CUE_GAP_MS = 1500;        // don't repeat the "lined up" beep more often than this

const $ = (id) => document.getElementById(id);
const state = {
  config: null, session: null, landmarker: null, runningMode: "VIDEO", mode: "closeup",
  stream: null, running: false, collectors: null, burstStarted: 0, processing: false, lastVideoTime: -1,
  steadySince: null, lastAlignedCue: 0,
};

// ---------- settings (remembered on this device) ----------

function loadSetting(key, fallback) {
  try { const v = localStorage.getItem(key); return v === null ? fallback : v === "1"; } catch { return fallback; }
}
function saveSetting(key, value) {
  try { localStorage.setItem(key, value ? "1" : "0"); } catch { /* private mode: lasts this visit only */ }
}
const settings = { auto: loadSetting("autoCapture", true), cues: loadSetting("cues", true) };

// ---------- sound and vibration cues ----------
// Tones are generated with Web Audio (no sound files). Browsers only allow sound after a tap, so the
// audio context is unlocked from the Start-camera tap. Vibration works on Android; iPhones ignore it.

let audio = null;
function unlockAudio() {
  try {
    audio ??= new (window.AudioContext || window.webkitAudioContext)();
    audio.resume();
  } catch { audio = null; }
}

function tone(freq, start, duration) {
  const osc = audio.createOscillator(), gain = audio.createGain();
  const t = audio.currentTime + start;
  osc.type = "sine";
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.0001, t);
  gain.gain.exponentialRampToValueAtTime(0.25, t + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, t + duration);
  osc.connect(gain).connect(audio.destination);
  osc.start(t);
  osc.stop(t + duration + 0.02);
}

const CUES = {
  aligned: { tones: [[660, 0, 0.12]], vibrate: 40 },                        // picture is good
  start: { tones: [[660, 0, 0.1], [880, 0.12, 0.14]], vibrate: 60 },        // capture begins
  done: { tones: [[880, 0, 0.12], [1175, 0.15, 0.2]], vibrate: [80, 60, 80] },
  fail: { tones: [[330, 0, 0.3]], vibrate: 250 },
};
function cue(name) {
  if (!settings.cues) return;
  const c = CUES[name];
  try { if (audio) c.tones.forEach(([f, start, d]) => tone(f, start, d)); } catch { /* no audio */ }
  try { navigator.vibrate?.(c.vibrate); } catch { /* no vibration */ }
}
const grab = document.createElement("canvas");
const grabCtx = grab.getContext("2d", { willReadFrequently: true });

// ---------- setup ----------

function setStatus(id, text, isError = false) {
  const el = $(id);
  el.textContent = text;
  el.classList.toggle("error", isError);
}

function show(section) {
  for (const id of ["intro", "camera", "results"]) $(id).hidden = id !== section;
  window.scrollTo(0, 0);
}

function selected(name) {
  return document.querySelector(`input[name=${name}]:checked`).value;
}

async function createLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
  const options = (delegate) => ({ baseOptions: { modelAssetPath: FACE_MODEL, delegate }, runningMode: "VIDEO", numFaces: 1 });
  try {
    return await FaceLandmarker.createFromOptions(fileset, options("GPU"));
  } catch {
    return await FaceLandmarker.createFromOptions(fileset, options("CPU"));   // some phones lack WebGL2
  }
}

async function loadModels() {
  try {
    ort.env.wasm.wasmPaths = ORT_DIST;
    ort.env.wasm.numThreads = 1;   // threads need cross-origin isolation, which simple hosting lacks
    const [config, session, landmarker] = await Promise.all([
      fetch("model/config.json").then((r) => { if (!r.ok) throw new Error(`config.json: HTTP ${r.status}`); return r.json(); }),
      ort.InferenceSession.create("model/cataract.onnx", { executionProviders: ["wasm"] }),
      createLandmarker(),
    ]);
    Object.assign(state, { config, session, landmarker });
    $("start").disabled = false;
    $("start").textContent = "Start camera";
    setStatus("load-status", "Ready.");
  } catch (e) {
    $("start").textContent = "Models failed to load";
    setStatus("load-status", `Could not load the models (${e.message}). Check the internet connection and reload the page.`, true);
    throw e;
  }
}

async function setRunningMode(mode) {
  if (state.runningMode !== mode) {
    await state.landmarker.setOptions({ runningMode: mode });
    state.runningMode = mode;
  }
}

// ---------- image helpers ----------

function grabRegion(source, x1, y1, x2, y2) {
  const w = x2 - x1, h = y2 - y1;
  grab.width = w;
  grab.height = h;
  grabCtx.drawImage(source, x1, y1, w, h, 0, 0, w, h);
  return V.toRGB(grabCtx.getImageData(0, 0, w, h));
}

function drawRGB(canvas, img) {
  canvas.width = img.width;
  canvas.height = img.height;
  const out = new ImageData(img.width, img.height);
  const c = img.channels;
  for (let i = 0; i < img.width * img.height; i++) {
    out.data[4 * i] = img.data[c * i];
    out.data[4 * i + 1] = img.data[c === 3 ? c * i + 1 : i];
    out.data[4 * i + 2] = img.data[c === 3 ? c * i + 2 : i];
    out.data[4 * i + 3] = 255;
  }
  canvas.getContext("2d").putImageData(out, 0, 0);
}

function withMaskInRed(rgb, mask) {
  const data = Uint8Array.from(rgb.data);
  for (let p = 0; p < mask.width * mask.height; p++) {
    if (!mask.data[p]) continue;
    data[3 * p] = Math.round(0.5 * data[3 * p] + 0.5 * 255);
    data[3 * p + 1] = Math.round(0.5 * data[3 * p + 1]);
    data[3 * p + 2] = Math.round(0.5 * data[3 * p + 2]);
  }
  return { ...rgb, data };
}

/** Close-up crop: the square around the ring, already at the burst size (180 px). */
function closeupCrop(source, box) {
  return V.resize(grabRegion(source, box.x1, box.y1, box.x2, box.y2), V.EYE_CROP_SIZE, V.EYE_CROP_SIZE, "area");
}

// ---------- camera ----------

function applyModeToCameraScreen() {
  const closeup = state.mode === "closeup";
  $("closeup-controls").hidden = !closeup;
  $("eyestate-R").hidden = closeup;
  $("eyestate-L-title").textContent = closeup ? "Eye" : "Left eye";
}

async function startCamera() {
  unlockAudio();   // must happen inside the tap, before any await
  state.mode = selected("mode");
  const facing = selected("facing");
  if (!navigator.mediaDevices?.getUserMedia) {
    show("intro");
    setStatus("load-status", "This browser cannot open the camera here. The page must be opened over https (or localhost).", true);
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
  } catch (e) {
    show("intro");
    setStatus("load-status", e.name === "NotAllowedError"
      ? "Camera access was refused. Allow the camera for this site in the browser settings, then try again."
      : `Could not open the camera (${e.message}).`, true);
    return;
  }
  const video = $("video");
  video.srcObject = state.stream;
  await video.play();
  $("viewer").classList.toggle("mirrored", facing === "user");
  await setRunningMode("VIDEO");
  state.collectors = state.mode === "closeup"
    ? { E: new V.EyeBurstCollector() }
    : { L: new V.EyeBurstCollector(), R: new V.EyeBurstCollector() };
  state.lastVideoTime = -1;
  state.steadySince = null;
  applyModeToCameraScreen();
  setStatus("camera-status", "");
  $("progress").hidden = true;
  show("camera");
  state.running = true;
  requestAnimationFrame(loop);
}

function stopCamera() {
  state.running = false;
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
}

function capturing() {
  return !!state.collectors && Object.values(state.collectors).some((c) => c.active);
}

function loop() {
  if (!state.running) return;
  const video = $("video");
  if (video.readyState >= 2 && video.currentTime !== state.lastVideoTime && !state.processing) {
    state.lastVideoTime = video.currentTime;
    if (state.mode === "closeup") {
      onCloseupFrame(video);
    } else {
      onFaceFrame(video, state.landmarker.detectForVideo(video, performance.now()).faceLandmarks?.[0]);
    }
    afterFrame();
  } else if (capturing() && performance.now() - state.burstStarted > BURST_WALL_TIMEOUT_MS) {
    // no new frames (camera stalled): end the burst anyway so the app never hangs
    Object.values(state.collectors).forEach((c) => { c.active = false; });
    afterFrame();
  }
  requestAnimationFrame(loop);
}

function overlayContext(vw, vh) {
  const overlay = $("overlay");
  if (overlay.width !== vw || overlay.height !== vh) { overlay.width = vw; overlay.height = vh; }
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, vw, vh);
  ctx.lineWidth = Math.max(3, Math.min(vw, vh) / 250);
  return ctx;
}

function setEyeState(label, text, passed) {
  const el = $(`state-${label}`);
  el.textContent = text;
  el.className = passed ? "pass" : "fail";
}

/**
 * Called every frame with whether the picture is good enough to capture. Returns the auto-capture
 * countdown (0..1, or 0 when auto-capture is off) and starts the burst when it reaches 1.
 */
function updateSteady(good) {
  const busy = state.processing || Object.values(state.collectors).some((c) => c.active || c.pending);
  if (!good || busy) { state.steadySince = null; return 0; }
  const now = performance.now();
  if (state.steadySince === null) {
    state.steadySince = now;
    if (now - state.lastAlignedCue > ALIGNED_CUE_GAP_MS) { cue("aligned"); state.lastAlignedCue = now; }
  }
  if (!settings.auto) return 0;
  const progress = Math.min((now - state.steadySince) / AUTO_HOLD_MS, 1);
  if (progress >= 1) startBurst();
  return progress;
}

function onCloseupFrame(video) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const ctx = overlayContext(vw, vh);
  const box = V.guideBox(vw, vh);
  const crop = closeupCrop(video, box);
  const q = V.qualityCheck(crop);
  state.collectors.E.offer(crop, box.irisRadius, box.cropRadius, q.passed);
  const hold = updateSteady(q.passed);
  const color = capturing() ? COLORS.busy : q.passed ? COLORS.pass : COLORS.fail;
  ctx.strokeStyle = color;
  ctx.beginPath();
  ctx.arc(box.cx, box.cy, box.ringRadius, 0, 2 * Math.PI);
  ctx.stroke();
  if (hold > 0) {                                   // auto-capture countdown around the ring
    ctx.strokeStyle = COLORS.busy;
    ctx.lineWidth *= 2;
    ctx.beginPath();
    ctx.arc(box.cx, box.cy, box.ringRadius + ctx.lineWidth * 1.5, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * hold);
    ctx.stroke();
    ctx.lineWidth /= 2;
    ctx.strokeStyle = color;
  }
  ctx.setLineDash([12, 10]);
  ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
  ctx.setLineDash([]);
  setEyeState("L", hold > 0 ? "hold still…" : q.passed ? "sharp, well lit: iris in the ring?" : q.reason.toLowerCase(), q.passed);
  $("capture").disabled = capturing() || !q.passed;
}

function onFaceFrame(video, face) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const ctx = overlayContext(vw, vh);
  const busy = capturing();
  let anyPassed = false;
  const check = face ? V.faceCheck(face, vw, vh) : { ok: false, reason: "no face found" };
  for (const [label, ids] of FACE_EYES) {
    if (!check.ok) { setEyeState(label, check.reason, false); continue; }
    const box = V.eyeBox(face, ids, vw, vh);
    if (!box.valid) { setEyeState(label, "out of view", false); continue; }
    const crop = grabRegion(video, box.x1, box.y1, box.x2, box.y2);
    const q = V.qualityCheck(crop);
    state.collectors[label].offer(crop, box.irisRadius, box.cropRadius, q.passed);
    anyPassed ||= q.passed;
    ctx.strokeStyle = busy ? COLORS.busy : q.passed ? COLORS.pass : COLORS.fail;
    ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
    setEyeState(label, q.passed ? "sharp, well lit" : q.reason.toLowerCase(), q.passed);
  }
  const hold = updateSteady(anyPassed);
  if (!busy) setStatus("camera-status", hold > 0 ? "Hold still…" : "");
  $("capture").disabled = busy || !anyPassed;
}

function afterFrame() {
  const collectors = Object.values(state.collectors);
  if (capturing()) {
    const pending = collectors.filter((c) => c.pending);
    const done = Math.min(...pending.map((c) => c.crops.length / c.nFrames));
    $("progress").firstElementChild.style.width = `${Math.round(100 * done)}%`;
    if (performance.now() - state.burstStarted > BURST_WALL_TIMEOUT_MS) {
      collectors.forEach((c) => { c.active = false; });   // ends the burst; finishBurst reports what was collected
    }
  }
  if (!capturing() && collectors.some((c) => c.pending)) finishBurst();
}

function startBurst() {
  state.steadySince = null;
  cue("start");
  Object.values(state.collectors).forEach((c) => c.start());
  state.burstStarted = performance.now();
  $("progress").hidden = false;
  setStatus("camera-status", "Capturing - hold still…");
}

async function finishBurst() {
  state.processing = true;
  setStatus("camera-status", "Analysing…");
  await new Promise((r) => setTimeout(r, 30));   // let the status paint before the heavy work
  const results = [];
  const eyes = state.mode === "closeup"
    ? [["E", `${selected("which-eye") === "L" ? "Left" : "Right"} eye`]]
    : FACE_EYES.map(([label, , name]) => [label, name]);
  const advice = state.mode === "closeup"
    ? "Hold the phone steady 10-15 cm from the eye, wait for the picture to be sharp, and use brighter, even light."
    : "Hold the phone steadier, keep the whole face in view and use brighter, even light.";
  try {
    for (const [label, name] of eyes) {
      const c = state.collectors[label];
      if (c.ready()) {
        results.push(await analyse(name, c.crops, c.irisRadii, c.cropRadii, { burst: true }));
      } else if (c.pending) {
        results.push({ name, error: `Only ${c.crops.length} of ${c.nFrames} sharp frames were captured. ${advice}` });
      }
      c.pending = false;
    }
  } catch (e) {
    results.push({ name: "Analysis", error: `Something went wrong (${e.message}).` });
  }
  stopCamera();
  state.processing = false;
  cue(results.some((r) => r.error) ? "fail" : "done");
  renderResults(results);
}

// ---------- analysis ----------

async function analyse(name, crops, irisRadii, cropRadii, info) {
  const res = V.glareSeparation(crops, irisRadii, cropRadii);
  const input = V.classifierInput(res, irisRadii, cropRadii);
  const out = await state.session.run({ image: new ort.Tensor("float32", input, [1, 224, 224, 3]) });
  const cataract = out.cataract_score.data[0], mature = out.mature_score.data[0];
  return { name, res, lastFrame: crops[crops.length - 1], cataract, mature,
           output: V.decide(cataract, mature, state.config), ...info };
}

function photoCanvas(source, width, height) {
  const scale = Math.min(1, MAX_PHOTO_SIDE / Math.max(width, height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas;
}

async function analysePhotoSource(source, width, height) {
  const canvas = photoCanvas(source, width, height);
  await setRunningMode("IMAGE");
  const face = state.landmarker.detect(canvas).faceLandmarks?.[0];
  if (!face) return [{ name: "Photo", error: "No face was found. Use a photo that shows the whole face, looking at the camera." }];
  const check = V.faceCheck(face, canvas.width, canvas.height);
  if (!check.ok) {
    return [{ name: "Photo", error: `Can't analyse this photo: ${check.reason}. The app needs a photo of the whole face, looking at the camera, taken from about 15-25 cm.` }];
  }
  const results = [];
  for (const [, ids, name] of FACE_EYES) {
    const box = V.eyeBox(face, ids, canvas.width, canvas.height);
    if (!box.valid) { results.push({ name, error: "This eye is outside the photo." }); continue; }
    const crop = grabRegion(canvas, box.x1, box.y1, box.x2, box.y2);
    const q = V.qualityCheck(crop);
    const eye = V.resize(crop, V.EYE_CROP_SIZE, V.EYE_CROP_SIZE, "area");
    const frames = Array(V.BURST_FRAMES).fill(eye);   // one photo: no glare separation possible
    results.push(await analyse(name, frames, Array(frames.length).fill(box.irisRadius),
                               Array(frames.length).fill(box.cropRadius), { photo: true, quality: q, faceCheck: check }));
  }
  return results;
}

async function analysePhotoFile(file) {
  show("intro");
  setStatus("load-status", "Analysing the photo…");
  try {
    const bitmap = await createImageBitmap(file);
    const results = await analysePhotoSource(bitmap, bitmap.width, bitmap.height);
    setStatus("load-status", "Ready.");
    renderResults(results);
  } catch (e) {
    setStatus("load-status", `Could not read that photo (${e.message}).`, true);
  }
}

// ---------- results ----------

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v; else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

function figure(img, caption) {
  const canvas = el("canvas");
  drawRGB(canvas, img);
  return el("figure", {}, canvas, el("figcaption", {}, caption));
}

function eyeIcon() {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  const use = document.createElementNS(ns, "use");
  use.setAttribute("href", "#eye");
  svg.append(use);
  return svg;
}

/** Score line: green up to the screening cut-off, amber above it, a marker at the score. */
function meter(score) {
  const cut = 100 * state.config.screen_threshold;
  const bar = el("div", { class: "meter" }, el("i", { style: `left:${(100 * Math.min(Math.max(score, 0), 1)).toFixed(1)}%` }));
  bar.style.background = `linear-gradient(90deg, var(--ok) 0 ${cut}%, var(--refer) ${cut}% 100%)`;
  return [bar, el("div", { class: "scale" }, el("span", {}, "clear"),
                  el("span", {}, `cut-off ${state.config.screen_threshold.toFixed(2)}`), el("span", {}, "cloudy"))];
}

function renderResults(results) {
  const box = $("result-cards");
  box.replaceChildren();
  for (const r of results) {
    const title = el("div", { class: "result-eye" }, eyeIcon(),
                     el("span", {}, r.error ? r.name : `${r.name} (person's ${r.name.split(" ")[0].toLowerCase()})`));
    if (r.error) {
      box.append(el("div", { class: "result" }, title,
        el("div", { class: "verdict none" }, el("small", {}, "Result"), el("strong", {}, "No result")),
        el("div", { class: "explain" }, el("p", {}, r.error))));
      continue;
    }
    const [word, text] = OUTPUT_TEXT[r.output];
    const photo = el("canvas", { class: "eye-photo" });
    drawRGB(photo, r.res.medianColor);
    const notes = [`Score ${r.cataract.toFixed(2)}`];
    if (r.output !== "normal") notes.push(`severity ${r.mature.toFixed(2)} (urgent from ${state.config.severity_threshold.toFixed(2)})`);
    notes.push(r.photo ? "single photo, glare separation skipped" : `${V.BURST_FRAMES} frames, glare removed`);
    if (r.photo && !r.quality.passed) notes.push(`photo quality: ${r.quality.reason.toLowerCase()}, so treat with extra caution`);
    const panels = el("div", { class: "panels" },
      figure(r.lastFrame, "last frame"),
      figure(r.res.medianColor, "de-glared"),
      figure(withMaskInRed(r.res.medianColor, r.res.opacityMask), `fixed bright area ${(100 * r.res.opacityFraction).toFixed(1)}%`),
      figure(r.res.glareMask, "moving glare"));
    box.append(el("div", { class: "result" }, title, photo,
      el("div", { class: `verdict ${r.output}` }, el("small", {}, "Result"),
         el("strong", {}, word[0] + word.slice(1).toLowerCase())),
      ...meter(r.cataract),
      el("div", { class: "explain" },
        el("p", {}, text),
        el("p", { class: "meta" }, notes.join(" · ")),
        el("details", {}, el("summary", {}, "Show processing steps"), panels))));
  }
  $("other-eye").hidden = !(state.mode === "closeup" && results.some((r) => r.burst));
  state.reportFile = null;
  $("share").disabled = true;
  buildReport(results).then((file) => { state.reportFile = file; $("share").disabled = false; })
    .catch(() => { $("share").hidden = true; });
  show("results");
}

// ---------- save / share the result ----------
// A summary image is drawn as soon as results appear, because iPhones only open the share sheet
// directly from a tap (no time for drawing in between). Always light colours, so it prints well.

const REPORT = {
  width: 1080, bg: "#f6f4ef", ink: "#13231f", muted: "#5d6b66", line: "#dcd8cf", accent: "#0e6b5c", soft: "#e1ece7",
  colors: { normal: "#1d7a4a", refer: "#a3620a", urgent: "#b3321f" },
  serif: "Newsreader, Georgia, serif", sans: "'IBM Plex Sans', system-ui, sans-serif",
};

function wrapText(ctx, text, x, y, maxWidth, lineHeight) {
  let line = "";
  for (const word of text.split(" ")) {
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width > maxWidth && line) { ctx.fillText(line, x, y); y += lineHeight; line = word; } else line = test;
  }
  if (line) { ctx.fillText(line, x, y); y += lineHeight; }
  return y;
}

/** The app logo (teal tile, eye with a half-clouded pupil, small plus) at x, y, size px. */
function drawLogo(ctx, x, y, size) {
  const s = size / 64;
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(s, s);
  const fill = (color, path) => { ctx.fillStyle = color; ctx.fill(new Path2D(path)); };
  fill("#0e6b5c", "M15 0h34a15 15 0 0 1 15 15v34a15 15 0 0 1-15 15H15A15 15 0 0 1 0 49V15A15 15 0 0 1 15 0z");
  fill("#f6f4ef", "M8 32c6-10 14-15 24-15s18 5 24 15c-6 10-14 15-24 15S14 42 8 32z");
  fill("#0e6b5c", "M43 32a11 11 0 1 1-22 0a11 11 0 1 1 22 0z");
  fill("#13231f", "M38 32a6 6 0 1 1-12 0a6 6 0 1 1 12 0z");
  fill("#b9d6cc", "M32 26a6 6 0 0 1 0 12z");
  ctx.strokeStyle = "#f6f4ef"; ctx.lineWidth = 2.8; ctx.lineCap = "round";
  ctx.stroke(new Path2D("M52 9v8M48 13h8"));
  ctx.restore();
}

async function buildReport(results) {
  try { await Promise.all([document.fonts.load(`500 40px Newsreader`), document.fonts.load(`400 20px "IBM Plex Sans"`)]); } catch { /* system fonts */ }
  const R = REPORT, pad = 80, w = R.width;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = 300 + results.length * 440 + 260;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = R.bg; ctx.fillRect(0, 0, w, canvas.height);
  ctx.textBaseline = "alphabetic";

  drawLogo(ctx, pad, 62, 60);
  ctx.fillStyle = R.ink; ctx.font = `600 30px ${R.sans}`; ctx.fillText("Cataract Screen", pad + 78, 103);
  ctx.fillStyle = R.muted; ctx.font = `400 24px ${R.sans}`;
  ctx.fillText(new Date().toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }), pad, 176);
  ctx.fillStyle = R.ink; ctx.font = `500 56px ${R.serif}`; ctx.fillText("Eye screening result", pad, 250);

  let y = 300;
  for (const r of results) {
    ctx.fillStyle = R.line; ctx.fillRect(pad, y, w - 2 * pad, 2);
    const photoSize = 300, px = pad, py = y + 50;
    if (r.res) {
      const img = document.createElement("canvas");
      drawRGB(img, r.res.medianColor);
      ctx.save();
      ctx.beginPath(); ctx.arc(px + photoSize / 2, py + photoSize / 2, photoSize / 2, 0, 2 * Math.PI); ctx.clip();
      ctx.drawImage(img, px, py, photoSize, photoSize);
      ctx.restore();
      ctx.strokeStyle = R.soft; ctx.lineWidth = 8;
      ctx.beginPath(); ctx.arc(px + photoSize / 2, py + photoSize / 2, photoSize / 2 + 14, 0, 2 * Math.PI); ctx.stroke();
    }
    const tx = pad + photoSize + 60, tw = w - tx - pad;
    ctx.fillStyle = R.muted; ctx.font = `600 22px ${R.sans}`; ctx.fillText(r.name.toUpperCase(), tx, py + 30);
    if (r.error) {
      ctx.fillStyle = R.muted; ctx.font = `500 64px ${R.serif}`; ctx.fillText("No result", tx, py + 110);
      ctx.fillStyle = R.ink; ctx.font = `400 24px ${R.sans}`; wrapText(ctx, r.error, tx, py + 160, tw, 34);
    } else {
      const [word, text] = OUTPUT_TEXT[r.output];
      ctx.fillStyle = R.colors[r.output]; ctx.font = `500 84px ${R.serif}`;
      ctx.fillText(word[0] + word.slice(1).toLowerCase(), tx, py + 120);
      // score line: green up to the cut-off, amber above, marker at the score
      const ly = py + 170, cut = state.config.screen_threshold;
      ctx.fillStyle = R.colors.normal; ctx.fillRect(tx, ly, tw * cut, 10);
      ctx.fillStyle = R.colors.refer; ctx.fillRect(tx + tw * cut, ly, tw * (1 - cut), 10);
      const mx = tx + tw * Math.min(Math.max(r.cataract, 0), 1);
      ctx.fillStyle = R.bg; ctx.strokeStyle = R.ink; ctx.lineWidth = 5;
      ctx.beginPath(); ctx.arc(mx, ly + 5, 13, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.fillStyle = R.muted; ctx.font = `400 20px ${R.sans}`;
      ctx.fillText("clear", tx, ly + 50);
      const cutLabel = `cut-off ${cut.toFixed(2)}`;
      ctx.fillText(cutLabel, tx + tw * cut - ctx.measureText(cutLabel).width / 2, ly + 50);
      ctx.fillText("cloudy", tx + tw - ctx.measureText("cloudy").width, ly + 50);
      ctx.fillStyle = R.ink; ctx.font = `400 24px ${R.sans}`;
      let ty = wrapText(ctx, text, tx, ly + 100, tw, 34);
      ctx.fillStyle = R.muted; ctx.font = `400 20px ${R.sans}`;
      wrapText(ctx, `Score ${r.cataract.toFixed(2)}${r.output !== "normal" ? ` · severity ${r.mature.toFixed(2)}` : ""} · ${r.photo ? "single photo" : `${V.BURST_FRAMES} frames, glare removed`}`, tx, ty + 6, tw, 28);
    }
    y += 440;
  }
  ctx.fillStyle = R.line; ctx.fillRect(pad, y, w - 2 * pad, 2);
  ctx.fillStyle = R.ink; ctx.font = `600 22px ${R.sans}`; ctx.fillText("Student prototype, not a diagnosis.", pad, y + 60);
  ctx.fillStyle = R.muted; ctx.font = `400 22px ${R.sans}`;
  y = wrapText(ctx, "This result comes from an unvalidated screening model and can be wrong either way. Please show it to an eye doctor (ophthalmologist) for a proper eye examination.", pad, y + 96, w - 2 * pad, 32);
  ctx.fillText(location.origin + location.pathname, pad, y + 20);

  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");
  return new File([blob], `cataract-screen-${stamp}.png`, { type: "image/png" });
}

async function shareReport() {
  const file = state.reportFile;
  if (!file) return;
  if (navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: "Cataract Screen result" });
      return;
    } catch (e) {
      if (e.name === "AbortError") return;   // user closed the share sheet
    }
  }
  const url = URL.createObjectURL(file);
  const a = el("a", { href: url, download: file.name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

// ---------- wiring ----------

function showModeSteps() {
  const mode = selected("mode");
  $("steps-closeup").hidden = mode !== "closeup";
  $("steps-face").hidden = mode !== "face";
}

$("start").addEventListener("click", startCamera);
$("capture").addEventListener("click", startBurst);
$("back").addEventListener("click", () => { stopCamera(); show("intro"); });
$("again").addEventListener("click", () => show("intro"));
$("share").addEventListener("click", shareReport);
$("other-eye").addEventListener("click", () => {
  const other = selected("which-eye") === "L" ? "R" : "L";
  document.querySelector(`input[name=which-eye][value=${other}]`).checked = true;
  startCamera();
});
document.querySelectorAll("input[name=mode]").forEach((r) => r.addEventListener("change", showModeSteps));
// Instruction illustrations: copy each <symbol> into place, because CSS animations don't reach
// shapes reused through <use> in every browser (Safari).
document.querySelectorAll(".ill use").forEach((u) => {
  const symbol = document.querySelector(u.getAttribute("href"));
  if (symbol) u.closest("svg").replaceChildren(...[...symbol.childNodes].map((n) => n.cloneNode(true)));
});
$("auto-capture").checked = settings.auto;
$("cues").checked = settings.cues;
$("auto-capture").addEventListener("change", (e) => { settings.auto = e.target.checked; saveSetting("autoCapture", settings.auto); state.steadySince = null; });
$("cues").addEventListener("change", (e) => {
  settings.cues = e.target.checked;
  saveSetting("cues", settings.cues);
  if (settings.cues) { unlockAudio(); cue("aligned"); }   // play a sample so the user hears it is on
});
$("photo-input").addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  e.target.value = "";
  if (file && state.session) analysePhotoFile(file);
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden && state.running) { stopCamera(); show("intro"); }   // release the camera in the background
});
showModeSteps();

const ready = loadModels();

// Test hooks: let automated browser checks run the pipeline without a camera or file picker.
window.cataractApp = {
  ready,
  lastReport: () => state.reportFile,
  async loadImage(url) {
    const img = new Image();
    img.src = url;
    await img.decode();
    return img;
  },
  // Whole-face photo, as the "Choose a photo" button does.
  async analyseImageUrl(url) {
    await ready;
    const img = await this.loadImage(url);
    const results = await analysePhotoSource(img, img.naturalWidth, img.naturalHeight);
    renderResults(results);
    return results.map((r) => (r.error ? { name: r.name, error: r.error }
      : { name: r.name, output: r.output, cataract: r.cataract, mature: r.mature, quality: r.quality, faceCheck: r.faceCheck }));
  },
  // One-eye close-up photo with the iris at (cx, cy), radius r px: framed exactly as close-up mode frames the ring.
  async analyseCloseupUrl(url, cx, cy, r) {
    await ready;
    const img = await this.loadImage(url);
    const half = Math.trunc(V.CLOSEUP_CROP_HALF_IRIS * r);
    const box = { x1: Math.round(cx - half), y1: Math.round(cy - half), x2: Math.round(cx + half), y2: Math.round(cy + half) };
    const eye = closeupCrop(img, box);
    const frames = Array(V.BURST_FRAMES).fill(eye);
    const res = await analyse("Left eye", frames, frames.map(() => r), frames.map(() => half), { photo: true, quality: V.qualityCheck(eye) });
    renderResults([res]);
    return { output: res.output, cataract: res.cataract, mature: res.mature, quality: res.quality.reason };
  },
  // A saved burst (the 180x180 frames the laptop demo writes to captures/).
  async analyseBurstUrls(urls, irisToCropRatio) {
    await ready;
    const frames = [];
    for (const url of urls) {
      const img = await this.loadImage(url);
      frames.push(grabRegion(img, 0, 0, img.naturalWidth, img.naturalHeight));
    }
    const ones = Array(frames.length).fill(1);
    const r = await analyse("Left eye", frames, ones.map(() => irisToCropRatio), ones, { burst: true });
    renderResults([r]);
    return { output: r.output, cataract: r.cataract, mature: r.mature, opacity: r.res.opacityFraction };
  },
};
