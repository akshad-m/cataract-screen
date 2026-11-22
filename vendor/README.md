# Bundled third-party files

These are copied here so the app needs no outside servers and works offline once saved on a phone
(see `../build_sw.py`). They are unmodified copies of the published packages.

| Folder | What | Version | Source | Licence |
|---|---|---|---|---|
| `ort/` | ONNX Runtime Web (`ort.wasm.min.js` + SIMD WebAssembly engine) | 1.20.1 | npm `onnxruntime-web` | MIT, (c) Microsoft Corporation |
| `mediapipe/vision_bundle.mjs`, `mediapipe/wasm/` | MediaPipe Tasks Vision | 0.10.21 | npm `@mediapipe/tasks-vision` | Apache-2.0, (c) Google LLC |
| `mediapipe/face_landmarker.task` | Face Landmarker model (float16, v1) | 1 | storage.googleapis.com/mediapipe-models | Apache-2.0, (c) Google LLC |
| `fonts/` | Newsreader, IBM Plex Sans, Noto Sans Devanagari, Noto Serif Devanagari (Latin, Latin-ext and Devanagari subsets, woff2) | Google Fonts, 2026-11 | fonts.googleapis.com | SIL Open Font License 1.1 |

To update a library: replace its files, change the paths/versions above, then run
`python web/build_sw.py` so phones download the new copy.
