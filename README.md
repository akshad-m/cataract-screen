# Cataract Screen (student project prototype)

**Open the app:** https://akshad-m.github.io/cataract-screen/

A final-year engineering project: a phone-browser app that captures an eye, rejects blurry or badly lit
shots, removes moving reflections from an 8-frame burst, and runs a small screening model that answers
NORMAL / REFER / URGENT.

> **Not a medical device.** The model was trained on public datasets (web photos and one hospital's phone
> photos) and has not been clinically validated. In cross-validation on phone photos it scored AUC 0.90,
> catching about 88% of cataracts while flagging about 1 in 4 normal eyes. It can be wrong either way.
> Anyone with eye problems should see an eye doctor.

## Privacy
Everything runs on the phone: the camera images are processed in the browser and are never uploaded
or stored. The page loads its libraries from public CDNs (MediaPipe, ONNX Runtime Web) and the face
landmark model from Google's storage.

## What is here
| File | Purpose |
|---|---|
| `index.html`, `app.js`, `style.css` | The app: camera, close-up ring or whole-face capture, results |
| `vision.js` | Quality gate, burst collection, glare separation, model input (port of the Python prototype) |
| `model/cataract.onnx`, `model/config.json` | MobileNetV3 backbone + screening and severity models, and their cut-offs |
| `build_web_model.py`, `test/` | How the model was converted and how the JavaScript was checked against the Python code (they need the full private project to run) |

Built with MediaPipe Face Landmarker and ONNX Runtime Web.
