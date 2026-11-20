"""
Cataract Screening Project - Build the Model for the Web App
--------------------------------------------------------------
Turns the demo's models into one ONNX file the browser can run with ONNX Runtime Web:

  models/xdomain/cataract_classifier.tflite   (web-trained MobileNetV3 backbone)
  + models/cv_phone/heads.npz                 (screening + severity logistic regressions)
  -> web/model/cataract.onnx                  input  "image": 1x224x224x3, RGB, 0-255 float
                                              output "cataract_score", "mature_score": 1x1
  -> web/model/config.json                    thresholds, so the app makes the same calls as the demo

Conversion notes:
  * A small dedicated converter (convert_backbone) handles exactly the ops this model uses:
    CONV_2D, DEPTHWISE_CONV_2D, ADD, MUL, HARD_SWISH, MEAN (all float32). Off-the-shelf
    tflite2onnx failed on MobileNetV3 (no HARD_SWISH, broadcast bugs).
  * Layout: the input stays NHWC (as in the app); one Transpose makes it NCHW for ONNX Conv.
  * HARD_SWISH(x) = x * relu6(x + 3) / 6 = x * HardSigmoid(x; alpha=1/6, beta=0.5)  (exact).
  * The graph stops at the global average pool (576 features); the two logistic-regression heads
    are appended as plain ONNX nodes.

Then checks the ONNX model against the laptop demo (OpenCV running the .tflite + the NumPy
heads) on real eye crops, and fails if they disagree.

Run (.venv-web has onnx, onnxruntime, tflite, numpy, pillow; .venv makes the OpenCV reference):
    .venv\\Scripts\\python web\\build_web_model.py --reference      # writes reference scores with OpenCV
    .venv-web\\Scripts\\python web\\build_web_model.py              # converts and checks
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TFLITE = ROOT / "models/xdomain/cataract_classifier.tflite"
HEADS = ROOT / "models/cv_phone/heads.npz"
OUT_DIR = ROOT / "web/model"
REFERENCE = ROOT / "web/model/reference_scores.json"
CROPS = ROOT / "data/combined/mixed"
N_REFERENCE = 60
MAX_SCORE_DIFF = 1e-3


def reference_images():
    """A fixed, mixed sample of real eye crops (phone and web, both classes)."""
    files = sorted(CROPS.rglob("*.png"))
    rng = np.random.default_rng(0)
    return [files[i] for i in sorted(rng.choice(len(files), N_REFERENCE, replace=False))]


def make_reference():
    """Runs in .venv: the laptop demo's own classifier is the ground truth."""
    sys.path.insert(0, str(ROOT))
    import cv2
    from eye_quality_gate_demo import EyeClassifier
    clf = EyeClassifier(str(TFLITE), str(HEADS))
    out = []
    for f in reference_images():
        pred = clf.predict(cv2.imread(str(f)))
        out.append({"file": str(f.relative_to(ROOT)).replace("\\", "/"),
                    "cataract_score": pred["cataract_score"], "mature_score": pred["mature_score"],
                    "output": pred["output"]})
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE.write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} reference predictions to {REFERENCE}")


OPSET = 13
IR_VERSION = 8          # the newest onnx library writes IR 14, which ONNX Runtime 1.20 (web) cannot load


def convert_backbone():
    """Returns (onnx.ModelProto, name of the pooled-feature tensor) for the TFLite backbone."""
    import onnx
    import tflite
    from onnx import TensorProto, helper, numpy_helper

    m = tflite.Model.GetRootAsModel(TFLITE.read_bytes(), 0)
    g = m.Subgraphs(0)
    op_name = {v: k for k, v in tflite.BuiltinOperator.__dict__.items() if not k.startswith("_")}

    def tensor_array(i):
        t = g.Tensors(i)
        data = m.Buffers(t.Buffer()).DataAsNumpy()
        if isinstance(data, int) or data is None or len(data) == 0:
            return None
        dtype = {tflite.TensorType.FLOAT32: np.float32, tflite.TensorType.INT32: np.int32}[t.Type()]
        return np.frombuffer(data.tobytes(), dtype=dtype).reshape(t.ShapeAsNumpy())

    def shape(i):
        return [int(x) for x in g.Tensors(i).ShapeAsNumpy()]

    def options(op, cls):
        o = cls()
        table = op.BuiltinOptions()
        o.Init(table.Bytes, table.Pos)
        return o

    nodes, inits = [], []
    names = {}                   # tflite tensor index -> ONNX name (activations, NCHW)
    counter = [0]

    def new(prefix):
        counter[0] += 1
        return f"{prefix}_{counter[0]}"

    def const(arr, prefix="c"):
        name = new(prefix)
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))
        return name

    def operand(i):
        """ONNX name for a tflite input: an activation, or a constant reshaped to broadcast in NCHW."""
        if i in names:
            return names[i]
        arr = tensor_array(i)
        assert arr is not None, f"tensor {i} is neither computed nor constant"
        if arr.size == 1:
            return const(arr.reshape([]).astype(np.float32))
        return const(arr.reshape(1, -1, 1, 1).astype(np.float32))   # per-channel constant

    def activation(x, fused, prefix):
        if fused == tflite.ActivationFunctionType.NONE:
            return x
        out = new(prefix)
        if fused == tflite.ActivationFunctionType.RELU:
            nodes.append(helper.make_node("Relu", [x], [out]))
        elif fused == tflite.ActivationFunctionType.RELU6:
            nodes.append(helper.make_node("Clip", [x, const(np.float32(0)), const(np.float32(6))], [out]))
        else:
            raise NotImplementedError(f"fused activation {fused}")
        return out

    def same_pads(in_hw, k_hw, stride, dil):
        pads = []
        for n, k, st, d in zip(in_hw, k_hw, stride, dil):
            out = -(-n // st)
            total = max((out - 1) * st + (k - 1) * d + 1 - n, 0)
            pads.append((total // 2, total - total // 2))
        return [pads[0][0], pads[1][0], pads[0][1], pads[1][1]]   # top, left, bottom, right

    image_idx = g.Inputs(0)
    names[image_idx] = new("nchw")
    nodes.append(helper.make_node("Transpose", ["image"], [names[image_idx]], perm=[0, 3, 1, 2]))

    pooled = None
    for k in range(g.OperatorsLength()):
        op = g.Operators(k)
        code = m.OperatorCodes(op.OpcodeIndex())
        kind = op_name[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())]
        ins = [op.Inputs(j) for j in range(op.InputsLength())]
        out_idx = op.Outputs(0)
        if kind in ("FULLY_CONNECTED", "LOGISTIC"):
            continue                                    # original web-only head: replaced by our heads
        if kind in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            depthwise = kind == "DEPTHWISE_CONV_2D"
            o = options(op, tflite.DepthwiseConv2DOptions if depthwise else tflite.Conv2DOptions)
            w = tensor_array(ins[1])
            b = tensor_array(ins[2]) if len(ins) > 2 and ins[2] >= 0 else None
            in_shape = shape(ins[0])                    # NHWC
            if depthwise:                               # [1, kh, kw, C*M] -> [C*M, 1, kh, kw]
                assert o.DepthMultiplier() == 1
                kh, kw, cm = w.shape[1:]
                w_onnx = w.reshape(kh, kw, cm).transpose(2, 0, 1)[:, None]
                group = in_shape[3]
            else:                                       # [O, kh, kw, I] -> [O, I, kh, kw]
                kh, kw = w.shape[1:3]
                w_onnx = w.transpose(0, 3, 1, 2)
                group = 1
            stride = [o.StrideH(), o.StrideW()]
            dil = [o.DilationHFactor(), o.DilationWFactor()]
            pads = (same_pads(in_shape[1:3], [kh, kw], stride, dil)
                    if o.Padding() == tflite.Padding.SAME else [0, 0, 0, 0])
            conv_in = [names[ins[0]], const(w_onnx.astype(np.float32), "w")]
            if b is not None:
                conv_in.append(const(b.astype(np.float32), "b"))
            y = new("conv")
            nodes.append(helper.make_node("Conv", conv_in, [y], kernel_shape=[kh, kw], strides=stride,
                                          dilations=dil, pads=pads, group=group))
            names[out_idx] = activation(y, o.FusedActivationFunction(), "act")
        elif kind in ("ADD", "MUL"):
            o = options(op, tflite.AddOptions if kind == "ADD" else tflite.MulOptions)
            y = new(kind.lower())
            nodes.append(helper.make_node("Add" if kind == "ADD" else "Mul",
                                          [operand(ins[0]), operand(ins[1])], [y]))
            names[out_idx] = activation(y, o.FusedActivationFunction(), "act")
        elif kind == "HARD_SWISH":
            x = names[ins[0]]
            gate, y = new("hsig"), new("hswish")
            nodes.append(helper.make_node("HardSigmoid", [x], [gate], alpha=1.0 / 6.0, beta=0.5))
            nodes.append(helper.make_node("Mul", [x, gate], [y]))
            names[out_idx] = y
        elif kind == "MEAN":
            o = options(op, tflite.ReducerOptions)
            axes = tensor_array(ins[1]).reshape(-1).tolist()
            assert sorted(axes) == [1, 2], f"MEAN over {axes}"
            y = new("mean")
            nodes.append(helper.make_node("ReduceMean", [names[ins[0]]], [y], axes=[2, 3],
                                          keepdims=int(bool(o.KeepDims()))))
            names[out_idx] = y
            if not o.KeepDims():
                pooled = y                              # the final global average pool: 576 features
        else:
            raise NotImplementedError(f"TFLite op {kind}")

    assert pooled is not None, "global average pool not found"
    graph = helper.make_graph(
        nodes, "cataract_backbone",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 224, 224, 3])],
        [helper.make_tensor_value_info(pooled, TensorProto.FLOAT, [1, 576])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)],
                              producer_name="cataract-project build_web_model.py")
    onnx.checker.check_model(model)
    return model, pooled


def add_heads(model, pooled):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    heads = np.load(HEADS)
    g = model.graph
    flat = "pooled_flat"
    g.node.append(helper.make_node("Flatten", [pooled], [flat], name="pooled_flatten", axis=1))
    for name, out in (("screen", "cataract_score"), ("severity", "mature_score")):
        mean, std, beta = (heads[f"{name}_{k}"].astype(np.float32) for k in ("mean", "std", "beta"))
        # logistic regression on standardised features, folded into one MatMul + Add:
        #   sigmoid(((x - mean) / std) . w + b) = sigmoid(x . (w / std) + (b - sum(mean * w / std)))
        w = (beta[:-1] / std).astype(np.float32).reshape(-1, 1)
        b = np.array([beta[-1] - float(np.sum(mean * beta[:-1] / std))], dtype=np.float32)
        g.initializer.extend([numpy_helper.from_array(w, f"{name}_w"), numpy_helper.from_array(b, f"{name}_b")])
        g.node.extend([
            helper.make_node("MatMul", [flat, f"{name}_w"], [f"{name}_logit0"], name=f"{name}_matmul"),
            helper.make_node("Add", [f"{name}_logit0", f"{name}_b"], [f"{name}_logit"], name=f"{name}_add"),
            helper.make_node("Sigmoid", [f"{name}_logit"], [out], name=f"{name}_sigmoid"),
        ])
    del g.output[:]
    g.output.extend([helper.make_tensor_value_info("cataract_score", TensorProto.FLOAT, [1, 1]),
                     helper.make_tensor_value_info("mature_score", TensorProto.FLOAT, [1, 1])])
    onnx.checker.check_model(model)
    return model


def check(model_path):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    ref = json.loads(REFERENCE.read_text())
    worst = 0.0
    same = 0
    heads = np.load(HEADS)
    for r in ref:
        img = read_rgb(ROOT / r["file"])
        cat, mat = sess.run(["cataract_score", "mature_score"], {inp.name: img[None]})
        cat, mat = float(cat.reshape(-1)[0]), float(mat.reshape(-1)[0])
        worst = max(worst, abs(cat - r["cataract_score"]), abs(mat - r["mature_score"]))
        out = ("normal" if cat < float(heads["screen_threshold"])
               else "urgent" if mat >= float(heads["severity_threshold"]) else "refer")
        same += out == r["output"]
    print(f"checked {len(ref)} eye crops: max score difference vs laptop demo {worst:.2e}, "
          f"same NORMAL/REFER/URGENT call on {same}/{len(ref)}")
    assert worst <= MAX_SCORE_DIFF and same == len(ref), "ONNX model disagrees with the laptop demo"


def read_rgb(path):
    """224x224 RGB float32 from a PNG (Pillow; OpenCV is not in .venv-web)."""
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def main(args):
    if args.reference:
        make_reference()
        return
    import onnx
    model, pooled = convert_backbone()
    print(f"backbone converted: {len(model.graph.node)} ONNX nodes")
    model = add_heads(model, pooled)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, str(OUT_DIR / "cataract.onnx"))
    heads = np.load(HEADS)
    (OUT_DIR / "config.json").write_text(json.dumps({
        "input_size": 224, "crop_half_width_iris": 2.2,
        "screen_threshold": float(heads["screen_threshold"]),
        "severity_threshold": float(heads["severity_threshold"]),
        "source": "models/xdomain/cataract_classifier.tflite + models/cv_phone/heads.npz"}, indent=1))
    size = (OUT_DIR / "cataract.onnx").stat().st_size / 1e6
    print(f"saved {OUT_DIR / 'cataract.onnx'} ({size:.1f} MB) and config.json")
    check(OUT_DIR / "cataract.onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="store_true", help="write reference scores with OpenCV (.venv)")
    main(parser.parse_args())
