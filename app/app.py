"""Flask demo: pick a validation image or upload one, see the top-5 guesses.

    python app/app.py --port 5088

The model is loaded once at start-up. Every response embeds the image as a data
URL so the page needs no second request and screenshots render offline.
"""

import argparse
import base64
import io
import os
import sys
import time

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SAMPLE_IMAGES, default_checkpoint  # noqa: E402
from src.data import load_sample_index  # noqa: E402
from src.predict import Predictor  # noqa: E402

app = Flask(__name__)
STATE = {}


def bootstrap(checkpoint=None):
    """Load the model and the demo gallery once, then warm it up."""
    if STATE.get("predictor"):
        return STATE

    predictor = Predictor(checkpoint)
    samples = load_sample_index()

    # First inference pays for CUDA kernel selection; do it now so the latency
    # shown in the UI is a steady-state number rather than a one-off 400 ms.
    predictor.predict(Image.new("RGB", (320, 320), (30, 30, 30)), topk=1)

    STATE["predictor"] = predictor
    STATE["samples"] = samples
    STATE["checkpoint"] = predictor.checkpoint
    return STATE


def to_data_url(image, size=420):
    image = image.convert("RGB")
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


@app.route("/")
def index():
    state = bootstrap()
    gallery = []
    for row in state["samples"][:12]:
        gallery.append({
            "file_name": row["file_name"],
            "class_name": row["class_name"],
            "thumb": to_data_url(Image.open(row["path"]), 260),
        })
    return render_template(
        "index.html",
        gallery=gallery,
        checkpoint=os.path.basename(state["checkpoint"]),
        sample_count=len(state["samples"]),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    state = bootstrap()
    predictor = state["predictor"]
    topk = int(request.form.get("topk", 5))

    sample_name = request.form.get("sample")
    upload = request.files.get("image")

    if sample_name:
        match = next((r for r in state["samples"] if r["file_name"] == sample_name), None)
        if match is None:
            return jsonify({"error": f"unknown sample {sample_name}"}), 404
        image = Image.open(match["path"])
        truth = match["class_name"]
        source = sample_name
    elif upload is not None:
        image = Image.open(upload.stream)
        truth = None
        source = upload.filename
    else:
        return jsonify({"error": "provide `sample` or `image`"}), 400

    started = time.perf_counter()
    predictions = predictor.predict(image, topk)
    elapsed_ms = (time.perf_counter() - started) * 1000

    return jsonify({
        "source": source,
        "ground_truth": truth,
        "predictions": predictions,
        "top1": predictions[0]["label"],
        "confidence": predictions[0]["probability"],
        "correct": None if truth is None else predictions[0]["label"] == truth,
        "inference_ms": round(elapsed_ms, 1),
        "image": to_data_url(image),
    })


@app.route("/sample-image/<path:name>")
def sample_image(name):
    path = os.path.join(SAMPLE_IMAGES, name)
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/api/health")
def health():
    state = bootstrap()
    return jsonify({
        "status": "ok",
        "checkpoint": os.path.relpath(state["checkpoint"]),
        "samples": len(state["samples"]),
        "device": str(state["predictor"].device),
        "classes": len(state["predictor"].labels),
    })


def parse_args():
    parser = argparse.ArgumentParser(description="ZhiHua flower classifier demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5088)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    state = bootstrap(args.checkpoint)
    print(f"checkpoint {state['checkpoint']}")
    print(f"device     {state['predictor'].device}")
    print(f"samples    {len(state['samples'])}")
    app.run(host=args.host, port=args.port, debug=False)
