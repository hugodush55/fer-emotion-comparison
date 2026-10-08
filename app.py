"""Interface Gradio : compare LBP+KNN, HOG+SVM et mini-Xception sur le même visage.

    python app.py                  # CNN sur CPU (comparaison de latence équitable)
    python app.py --device cuda    # CNN sur GPU
    python app.py --share          # lien public temporaire
"""
import argparse
from collections import defaultdict

import cv2
import gradio as gr
import numpy as np

from src.config import LABELS, ROOT
from src.pipeline import MODEL_NAMES, EmotionPipeline

EXAMPLES_DIR = ROOT / "examples"
pipeline: EmotionPipeline = None
session_latency = defaultdict(list)  # latences cumulées depuis le lancement, par modèle


def export_examples():
    """Exporte une image FER-2013 de test par classe (si le dataset est présent) pour gr.Examples."""
    from src.data import FER_DIR
    paths = []
    for name in LABELS:
        src = sorted((FER_DIR / "test" / name.lower()).glob("*.jpg"))
        if src:
            EXAMPLES_DIR.mkdir(exist_ok=True)
            dst = EXAMPLES_DIR / f"{name.lower()}.png"
            if not dst.exists():
                cv2.imwrite(str(dst), cv2.resize(cv2.imread(str(src[0]), 0), (192, 192), interpolation=cv2.INTER_CUBIC))
            paths.append(str(dst))
    return paths


def card(pred):
    lat = session_latency[pred.model]
    return (f"### {MODEL_NAMES[pred.model]}\n"
            f"**{pred.label}** · confiance **{pred.confidence:.0%}**\n\n"
            f"⏱ **{pred.latency_ms:.1f} ms** (moyenne session : {np.mean(lat):.1f} ms sur {len(lat)} images)")


def empty_card(name):
    return f"### {MODEL_NAMES[name]}\n—"


def annotate(image, faces):
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    thickness = max(2, out.shape[1] // 250)
    for i, (x, y, w, h) in enumerate(faces):
        color = (0, 200, 0) if i == 0 else (255, 165, 0)  # vert : visage analysé, orange : autres
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def analyze(image, precropped=False, align=True):
    n_out = 3 + 2 * len(MODEL_NAMES)
    if image is None:
        return [None, None, "Aucune image."] + [None] * (n_out - 3)

    res = pipeline.process(image, detect=not precropped, align=align)
    if res.face48 is None:
        status = f"⚠️ Aucun visage détecté (détection : {res.detect_ms:.1f} ms)."
        outputs = [image, None, status]
        for name in MODEL_NAMES:
            outputs += [empty_card(name), None]
        return outputs

    if precropped:
        status = "Image utilisée telle quelle (déjà recadrée), sans détection."
    else:
        n = len(res.faces)
        status = f"{n} visage{'s' if n > 1 else ''} détecté{'s' if n > 1 else ''} en {res.detect_ms:.1f} ms"
        if n > 1:
            status += " : le plus grand (en vert) est analysé."
    face_view = cv2.resize(res.face48, (192, 192), interpolation=cv2.INTER_NEAREST)
    outputs = [annotate(image, res.faces), face_view, status]
    for pred in res.predictions:
        session_latency[pred.model].append(pred.latency_ms)
        outputs += [card(pred), {lab: float(p) for lab, p in zip(LABELS, pred.probs)}]
    return outputs


def build_ui():
    with gr.Blocks(title="Émotions faciales : 3 modèles") as demo:
        gr.Markdown(
            "# Reconnaissance d'émotions faciales\n"
            "Le même visage (détecté par Haar cascade, recadré, aligné, 48×48 en niveaux de gris) "
            "est classé par les 3 modèles. Latence = extraction des features + inférence, mesurée séparément pour chaque modèle."
        )
        align = gr.Checkbox(value=True, label="Aligner le visage sur les yeux")

        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    img_in = gr.Image(type="numpy", sources=["upload", "clipboard"], label="Image", height=320)
                    with gr.Column():
                        precropped = gr.Checkbox(value=False, label="Image déjà recadrée (FER-2013) : pas de détection")
                        btn = gr.Button("Analyser", variant="primary")
                        examples = export_examples()
                        if examples:
                            gr.Examples(examples, inputs=img_in, label="Exemples FER-2013 (cocher « déjà recadrée »)")
            with gr.Tab("Webcam"):
                cam = gr.Image(type="numpy", sources=["webcam"], streaming=True, label="Webcam", height=320)

        with gr.Row():
            annotated = gr.Image(label="Détection", height=280, interactive=False)
            face = gr.Image(label="Visage 48×48 envoyé aux modèles", height=280, width=280, interactive=False)
        status = gr.Markdown()

        outputs = [annotated, face, status]
        with gr.Row(equal_height=True):
            for name in MODEL_NAMES:
                with gr.Column(variant="panel"):
                    outputs.append(gr.Markdown(empty_card(name)))
                    outputs.append(gr.Label(num_top_classes=7, show_label=False))

        btn.click(analyze, [img_in, precropped, align], outputs)
        img_in.upload(analyze, [img_in, precropped, align], outputs)
        cam.stream(lambda im, al: analyze(im, False, al), [cam, align], outputs,
                   stream_every=0.25, concurrency_limit=1, show_progress="hidden")
    return demo


def main():
    global pipeline
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu ou cuda (pour le CNN)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    pipeline = EmotionPipeline(device=args.device)
    build_ui().launch(server_name="0.0.0.0", server_port=args.port, share=args.share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
