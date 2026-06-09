"""
EmotionAI Web — Backend Flask
Raspberry Pi 4 + RB Cam + Google Coral USB Accelerator
"""

import cv2
import numpy as np
import time
import threading
import logging
from flask import Flask, jsonify
from flask_cors import CORS

from database import init_db, save_detection

# ── Intentar cargar PyCoral / TFLite (si no está disponible, usar modo simulado) ──
try:
    from pycoral.utils.edgetpu import make_interpreter
    from pycoral.adapters.common import input_size, set_input
    from pycoral.adapters.classify import get_classes
    CORAL_AVAILABLE = True
except ImportError:
    CORAL_AVAILABLE = False
    logging.warning("PyCoral no disponible. Usando TFLite sin Edge TPU.")

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        logging.warning("TFLite no disponible. El servidor arrancará en modo demo.")

# ── Configuración ─────────────────────────────────────────────────────────────
MODEL_PATH      = "models/emociones_edgetpu.tflite"   # modelo Edge TPU
MODEL_PATH_CPU  = "models/emociones.tflite"           # fallback sin TPU
CASCADE_PATH    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CAMERA_INDEX    = 0        # índice de la RB Cam
INPUT_SIZE      = (48, 48) # resolución que espera el modelo
CONFIDENCE_MIN  = 0.30     # descartar detecciones por debajo de este umbral
FRAME_SKIP      = 2        # procesar 1 de cada N frames (reduce carga CPU)

EMOTION_LABELS = ["enojado", "sorprendido", "neutral", "feliz", "triste"]
# Orden según las clases reales de tu modelo .tflite

# ── Estado global compartido ──────────────────────────────────────────────────
state = {
    "emotion":    None,
    "confidence": 0.0,
    "fps":        0,
    "tpu":        "Desconectado",
    "face_count": 0,
}
state_lock = threading.Lock()

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # permite que el HTML externo haga fetch sin CORS error

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


# ── Carga del modelo ──────────────────────────────────────────────────────────
def load_model():
    """Intenta cargar el modelo en este orden: Coral TPU → TFLite CPU → modo demo."""
    if CORAL_AVAILABLE:
        try:
            interp = make_interpreter(MODEL_PATH)
            interp.allocate_tensors()
            logging.info("✅ Modelo cargado en Coral Edge TPU")
            return interp, "tpu"
        except Exception as e:
            logging.warning(f"Coral TPU falló: {e}. Probando TFLite CPU...")

    if TFLITE_AVAILABLE:
        cpu_path = MODEL_PATH_CPU if __import__("os").path.exists(MODEL_PATH_CPU) else MODEL_PATH.replace("_edgetpu", "")
        try:
            interp = tflite.Interpreter(model_path=cpu_path)
            interp.allocate_tensors()
            logging.info(f"✅ Modelo cargado en CPU ({cpu_path})")
            return interp, "cpu"
        except Exception as e:
            logging.warning(f"TFLite CPU falló: {e}. Usando modo demo.")

    logging.warning("⚠️  Sin modelo disponible — arrancando en modo DEMO.")
    return None, "demo"


# ── Preprocesado de imagen ────────────────────────────────────────────────────
def preprocess_face(face_img):
    """Convierte el ROI del rostro al tensor que espera el modelo."""
    img = cv2.resize(face_img, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # el modelo FER trabaja en escala de grises
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))        # (1, 48, 48, 1)
    return img


# ── Inferencia ────────────────────────────────────────────────────────────────
def run_inference(interpreter, face_tensor, mode):
    """Ejecuta la inferencia y devuelve (label, confidence)."""
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Algunos modelos Edge TPU esperan uint8
    if input_details[0]["dtype"] == np.uint8:
        scale, zero_point = input_details[0]["quantization"]
        face_tensor = (face_tensor / scale + zero_point).astype(np.uint8)

    interpreter.set_tensor(input_details[0]["index"], face_tensor)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]
    # output shape: (num_classes,) con probabilidades o logits
    if output.dtype == np.uint8:
        # dequantizar si es necesario
        scale, zero_point = output_details[0]["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale

    # softmax si el modelo devuelve logits (sin activación final)
    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    probs = softmax(output) if output.max() > 1 else output
    idx   = int(np.argmax(probs))
    label = EMOTION_LABELS[idx] if idx < len(EMOTION_LABELS) else "neutral"
    conf  = float(probs[idx])
    return label, conf


# ── Demo mode (sin modelo real) ───────────────────────────────────────────────
_demo_emotions = ["feliz", "neutral", "triste", "sorprendido", "enojado"]
_demo_idx      = 0
_demo_last     = 0

def demo_inference():
    """Rota emociones ficticias cada 3 segundos para demostración sin hardware."""
    global _demo_idx, _demo_last
    now = time.time()
    if now - _demo_last > 3:
        _demo_idx  = (_demo_idx + 1) % len(_demo_emotions)
        _demo_last = now
    label = _demo_emotions[_demo_idx]
    conf  = round(0.75 + 0.24 * abs(np.sin(now)), 4)
    return label, conf


# ── Hilo principal de captura ─────────────────────────────────────────────────
def capture_loop():
    global state

    interpreter, mode = load_model()
    tpu_label = "Conectado" if mode == "tpu" else ("CPU" if mode == "cpu" else "Demo")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap          = None

    if mode != "demo":
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            logging.error("No se pudo abrir la cámara. Cambiando a modo demo.")
            mode      = "demo"
            tpu_label = "Demo"
            cap       = None

    frame_count = 0
    fps_timer   = time.time()
    fps_val     = 0

    logging.info(f"🎥 Captura iniciada — modo: {mode}")

    while True:
        try:
            # ── FPS ──
            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_val    = frame_count
                frame_count = 0
                fps_timer  = now

            # ── Demo sin cámara ──
            if mode == "demo" or cap is None:
                label, conf = demo_inference()
                with state_lock:
                    state.update({
                        "emotion":    label,
                        "confidence": conf,
                        "fps":        fps_val or 24,
                        "tpu":        tpu_label,
                        "face_count": 1,
                    })
                time.sleep(0.1)
                continue

            # ── Captura real ──
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Saltar frames para no saturar la CPU
            if frame_count % FRAME_SKIP != 0:
                continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )

            if len(faces) == 0:
                with state_lock:
                    state.update({
                        "emotion":    None,
                        "confidence": 0.0,
                        "fps":        fps_val,
                        "tpu":        tpu_label,
                        "face_count": 0,
                    })
                continue

            # Tomar el rostro más grande
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            roi = frame[y:y+h, x:x+w]

            # Inferencia
            if interpreter is not None:
                tensor         = preprocess_face(roi)
                label, conf    = run_inference(interpreter, tensor, mode)
            else:
                label, conf = demo_inference()

            # Descartar baja confianza
            if conf < CONFIDENCE_MIN:
                label = None

            with state_lock:
                state.update({
                    "emotion":    label,
                    "confidence": round(conf, 4),
                    "fps":        fps_val,
                    "tpu":        tpu_label,
                    "face_count": len(faces),
                })

            if label:
                save_detection(label, conf)

        except Exception as e:
            logging.error(f"Error en capture_loop: {e}")
            time.sleep(0.5)


# ── Endpoints Flask ───────────────────────────────────────────────────────────
@app.route("/emotion")
def emotion():
    """
    Endpoint principal que consume el frontend existente.
    Devuelve exactamente el JSON que espera emotionai_web.html:
      { emotion, confidence, fps, tpu }
    """
    with state_lock:
        snap = dict(state)

    return jsonify({
        "emotion":    snap["emotion"] or "",   # "" → frontend limpia la UI
        "confidence": snap["confidence"],
        "fps":        snap["fps"],
        "tpu":        snap["tpu"],
        "face_count": snap["face_count"],
    })


@app.route("/history")
def history():
    """Devuelve las últimas 50 detecciones del historial SQLite."""
    from database import get_history
    rows = get_history(limit=50)
    return jsonify(rows)


@app.route("/stats")
def stats():
    """Resumen de emociones detectadas (para futuro dashboard)."""
    from database import get_stats
    return jsonify(get_stats())


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ── Arranque ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()

    # Hilo de captura en background
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    logging.info("🚀 EmotionAI Flask corriendo en http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
