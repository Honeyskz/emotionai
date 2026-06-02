# EmotionAI Web 🧠

**Detector inteligente de emociones en tiempo real**  
Raspberry Pi 4 · OpenCV · TensorFlow Lite · Google Coral USB Accelerator · Flask

---

## Estructura del proyecto

```
emotionai/
├── app.py                  ← Backend Flask (OpenCV + TFLite + Coral)
├── database.py             ← Historial emocional con SQLite
├── requirements.txt        ← Dependencias Python
├── install.sh              ← Instalador automático para Raspberry Pi
├── emotionai_web.html      ← Frontend (abrir en cualquier navegador de la red)
├── models/
│   ├── emociones_edgetpu.tflite   ← Modelo para Coral TPU
│   └── emociones.tflite           ← Fallback para CPU
└── emotionai.db            ← Base de datos SQLite (se crea automáticamente)
```

---

## Requisitos de hardware

| Componente             | Detalle                          |
|------------------------|----------------------------------|
| Raspberry Pi 4         | 4 GB RAM recomendado             |
| RB Cam / USB Cam       | Compatible con OpenCV (V4L2)     |
| Google Coral USB       | Acelerador Edge TPU              |
| Fuente de alimentación | 5V / 3A mínimo                   |

---

## Instalación rápida (Raspberry Pi OS)

```bash
# 1. Cloná o copiá el proyecto
cd ~
git clone <url-del-repo> emotionai
cd emotionai

# 2. Ejecutá el instalador
chmod +x install.sh
./install.sh
```

El script instala automáticamente:
- Dependencias del sistema (`libopencv`, `libatlas`, etc.)
- Python virtual environment
- TFLite runtime
- Flask + Flask-CORS + NumPy
- libedgetpu + pycoral (Coral TPU)
- Servicio systemd para arranque automático

---

## Instalación manual paso a paso

```bash
# Dependencias del sistema
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libopencv-dev python3-opencv \
    libatlas-base-dev git curl wget

# Entorno virtual
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Dependencias Python
pip install flask flask-cors numpy opencv-python-headless tflite-runtime Pillow

# Coral TPU (requiere internet)
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std python3-pycoral
```

---

## Modelo TFLite

El sistema busca los modelos en este orden:

1. `models/emociones_edgetpu.tflite` — compilado para Edge TPU
2. `models/emociones.tflite` — TFLite estándar para CPU
3. **Modo Demo** — rota emociones ficticias (sin modelo, para pruebas de UI)

### Dónde obtener un modelo

**Opción A — Modelo preentrenado FER2013:**
```bash
# Descargar modelo CPU (ejemplo)
wget -O models/emociones.tflite \
  https://raw.githubusercontent.com/niconielsen32/EmotionDetection/main/model/model.tflite
```

**Opción B — Compilar para Edge TPU:**
```bash
# Requiere Edge TPU Compiler instalado en tu PC (no en la Raspberry Pi)
edgetpu_compiler models/emociones.tflite
# Genera: emociones_edgetpu.tflite → moverlo a la Raspberry Pi
```

### Clases del modelo

El orden de las clases en `EMOTION_LABELS` dentro de `app.py` debe coincidir  
con el modelo que uses. El orden por defecto es FER2013:

```python
EMOTION_LABELS = ["enojado", "sorprendido", "neutral", "feliz", "triste"]
```

Si tu modelo tiene otro orden, modificá esa lista en `app.py`.

---

## Uso

```bash
# Activar entorno virtual
source venv/bin/activate

# Iniciar servidor
python app.py
```

El servidor arranca en `http://0.0.0.0:5000`.

**Abrir el frontend:**
1. Abrí `emotionai_web.html` en cualquier navegador (PC, tablet, TV)
2. En el campo **IP Raspberry Pi**, escribí la IP de tu Raspberry Pi
3. La interfaz se actualiza automáticamente cada 700 ms

---

## Endpoints de la API

| Endpoint     | Descripción                                      |
|--------------|--------------------------------------------------|
| `GET /emotion` | Emoción actual, confianza, FPS, estado TPU     |
| `GET /history` | Últimas 50 detecciones guardadas en SQLite     |
| `GET /stats`   | Conteo por emoción en las últimas 24 horas     |
| `GET /health`  | Estado del servidor                            |

### Ejemplo de respuesta `/emotion`

```json
{
  "emotion":    "feliz",
  "confidence": 0.9312,
  "fps":        28,
  "tpu":        "Conectado",
  "face_count": 1
}
```

---

## Historial SQLite

Las detecciones se guardan automáticamente en `emotionai.db` (máximo 1 por segundo).

```bash
# Ver historial directamente
sqlite3 emotionai.db "SELECT * FROM detections ORDER BY timestamp DESC LIMIT 20;"

# Limpiar registros > 7 días
python3 -c "from database import purge_old; purge_old(7)"
```

---

## Arranque automático con systemd

```bash
# Habilitar (hecho por install.sh)
sudo systemctl enable emotionai

# Comandos útiles
sudo systemctl start emotionai
sudo systemctl stop emotionai
sudo systemctl restart emotionai
journalctl -u emotionai -f        # ver logs en tiempo real
```

---

## Modos de operación

El servidor detecta automáticamente el hardware disponible:

| Modo   | Condición                          | `tpu` en JSON    |
|--------|------------------------------------|------------------|
| **TPU**  | Coral + pycoral disponibles      | `"Conectado"`    |
| **CPU**  | Solo TFLite sin Coral            | `"CPU"`          |
| **Demo** | Sin modelo ni cámara             | `"Demo"`         |

En **Modo Demo** las emociones rotan ficticias cada 3 segundos.  
Ideal para probar la interfaz sin hardware.

---

## Solución de problemas

| Síntoma | Causa probable | Solución |
|---------|---------------|----------|
| `Sin conexión` en el frontend | Flask no está corriendo | `python app.py` |
| `TPU desconectado` | Coral no detectado | Verificar USB, `lsusb \| grep Google` |
| Cámara no abre | Índice incorrecto | Cambiar `CAMERA_INDEX = 1` en `app.py` |
| FPS muy bajo | FRAME_SKIP bajo | Subir `FRAME_SKIP = 4` en `app.py` |
| Detección errática | Umbral muy bajo | Subir `CONFIDENCE_MIN = 0.5` |
| `ModuleNotFoundError: pycoral` | pycoral no instalado | Correr `install.sh` o instalar manualmente |

---

## Personalización

Todas las constantes ajustables están al inicio de `app.py`:

```python
MODEL_PATH      = "models/emociones_edgetpu.tflite"
CAMERA_INDEX    = 0        # cambiar si usás otra cámara
INPUT_SIZE      = (48, 48) # depende del modelo
CONFIDENCE_MIN  = 0.30     # umbral mínimo de confianza
FRAME_SKIP      = 2        # 1 = procesar todos, 4 = más liviano
EMOTION_LABELS  = [...]    # orden de clases de tu modelo
```

---

## Mejoras futuras contempladas

- [ ] WebSocket en lugar de polling (menor latencia)
- [ ] Dashboard con gráfico de historial emocional
- [ ] Detección múltiple de rostros
- [ ] Entrenamiento de modelo personalizado
- [ ] Exportación de historial a CSV

---

*Proyecto desarrollado para exposición técnica escolar.*  
*Raspberry Pi 4 · OpenCV · TFLite · Coral USB Accelerator · Flask*
