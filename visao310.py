'''
conda create -n gtm310 python=3.10
conda activate gtm310

pip install "tensorflow==2.15.0" numpy opencv-python h5py

python -c "import tensorflow as tf; print(tf.__version__); print(hasattr(tf, 'keras'))"
deve gerar a saida: 2.15.0 True

'''

import cv2
import numpy as np
import tensorflow as tf
import os

# -------------------------------------------------
# 1) Carregar modelo e labels
# -------------------------------------------------
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

print(f"Carregando modelo de: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

class_names = None
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_names = [l.strip() for l in f.readlines()]
    print(f"{len(class_names)} classes carregadas de {LABELS_PATH}")
else:
    print("labels.txt não encontrado - usarei só o índice da classe.")

# descobrir forma de entrada do modelo: (None, h, w, c)
input_shape = model.input_shape
if isinstance(input_shape, list):
    input_shape = input_shape[0]

_, h, w, c = input_shape
print(f"Entrada do modelo: altura={h}, largura={w}, canais={c}")

# -------------------------------------------------
# 2) Pré-processamento
# -------------------------------------------------
def preprocess_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_AREA)
    img = frame_resized.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)   # (1, h, w, c)
    return img

# -------------------------------------------------
# 3) Webcam
# -------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Não consegui abrir a webcam.")
    raise SystemExit(1)

print("✅ Webcam iniciada. Pressione 'q' para sair.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar frame.")
            break

        frame = cv2.flip(frame, 1)  # espelhar

        x = preprocess_frame(frame)

        preds = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))

        if class_names and 0 <= idx < len(class_names):
            label = f"{class_names[idx]} ({conf*100:.1f}%)"
        else:
            label = f"Classe {idx} ({conf*100:.1f}%)"

        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Classificacao em tempo real - 'q' para sair", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

