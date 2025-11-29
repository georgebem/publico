'''
git do Yolo:
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt


conda create -n yolo11 python=3.13.9
conda activate yolo11

pip install ultralytics opencv-python


'''

import cv2
from ultralytics import YOLO

# Carregar o modelo YOLO11n (arquivo deve estar na mesma pasta)
model = YOLO("yolo11n.pt")

# Inicializar webcam (0 = webcam padrão)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

print("Pressione 'q' para encerrar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar imagem.")
        break

    # Rodar detecção com YOLO
    results = model.predict(frame, imgsz=640, conf=0.5)

    # Desenhar as detecções no frame
    annotated_frame = results[0].plot()

    # Mostrar imagem com detecções
    cv2.imshow("YOLO11n - Webcam", annotated_frame)

    # Encerrar com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
