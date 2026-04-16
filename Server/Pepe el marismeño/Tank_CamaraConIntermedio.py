import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
from libcamera import Transform
from Estado_compartido import EstadoCompartido # 🔁 Módulo de datos compartidos


def camara(estado):
# 📌 Configuración YOLO
    model_path     = "best.onnx"
    input_size     = 640
    conf_threshold = 0.1
    iou_threshold  = 0.4

    # 🧠 Cargar modelo ONNX
    session    = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # 📸 Configurar Picamera2
    transform = Transform(hflip=1, vflip=1)
    picam2    = Picamera2()
    config    = picam2.create_preview_configuration(
                    main={"size": (input_size, input_size)},
                    transform=transform
                )
    picam2.configure(config)
    picam2.start()

    try:
        while True:
            # 📷 Captura de imagen
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 🔄 Preprocesamiento
            img_norm = frame.astype(np.float32) / 255.0
            img_trans = cv2.resize(img_norm, (input_size, input_size))
            img_trans = np.transpose(img_trans, (2, 0, 1))[None, ...]
            input_tensor = np.ascontiguousarray(img_trans)

            # 🧠 Inferencia
            outputs = session.run(None, {input_name: input_tensor})
            preds = outputs[0][0]

            # 🧱 Extracción de detecciones
            boxes, scores, class_ids = [], [], []
            h, w, _ = frame.shape
            for pred in preds:
                conf = float(pred[4])
                class_scores = pred[5:]
                cid = int(np.argmax(class_scores))
                score = conf * class_scores[cid]
                if score < conf_threshold:
                    continue

                cx, cy, bw, bh = pred[:4]
                x1 = int((cx - bw/2) * w / input_size)
                y1 = int((cy - bh/2) * h / input_size)
                x2 = int((cx + bw/2) * w / input_size)
                y2 = int((cy + bh/2) * h / input_size)

                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(score)
                class_ids.append(cid)

            # 🔍 NMS y actualización del estado
            if boxes:
                indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
                for i in indices.flatten():
                    x, y, bw, bh = boxes[i]
                    cx_objeto = x + bw // 2
                    cy_objeto = y + bh // 2
                    clase = class_ids[i]
                    score = scores[i]

                    # 🔁 Actualiza estado global
                    estado.agregar(cx_objeto, cy_objeto,x1,x2,y1,y2, clase, score)
                    
                    # 🖼 Dibujar en pantalla
                    label = f"ID {clase} {score:.2f}"
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 📺 Mostrar frame
            cv2.imshow("YOLOv5 ONNX + Picamera2", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.close()
        cv2.destroyAllWindows()
