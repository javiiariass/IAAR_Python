"""
yolo_inferencia.py — Módulo de detección YOLOv8n para la Raspberry Pi.

Carga un modelo YOLOv8n exportado a ONNX y ejecuta inferencia usando OpenCV DNN.
No requiere PyTorch ni ultralytics instalados en la RPi.

Clases:
  0 = bola_roja
  1 = linea_verde

Formato de salida de YOLOv8 ONNX:
  - Tensor de forma [1, 6, 8400]
    · Eje 0: batch (siempre 1)
    · Eje 1: 4 coordenadas (cx, cy, w, h) + 2 scores de clase (bola_roja, linea_verde)
    · Eje 2: 8400 candidatos (las "cajas" que el modelo propone)
  - Tras transponer queda [8400, 6]: cada fila es un candidato con formato:
    [cx, cy, w, h, score_bola_roja, score_linea_verde]
  - A diferencia de YOLOv5, NO hay columna separada de obj_conf;
    los scores de clase ya son la confianza directa.
"""
import cv2
import numpy as np


class YOLODetector:
    def __init__(self, model_path="best.onnx", conf_threshold=0.45, iou_threshold=0.45, img_size=320):
        """
        Args:
            model_path: Ruta al archivo .onnx exportado desde YOLOv8.
            conf_threshold: Umbral mínimo de confianza para aceptar una detección.
            iou_threshold: Umbral de IoU para Non-Maximum Suppression (NMS).
                           NMS descarta cajas duplicadas que se solapan mucho,
                           quedándose solo con la de mayor confianza.
            img_size: Tamaño de imagen de entrada al modelo.
                      Debe coincidir con el --imgsz usado en el entrenamiento.
        """
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.classes = ["bola_roja", "linea_verde"]

    def detect(self, frame):
        """
        Ejecuta detección YOLOv8 sobre un frame BGR de OpenCV.

        El proceso completo es:
        1. Redimensionar la imagen al tamaño que espera el modelo (320x320)
        2. Normalizar los píxeles de 0-255 a 0.0-1.0
        3. Pasar la imagen por la red neuronal (inferencia)
        4. Interpretar la tabla de salida: filtrar por confianza
        5. Aplicar NMS para eliminar detecciones duplicadas
        6. Convertir las coordenadas al tamaño de la imagen original

        Args:
            frame: imagen BGR (numpy array), normalmente 320x240 de la cámara.

        Returns:
            Lista de detecciones, cada una es una tupla:
            (class_id, class_name, confidence, x, y, w, h)
            donde x, y es la esquina superior-izquierda del rectángulo
            y w, h son ancho y alto, todo en píxeles de la imagen original.
        """
        h_orig, w_orig = frame.shape[:2]

        # --- PASO 1-2: Preprocesar imagen ---
        # blobFromImage hace resize a (img_size x img_size), divide por 255,
        # y convierte de BGR a RGB (swapRB=True)
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self.img_size, self.img_size),
            swapRB=True, crop=False
        )

        # --- PASO 3: Inferencia ---
        self.net.setInput(blob)
        outputs = self.net.forward()
        # outputs tiene forma [1, 6, 8400]

        # --- PASO 4: Interpretar la salida ---
        # Sacamos el batch (índice 0) y transponemos:
        #   de [6, 8400] → a [8400, 6]
        # Ahora cada fila es: [cx, cy, w, h, score_bola, score_linea]
        outputs = outputs[0].T  # T = transpose

        # Factores para convertir coordenadas del modelo (320x320) a la imagen real
        x_scale = w_orig / self.img_size
        y_scale = h_orig / self.img_size

        boxes = []
        confidences = []
        class_ids = []

        for row in outputs:
            # Las primeras 4 columnas son la posición y tamaño de la caja
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]

            # Las columnas restantes son los scores de cada clase
            class_scores = row[4:]  # [score_bola_roja, score_linea_verde]

            # La clase predicha es la de mayor score
            class_id = np.argmax(class_scores)
            confidence = float(class_scores[class_id])

            # Descartar si la confianza es baja
            if confidence < self.conf_threshold:
                continue

            # Convertir de (centro_x, centro_y, ancho, alto) a (esquina_x, esquina_y, ancho, alto)
            # y escalar a las dimensiones de la imagen original
            x = int((cx - bw / 2) * x_scale)
            y = int((cy - bh / 2) * y_scale)
            w = int(bw * x_scale)
            h = int(bh * y_scale)

            # Asegurarnos de que no se sale de la imagen
            x = max(0, x)
            y = max(0, y)

            boxes.append([x, y, w, h])
            confidences.append(confidence)
            class_ids.append(int(class_id))

        # --- PASO 5: Non-Maximum Suppression (NMS) ---
        # Si el modelo detecta la misma bola 3 veces con cajas muy parecidas,
        # NMS se queda solo con la mejor y descarta las duplicadas.
        results = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
            if len(indices) > 0:
                for i in indices.flatten():
                    results.append((
                        class_ids[i],
                        self.classes[class_ids[i]],
                        confidences[i],
                        boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                    ))
        return results

    def get_ball_info(self, detections, frame_width):
        """
        De todas las bolas detectadas, devuelve info de la MÁS GRANDE
        (que normalmente es la más cercana al robot).

        Args:
            detections: lista devuelta por detect()
            frame_width: ancho de la imagen en píxeles (320)

        Returns:
            (encontrada, centro_x_normalizado, area_relativa)
            - encontrada: True si hay al menos una bola
            - centro_x_normalizado: posición horizontal de la bola
              0.0 = borde izquierdo, 0.5 = centro, 1.0 = borde derecho
              → Esto se usa para saber hacia dónde girar
            - area_relativa: qué porción del frame ocupa la bola
              → Cuanto mayor, más cerca está
        """
        balls = [d for d in detections if d[0] == 0]  # clase 0 = bola_roja
        if not balls:
            return False, 0.5, 0.0

        # Seleccionar la bola más grande (w * h mayor)
        biggest = max(balls, key=lambda d: d[5] * d[6])
        _, _, conf, x, y, w, h = biggest

        centro_x = (x + w / 2.0) / frame_width
        area = (w * h) / (frame_width * frame_width)

        return True, centro_x, area

    def get_line_info(self, detections, frame_width, frame_height):
        """
        Analiza las líneas verdes detectadas y devuelve información
        sobre su posición para decidir cómo evadir.

        Solo consideramos "peligro" si la línea está en el tercio inferior
        del frame (zona cercana al robot). Las líneas que se ven lejos
        (tercio superior/medio) se ignoran para no frenar innecesariamente.

        Args:
            detections: lista devuelta por detect()
            frame_width: ancho de la imagen en píxeles (320)
            frame_height: alto de la imagen en píxeles (240)

        Returns:
            dict con:
            - "detectada": bool, si hay al menos una línea visible
            - "peligro": bool, si alguna línea está en el tercio inferior
            - "posicion_x": float 0.0-1.0, centro X promedio de las líneas
              peligrosas (0.0=izquierda, 0.5=centro, 1.0=derecha)
              → Se usa para girar en la dirección OPUESTA
            - "posicion_y": float 0.0-1.0, centro Y más bajo (más cercano)
              de las líneas peligrosas
            - "esquina": bool, True si hay líneas peligrosas tanto a la
              izquierda como a la derecha (el robot está en una esquina)
            - "num_peligrosas": int, cuántas líneas en zona de peligro
        """
        lines = [d for d in detections if d[0] == 1]  # clase 1 = linea_verde
        if not lines:
            return {
                "detectada": False, "peligro": False,
                "posicion_x": 0.5, "posicion_y": 0.0,
                "esquina": False, "num_peligrosas": 0
            }

        # Filtrar solo las líneas en el tercio inferior (zona de peligro)
        umbral_y = frame_height * 2 / 3
        peligrosas = []
        for _, _, conf, x, y, w, h in lines:
            centro_y = y + h / 2
            if centro_y > umbral_y:
                centro_x = (x + w / 2.0) / frame_width
                peligrosas.append((centro_x, centro_y / frame_height))

        if not peligrosas:
            return {
                "detectada": True, "peligro": False,
                "posicion_x": 0.5, "posicion_y": 0.0,
                "esquina": False, "num_peligrosas": 0
            }

        # Calcular posición promedio X de las líneas peligrosas
        avg_x = sum(p[0] for p in peligrosas) / len(peligrosas)
        # Posición Y más baja (más cercana al robot)
        max_y = max(p[1] for p in peligrosas)

        # Detectar esquina: hay líneas peligrosas a ambos lados
        hay_izquierda = any(p[0] < 0.35 for p in peligrosas)
        hay_derecha = any(p[0] > 0.65 for p in peligrosas)
        esquina = hay_izquierda and hay_derecha

        return {
            "detectada": True, "peligro": True,
            "posicion_x": avg_x, "posicion_y": max_y,
            "esquina": esquina, "num_peligrosas": len(peligrosas)
        }


# ================================================================
# TEST — Ejecutar directamente en la RPi para probar la inferencia
# Uso: sudo python yolo_inferencia.py
# ================================================================
if __name__ == '__main__':
    import time
    from camera import Camera

    MODEL = "best.onnx"
    print(f"Cargando modelo: {MODEL}")
    detector = YOLODetector(MODEL, conf_threshold=0.40)

    cap = Camera(stream_size=(320, 240), hflip=True, vflip=True)
    cap.start_stream()
    time.sleep(1)
    print("Cámara lista. Ctrl+C para salir.\n")

    try:
        while True:
            frame_bytes = cap.get_frame()
            if frame_bytes is None:
                continue

            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            t0 = time.time()
            dets = detector.detect(frame)
            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0

            if dets:
                for d in dets:
                    print(f"  {d[1]}: {d[2]:.2f} @ ({d[3]},{d[4]},{d[5]},{d[6]})")

            ball_found, ball_cx, ball_area = detector.get_ball_info(dets, frame.shape[1])
            line_det, line_danger = detector.get_line_detected(dets, frame.shape[0])

            status = ""
            if ball_found:
                status += f"BOLA cx={ball_cx:.2f} area={ball_area:.3f} | "
            if line_det:
                status += f"LINEA {'PELIGRO' if line_danger else 'lejos'} | "

            print(f"\rFPS: {fps:.1f} | {status}Detecciones: {len(dets)}          ", end="")

    except KeyboardInterrupt:
        print("\nCerrando...")
    finally:
        cap.stop_stream()
        cap.close()
        print("Fin del test.")
