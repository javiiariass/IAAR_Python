"""
yolo_inferencia_ncnn.py — Módulo de detección YOLOv8n usando NCNN.

Carga un modelo YOLOv8n exportado a NCNN y ejecuta inferencia.
NCNN es un framework optimizado para ARM (Raspberry Pi) y suele ser
más rápido que ONNX+OpenCV DNN en este tipo de hardware.

Misma interfaz que yolo_inferencia.py (YOLODetector) para que sean
intercambiables. Solo cambia el motor de inferencia por debajo.

Requisitos en la RPi:
  pip install ncnn --break-system-packages

Archivos del modelo (generados al exportar desde Colab):
  best_ncnn_model/model.ncnn.param   (arquitectura de la red)
  best_ncnn_model/model.ncnn.bin     (pesos)

Clases:
  0 = bola_roja
  1 = linea_verde

Formato de salida de YOLOv8 NCNN:
  - Mismo que ONNX: tensor [1, 6, 8400]
  - Tras transponer: [8400, 6] → cada fila: [cx, cy, w, h, score_bola, score_linea]
"""
import cv2
import numpy as np

try:
    import ncnn
    NCNN_DISPONIBLE = True
except ImportError:
    NCNN_DISPONIBLE = False


class YOLODetectorNCNN:
    def __init__(self, model_dir="best_ncnn_model", conf_threshold=0.45,
                 iou_threshold=0.45, img_size=320):
        """
        Args:
            model_dir: Carpeta con best.ncnn.param y best.ncnn.bin
            conf_threshold: Umbral mínimo de confianza para aceptar una detección.
            iou_threshold: Umbral de IoU para Non-Maximum Suppression (NMS).
            img_size: Tamaño de imagen de entrada al modelo (debe coincidir
                      con el --imgsz usado en el entrenamiento).
        """
        if not NCNN_DISPONIBLE:
            raise ImportError(
                "La librería 'ncnn' no está instalada.\n"
                "Instálala con: pip install ncnn --break-system-packages"
            )

        param_path = f"{model_dir}/model.ncnn.param"
        bin_path = f"{model_dir}/model.ncnn.bin"

        self.net = ncnn.Net()
        # Usar todos los cores disponibles en la RPi
        self.net.opt.num_threads = 4
        self.net.opt.use_vulkan_compute = False  # Sin GPU en la RPi

        self.net.load_param(param_path)
        self.net.load_model(bin_path)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.classes = ["bola_roja", "linea_verde"]

    def detect(self, frame):
        """
        Ejecuta detección YOLOv8 sobre un frame BGR de OpenCV.

        El proceso es el mismo que con ONNX:
        1. Redimensionar la imagen a img_size x img_size
        2. Normalizar píxeles de 0-255 a 0.0-1.0
        3. Pasar por la red neuronal (inferencia con NCNN)
        4. Interpretar la salida: filtrar por confianza
        5. Aplicar NMS para eliminar detecciones duplicadas
        6. Convertir coordenadas al tamaño de la imagen original

        Args:
            frame: imagen BGR (numpy array), normalmente 320x240.

        Returns:
            Lista de detecciones, cada una es una tupla:
            (class_id, class_name, confidence, x, y, w, h)
            Mismo formato que yolo_inferencia.py para compatibilidad.
        """
        h_orig, w_orig = frame.shape[:2]

        # --- PASO 1-2: Preprocesar imagen ---
        # ncnn.Mat.from_pixels_resize hace resize y convierte formato
        # NCNN espera RGB, así que convertimos de BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mat_in = ncnn.Mat.from_pixels_resize(
            frame_rgb, ncnn.Mat.PixelType.PIXEL_RGB,
            w_orig, h_orig, self.img_size, self.img_size
        )

        # Normalizar: dividir por 255 (equivalente a 1/255.0 por canal)
        norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
        mat_in.substract_mean_normalize([], norm_vals)

        # --- PASO 3: Inferencia ---
        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        ret, mat_out = ex.extract("out0")

        # mat_out es la salida del modelo.
        # Convertir a numpy: la forma será [6, 8400] (sin batch)
        # (NCNN no incluye la dimensión batch como ONNX)
        output = np.array(mat_out)

        # Transponer de [6, 8400] a [8400, 6]
        # Cada fila: [cx, cy, w, h, score_bola_roja, score_linea_verde]
        if output.ndim == 3:
            output = output[0]  # Quitar batch si lo hubiera
        outputs = output.T

        # --- PASO 4: Interpretar la salida (VECTORIZADO) ---
        data = outputs  # shape [8400, 6]

        # Extraer scores de clase (columnas 4 y 5) de golpe
        class_scores = data[:, 4:]                         # [8400, 2]
        class_ids_all = np.argmax(class_scores, axis=1)    # [8400]
        confidences_all = np.max(class_scores, axis=1)     # [8400]

        # Filtrar por confianza — máscara booleana, sin bucle
        mask = confidences_all > self.conf_threshold
        if not np.any(mask):
            return []

        # Quedarnos solo con los candidatos que pasan el filtro
        filtered = data[mask]                    # [N, 6]  N << 8400
        class_ids = class_ids_all[mask]          # [N]
        confidences = confidences_all[mask]      # [N]

        # Convertir de (cx, cy, w, h) → (x, y, w, h) en coordenadas originales
        x_scale = w_orig / self.img_size
        y_scale = h_orig / self.img_size

        cx = filtered[:, 0]
        cy = filtered[:, 1]
        bw = filtered[:, 2]
        bh = filtered[:, 3]

        x = ((cx - bw / 2) * x_scale).astype(np.int32)
        y = ((cy - bh / 2) * y_scale).astype(np.int32)
        w = (bw * x_scale).astype(np.int32)
        h = (bh * y_scale).astype(np.int32)

        # Clamp a >= 0
        np.maximum(x, 0, out=x)
        np.maximum(y, 0, out=y)

        # Preparar listas para NMS
        boxes = np.stack([x, y, w, h], axis=1).tolist()
        confs_list = confidences.tolist()
        ids_list = class_ids.astype(int).tolist()

        # --- PASO 5: Non-Maximum Suppression (NMS) ---
        results = []
        indices = cv2.dnn.NMSBoxes(boxes, confs_list,
                                   self.conf_threshold, self.iou_threshold)
        if len(indices) > 0:
            for i in indices.flatten():
                results.append((
                    ids_list[i],
                    self.classes[ids_list[i]],
                    confs_list[i],
                    boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                ))
        return results

    def get_ball_info(self, detections, frame_width):
        """
        De todas las bolas detectadas, devuelve info de la MÁS GRANDE
        (que normalmente es la más cercana al robot).

        Returns:
            (encontrada, centro_x_normalizado, area_relativa)
        """
        balls = [d for d in detections if d[0] == 0]
        if not balls:
            return False, 0.5, 0.0

        biggest = max(balls, key=lambda d: d[5] * d[6])
        _, _, conf, x, y, w, h = biggest

        centro_x = (x + w / 2.0) / frame_width
        area = (w * h) / (frame_width * frame_width)

        return True, centro_x, area

    def get_line_info(self, detections, frame_width, frame_height):
        """
        Analiza las líneas verdes detectadas y devuelve información
        sobre su posición para decidir cómo evadir.

        Solo consideramos "peligro" si la línea está en el tercio inferior.

        Returns:
            dict con: detectada, peligro, posicion_x, posicion_y, esquina, num_peligrosas
        """
        lines = [d for d in detections if d[0] == 1]
        if not lines:
            return {
                "detectada": False, "peligro": False,
                "posicion_x": 0.5, "posicion_y": 0.0,
                "esquina": False, "num_peligrosas": 0
            }

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

        avg_x = sum(p[0] for p in peligrosas) / len(peligrosas)
        max_y = max(p[1] for p in peligrosas)

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
# Uso: sudo python yolo_inferencia_ncnn.py
# ================================================================
if __name__ == '__main__':
    import time
    from camera import Camera

    MODEL_DIR = "best_ncnn_model"
    print(f"Cargando modelo NCNN desde: {MODEL_DIR}")
    detector = YOLODetectorNCNN(MODEL_DIR, conf_threshold=0.40)

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
            line_info = detector.get_line_info(dets, frame.shape[1], frame.shape[0])

            status = ""
            if ball_found:
                status += f"BOLA cx={ball_cx:.2f} area={ball_area:.3f} | "
            if line_info["detectada"]:
                peligro = "PELIGRO" if line_info["peligro"] else "lejos"
                status += f"LINEA {peligro} | "

            print(f"\rFPS: {fps:.1f} | {status}Detecciones: {len(dets)}          ", end="")

    except KeyboardInterrupt:
        print("\nCerrando...")
    finally:
        cap.stop_stream()
        cap.close()
        print("Fin del test.")
