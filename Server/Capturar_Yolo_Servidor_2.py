import sys
import cv2
import numpy as np
import time
import os

# Importaciones de PyQt5 para la interfaz gráfica
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from camera import Camera

# ==========================================
# TUS FUNCIONES ORIGINALES INTACTAS
# ==========================================

def detectar_linea_verde(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    verde_bajo = np.array([40, 50, 50])
    verde_alto = np.array([85, 255, 255])
    mascara = cv2.inRange(hsv, verde_bajo, verde_alto)

    bboxes = []
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        for contorno in contornos:
            if cv2.contourArea(contorno) > 500:
                x, y, w, h = cv2.boundingRect(contorno)
                bboxes.append((x, y, w, h))

    return (len(bboxes) > 0), 0, bboxes

def detectar_bola_roja(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rojo_bajo1 = np.array([0, 150, 100])
    rojo_alto1 = np.array([8, 255, 255])
    rojo_bajo2 = np.array([170, 150, 100])
    rojo_alto2 = np.array([180, 255, 255])
    mascara1 = cv2.inRange(hsv, rojo_bajo1, rojo_alto1)
    mascara2 = cv2.inRange(hsv, rojo_bajo2, rojo_alto2)
    mascara_roja = cv2.add(mascara1, mascara2)
    contornos, _ = cv2.findContours(mascara_roja, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    UMBRAL_ROJO = 1500
    bboxes = []
    if contornos:
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area > UMBRAL_ROJO:
                x, y, w, h = cv2.boundingRect(contorno)
                aspect_ratio = float(w) / h
                if 0.5 <= aspect_ratio <= 2.0:
                    bboxes.append((x, y, w, h))
    if bboxes:
        return True, 0, bboxes
    return False, 0, []

def guardar_imagen_yolo(frame, detecciones, base_dir="dataset_clasificacion", prefix="manual_yolo"):
    os.makedirs(base_dir, exist_ok=True)
    idx = 0
    while True:
        nombre_base = f"{prefix}_{idx:04d}"
        ruta_imagen = os.path.join(base_dir, f"{nombre_base}.jpg")
        ruta_txt = os.path.join(base_dir, f"{nombre_base}.txt")
        if not os.path.exists(ruta_imagen) and not os.path.exists(ruta_txt):
            break
        idx += 1

    cv2.imwrite(ruta_imagen, frame)

    with open(ruta_txt, "w") as f:
        alto_img, ancho_img = frame.shape[:2]
        for class_id, bbox in detecciones:
            if bbox is not None:
                x, y, w, h = bbox
                x_centro_norm = (x + (w / 2.0)) / ancho_img
                y_centro_norm = (y + (h / 2.0)) / alto_img
                ancho_norm = w / float(ancho_img)
                alto_norm = h / float(alto_img)
                f.write(f"{class_id} {x_centro_norm:.6f} {y_centro_norm:.6f} {ancho_norm:.6f} {alto_norm:.6f}\n")

    print(f"Imagen y {len(detecciones)} anotaciones guardadas en {nombre_base}")

# ==========================================
# NUEVA INTERFAZ GRÁFICA CON PYQT5
# ==========================================

class YoloApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generador de Dataset YOLO - Tanque Freenove")
        self.resize(400, 350)

        # Variables para la cámara
        self.cap = Camera(stream_size=(320, 240), hflip=True, vflip=True)
        self.cap.start_stream()
        self.current_frame = None

        self.initUI()

        # Configurar un temporizador para actualizar el video (aprox 30 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.actualizar_video)
        self.timer.start(30)

    def initUI(self):
        # Etiqueta donde se mostrará el video
        self.label_video = QLabel(self)
        self.label_video.setFixedSize(320, 240)
        self.label_video.setStyleSheet("background-color: black;")

        # Botones
        self.btn_capturar = QPushButton("Capturar y Etiquetar (C)", self)
        self.btn_capturar.clicked.connect(self.procesar_captura)
        
        self.btn_salir = QPushButton("Salir", self)
        self.btn_salir.clicked.connect(self.close)

        # Layouts para organizar todo
        layout_botones = QHBoxLayout()
        layout_botones.addWidget(self.btn_capturar)
        layout_botones.addWidget(self.btn_salir)

        layout_principal = QVBoxLayout()
        layout_principal.addWidget(self.label_video)
        layout_principal.addLayout(layout_botones)

        self.setLayout(layout_principal)

    def actualizar_video(self):
        # Obtener frame de la cámara
        frame_bytes = self.cap.get_frame()
        if frame_bytes is None:
            return

        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Guardamos el frame original para la captura
        self.current_frame = frame.copy()

        # Convertir de BGR (OpenCV) a RGB (PyQt5)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        # Crear y mostrar la imagen en la interfaz
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(q_img))

    def procesar_captura(self):
        if self.current_frame is None:
            print("Esperando a la cámara...")
            return

        print("Capturando imagen y detectando clases...")
        detecciones = []
        frame_capturado = self.current_frame.copy()

        # Tus lógicas de detección
        bola_detectada, _, bboxes_bola = detectar_bola_roja(frame_capturado)
        if bola_detectada and bboxes_bola:
            print(f"- {len(bboxes_bola)} Pelota(s) rojas detectada(s)!")
            for bbox in bboxes_bola:
                detecciones.append((0, bbox))

        linea_detectada, _, bboxes_linea = detectar_linea_verde(frame_capturado)
        if linea_detectada and bboxes_linea:
            print(f"- {len(bboxes_linea)} Línea(s) verde(s) detectada(s)!")
            for bbox in bboxes_linea:
                detecciones.append((1, bbox))

        if not detecciones:
            print("- Ningún objeto detectado. Guardando sin anotaciones (como imagen de fondo).")

        guardar_imagen_yolo(frame_capturado, detecciones, prefix="yolo_manual")

    def closeEvent(self, event):
        # Apagar todo limpiamente al cerrar la ventana
        self.timer.stop()
        self.cap.stop_stream()
        self.cap.close()
        event.accept()

if __name__ == '__main__':
    # Necesario para que PyQt5 inicie
    app = QApplication(sys.argv)
    ventana = YoloApp()
    ventana.show()
    sys.exit(app.exec_())
