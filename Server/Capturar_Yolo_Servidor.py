import sys
import cv2
import numpy as np
import time
import os

import sys
import cv2
import numpy as np
import time
import os
import struct
import threading

from server import TankServer
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
# SERVIDOR COMPATIBLE CON CLIENTE WINDOWS Y SSH
# ==========================================

class CapturarYoloServidorSSH:
    def __init__(self):
        self.tcp_server = TankServer()
        self.camera = Camera(stream_size=(400, 300), hflip=True, vflip=True)
        self.video_thread = None
        self.video_thread_is_running = False
        self.current_frame = None

    def start(self):
        self.tcp_server.startTcpServer()
        
        self.video_thread_is_running = True
        self.video_thread = threading.Thread(target=self.threading_video_send)
        self.video_thread.start()
        
        print("\n\n====== GENERADOR DE YOLO VÍA SSH Y CLIENTE DE WINDOWS ======")
        print("1. Abre la aplicación de Windows de Freenove (Tank.exe / Client).")
        print("2. Escribe la IP de la Raspberry Pi y pulsa Connect para ver el vídeo.")
        print("3. Para HACER UNA FOTO Y ETIQUETAR, simplemente pulsa Enter aquí en el terminal SSH.")
        print("4. Escribe 'q' y pulsa Enter para salir del programa.")
        print("============================================================\n")
        
        try:
            while True:
                # Se queda esperando cualquier cosa escrita por SSH (o un simple Enter)
                val = input("")
                if val.lower() == 'q':
                    break
                else:
                    self.procesar_captura()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def procesar_captura(self):
        if self.current_frame is None:
            print("Esperando a que la cámara arranque...")
            return

        print("Capturando imagen y detectando clases...")
        detecciones = []
        frame_capturado = self.current_frame.copy()

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

    def threading_video_send(self):
        self.camera.start_stream()
        time.sleep(1) # Tiempo de calentamiento de cámara
        while self.video_thread_is_running:
            frame_bytes = self.camera.get_frame()
            if frame_bytes is None:
                time.sleep(0.01)
                continue
                
            # Guardamos el frame más reciente para cuando el usuario pulse Enter en SSH
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            self.current_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Si la app cliente de Windows está conectada (para que tengas streaming en vivo en el PC)
            if self.tcp_server.isVideoServerConnected():
                lenFrame = len(frame_bytes)
                lengthBin = struct.pack('<I', lenFrame)
                try:
                    self.tcp_server.sendDataToVideoClient(lengthBin)
                    self.tcp_server.sendDataToVideoClient(frame_bytes)
                except Exception as e:
                    pass

            time.sleep(0.03)

    def stop(self):
        print("Cerrando el servidor de captura...")
        self.video_thread_is_running = False
        self.tcp_server.stopTcpServer()
        if self.video_thread:
            self.video_thread.join(timeout=1.0)
        self.camera.stop_stream()
        self.camera.close()

if __name__ == '__main__':
    app = CapturarYoloServidorSSH()
    app.start()
