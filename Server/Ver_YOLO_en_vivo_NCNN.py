"""
Ver_YOLO_en_vivo_NCNN.py — Visualización en tiempo real de las detecciones YOLO (NCNN).

Igual que Ver_YOLO_en_vivo.py pero usa NCNN como motor de inferencia.

Instrucciones:
  1. Copiar best_ncnn_model/ a la carpeta Server/
  2. Ejecutar: sudo python Ver_YOLO_en_vivo_NCNN.py
  3. Abrir la app de Freenove en el PC, escribir la IP de la RPi y pulsar Connect
  4. Verás el vídeo con los rectángulos de detección en tiempo real
  5. Pulsar 'q' en el terminal SSH para salir
"""
import cv2
import numpy as np
import time
import struct
import threading
import sys
import tty
import termios

from server import TankServer
from camera import Camera
from yolo_inferencia_ncnn import YOLODetectorNCNN

# --- Colores BGR para dibujar ---
COLOR_BOLA = (0, 0, 255)     # Rojo
COLOR_LINEA = (0, 255, 0)    # Verde
COLOR_TEXTO = (255, 255, 255) # Blanco


def dibujar_detecciones(frame, detecciones):
    """Dibuja los bounding boxes y etiquetas sobre el frame."""
    for class_id, nombre, confianza, x, y, w, h in detecciones:
        color = COLOR_BOLA if class_id == 0 else COLOR_LINEA

        # Rectángulo
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Etiqueta con fondo
        etiqueta = f"{nombre} {confianza:.0%}"
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(frame, etiqueta, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXTO, 1)

    return frame


def leer_tecla():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    if ch == '\x03':
        raise KeyboardInterrupt
    return ch


def main():
    print("==============================================")
    print(" VISUALIZACIÓN YOLO EN TIEMPO REAL (NCNN)")
    print("==============================================")
    print("1. Abre la app Freenove en tu PC y conecta a la IP.")
    print("2. Verás el vídeo con los bounding boxes dibujados.")
    print("3. Pulsa 'q' aquí para salir.\n")

    # Inicializar componentes — NCNN en vez de ONNX
    detector = YOLODetectorNCNN("best_ncnn_model", conf_threshold=0.40)
    tcp_server = TankServer()
    tcp_server.startTcpServer()

    cap = Camera(stream_size=(320, 240), hflip=True, vflip=True)
    cap.start_stream()
    time.sleep(1)

    estado = {"corriendo": True}

    def hilo_video():
        while estado["corriendo"]:
            frame_bytes = cap.get_frame()
            if frame_bytes is None:
                time.sleep(0.01)
                continue

            # Decodificar JPEG a imagen OpenCV
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Ejecutar YOLO
            t0 = time.time()
            detecciones = detector.detect(frame)
            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0

            # Dibujar las cajas sobre el frame
            frame_anotado = dibujar_detecciones(frame, detecciones)

            # Dibujar FPS en esquina superior izquierda
            cv2.putText(frame_anotado, f"FPS: {fps:.1f}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Re-codificar a JPEG para enviar al cliente
            _, jpeg_anotado = cv2.imencode('.jpg', frame_anotado)
            frame_bytes_anotado = jpeg_anotado.tobytes()

            # Enviar al cliente de Freenove si está conectado
            if tcp_server.isVideoServerConnected():
                length_bin = struct.pack('<I', len(frame_bytes_anotado))
                try:
                    tcp_server.sendDataToVideoClient(length_bin)
                    tcp_server.sendDataToVideoClient(frame_bytes_anotado)
                except Exception:
                    pass

            # Imprimir estado en consola
            nombres = [d[1] for d in detecciones]
            print(f"\rFPS: {fps:.1f} | Detecciones: {nombres if nombres else 'ninguna'}          ", end="")

    # Lanzar hilo de vídeo
    t = threading.Thread(target=hilo_video)
    t.start()

    # Hilo principal: esperar tecla 'q' para salir
    try:
        while True:
            val = leer_tecla()
            if val.lower() == 'q':
                break
    except KeyboardInterrupt:
        pass
    finally:
        estado["corriendo"] = False
        t.join(timeout=2.0)
        tcp_server.stopTcpServer()
        cap.stop_stream()
        cap.close()
        print("\nCerrado correctamente.")


if __name__ == '__main__':
    main()
