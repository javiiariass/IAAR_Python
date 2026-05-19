import cv2
import numpy as np
import time
import os
import threading
import struct
from server import TankServer

from camera import Camera
from motor import tankMotor
from servo import Servo

# ==========================================
# FUNCIONES YOLO
# ==========================================
def detectar_linea_verde_yolo(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    verde_bajo = np.array([40, 50, 50])
    verde_alto = np.array([85, 255, 255])
    mascara = cv2.inRange(hsv, verde_bajo, verde_alto)
    
    bboxes = []
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        for contorno in contornos:
            if cv2.contourArea(contorno) > 700:
                x, y, w, h = cv2.boundingRect(contorno)
                bboxes.append((x, y, w, h))
                
    return (len(bboxes) > 0), 0, bboxes

def detectar_bola_roja_yolo(frame):
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
                if 0.2 <= aspect_ratio <= 5.0:
                    bboxes.append((x, y, w, h))
    return (len(bboxes) > 0), 0, bboxes

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
# FUNCIONES DE MOVIMIENTO MANUAL
# ==========================================
motor = tankMotor()
servo_obj = Servo()

def levantar_gancho():
    servo_obj.setServoAngle('1', 140)
    time.sleep(0.5)

def detener():
    motor.setMotorModel(0, 0)

def avanzar(velocidad=1200):
    factor_correccion = 1.2
    motor.setMotorModel(-velocidad, -velocidad*factor_correccion)

def retroceder(velocidad=1200):
    factor_correccion = 1.2
    # Invertido respecto a avanzar
    motor.setMotorModel(velocidad, velocidad*factor_correccion)

def main():
    print("========================================")
    print(" CONDUCCIÓN MANUAL + TECLA CAPTURA YOLO ")
    print("========================================")
    print("1. Abre la app de Freenove (Client) en tu PC y conecta a la IP para ver el vídeo.")
    print("2. MANTEN 'w'/Enter para AVANZAR (Por consola)")
    print("3. PULSA 'd'/Enter para TOMAR FOTO")
    print("4. PULSA 'q'/Enter para SALIR")
    print("========================================")
    
    tcp_server = TankServer()
    tcp_server.startTcpServer()
    
    cap = Camera(stream_size=(320, 240), hflip=True, vflip=True)
    cap.start_stream()
    
    levantar_gancho()
    
    estado = {"corriendo": True, "comando": ""}
    
    def hilo_camara():
        while estado["corriendo"]:
            frame_bytes = cap.get_frame()
            if frame_bytes is None: 
                time.sleep(0.01)
                continue
                
            # --- TCP SERVER VIDEO STREAMING ---
            if tcp_server.isVideoServerConnected():
                lenFrame = len(frame_bytes)
                lengthBin = struct.pack('<I', lenFrame)
                try:
                    tcp_server.sendDataToVideoClient(lengthBin)
                    tcp_server.sendDataToVideoClient(frame_bytes)
                except Exception:
                    pass

            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            comando = estado["comando"]
            estado["comando"] = ""

            # 1. EVALUAR CAPTURA MANUAL
            if comando == "d":
                detecciones_manuales = []
                bola_detectada, _, bboxes_bola = detectar_bola_roja_yolo(frame)
                if bola_detectada:
                    for bbox in bboxes_bola: detecciones_manuales.append((0, bbox))
                
                linea_detectada, _, bboxes_linea = detectar_linea_verde_yolo(frame)
                if linea_detectada:
                    for bbox in bboxes_linea: detecciones_manuales.append((1, bbox))
                    
                guardar_imagen_yolo(frame, detecciones_manuales, prefix="movimiento_manual")
                print(f"[FOTO MANUAL] Captura guardada con {len(detecciones_manuales)} detecciones.      ")
            
            # 2. EVALUAR CONDUCCIÓN
            elif comando == "w":
                avanzar()
                time.sleep(0.3)
                detener()
            elif comando == "s":
                retroceder()
                time.sleep(0.3)
                detener()
                
    t = threading.Thread(target=hilo_camara)
    t.start()

    try:
        while True:
            val = input("")
            if val.lower() == 'q':
                estado["corriendo"] = False
                break
            elif val.lower() == 'd':
                 estado["comando"] = "d"
            elif val.lower() == 'w':
                 estado["comando"] = "w"
            elif val.lower() == 's':
                 estado["comando"] = "s"

    except KeyboardInterrupt:
        estado["corriendo"] = False
    finally:
        estado["corriendo"] = False
        tcp_server.stopTcpServer()
        t.join(timeout=1.0)
        detener()
        motor.close()
        servo_obj.setServoStop()
        cap.stop_stream()
        cap.close()
        print("\nRobot detenido de forma segura.")

if __name__ == '__main__':
    main()
