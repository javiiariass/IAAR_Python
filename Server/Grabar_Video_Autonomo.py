import cv2
import numpy as np
import time
import os
import random
import threading
import struct
from datetime import datetime
from server import TankServer

from camera import Camera
from motor import tankMotor
from servo import Servo
from ultrasonic import Ultrasonic

# ==========================================
# FUNCIONES DE MOVIMIENTO REACTIVO
# ==========================================
motor = tankMotor()
servo_obj = Servo()

def levantar_gancho():
    servo_obj.setServoAngle('1', 140)
    time.sleep(0.5)

def detener():
    motor.setMotorModel(0, 0)

def avanzar(velocidad=1000):
    factor_correccion = 1.2
    motor.setMotorModel(-velocidad, -velocidad*factor_correccion)

def girar_aleatorio(tiempo_min=0.5, tiempo_max=1.5, retroceder=True, velocidad_giro=1500):
    velocidad_retroceso = 1200
    if retroceder:
        motor.setMotorModel(velocidad_retroceso, velocidad_retroceso)
        time.sleep(0.3)
        detener()
        time.sleep(0.1)

    direccion = random.choice(["izquierda", "derecha"])
    if direccion == "derecha":
        motor.setMotorModel(-velocidad_giro, velocidad_giro)
    else:
        motor.setMotorModel(velocidad_giro, -velocidad_giro)

    tiempo_giro = random.uniform(tiempo_min, tiempo_max)
    time.sleep(tiempo_giro)
    detener()

def evaluar_linea_reactiva(frame):
    altura, anchura = frame.shape[:2]
    roi = frame[int(altura * 2 / 3):altura, 0:anchura]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    verde_bajo = np.array([40, 50, 50])
    verde_alto = np.array([85, 255, 255])
    mascara = cv2.inRange(hsv, verde_bajo, verde_alto)
    
    pixeles_verdes = cv2.countNonZero(mascara)
    if pixeles_verdes > 3000:
        return True, pixeles_verdes
    return False, pixeles_verdes

def main():
    print("==================================================")
    print(" GRABACIÓN DE VÍDEO CON NAVEGACIÓN AUTÓNOMA ")
    print("==================================================")
    print("1. El robot conduce solo y GRABA TODO EN VÍDEO AVI.")
    print("2. Abre la app de Freenove (Client) en tu PC y conecta a la IP para ver el vídeo.")
    print("3. Escribe 'q' y pulsa Enter para salir y GUARDAR EL VÍDEO correctamente.\n")

    tcp_server = TankServer()
    tcp_server.startTcpServer()
    
    ancho_frame, alto_frame = 320, 240
    cap = Camera(stream_size=(ancho_frame, alto_frame), hflip=True, vflip=True)
    sonar = Ultrasonic()
    cap.start_stream()
    
    levantar_gancho()
    time.sleep(1)

    # Configuración de la grabación de vídeo
    directorio_videos = "videos_prueba"
    os.makedirs(directorio_videos, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_video = os.path.join(directorio_videos, f"test_yolo_{timestamp}.avi")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps_estimados = 12.0
    out_video = cv2.VideoWriter(ruta_video, fourcc, fps_estimados, (ancho_frame, alto_frame))

    print(f"Guardando vídeo en: {ruta_video}")

    estado = {"corriendo": True}

    def hilo_conduccion():
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
            
            # GUARDAR EL FRAME EN EL VÍDEO (.avi)
            out_video.write(frame)
            
            # --- COMPORTAMIENTO REACTIVO ---
            distancia = sonar.get_distance()
            if distancia < 0: distancia = 999.0
            limite_detectado, cantidad_pixeles = evaluar_linea_reactiva(frame)

            estado_sonar_str = f"| Distancia: {distancia:5.1f}cm" if distancia != 999.0 else "| Distancia: Error"
            print(f"[GRABANDO] Px verdes: {cantidad_pixeles} {estado_sonar_str}      ", end="\r")

            if limite_detectado or (0 <= distancia <= 20):
                detener()
                time.sleep(0.3)
                girar_aleatorio(retroceder=False)
                detener()
                time.sleep(0.2)
                
            elif 20 < distancia <= 40:
                girar_aleatorio(tiempo_min=0.2, tiempo_max=0.4, retroceder=False, velocidad_giro=1500)
                detener()
                time.sleep(0.1)

            else:
                avanzar()

    t = threading.Thread(target=hilo_conduccion)
    t.start()

    try:
        while True:
            val = input("")
            if val.lower() == 'q':
                estado["corriendo"] = False
                break
    except KeyboardInterrupt:
        estado["corriendo"] = False
    finally:
        estado["corriendo"] = False
        tcp_server.stopTcpServer()
        t.join(timeout=1.0)
        detener()
        out_video.release() # CRÍTICO: Cierra y finaliza el archivo de vídeo
        motor.close()
        servo_obj.setServoStop()
        sonar.close()
        cap.stop_stream()
        cap.close()
        print(f"\n¡Grabación finalizada correctamente! VÍDEO: {ruta_video}")

if __name__ == '__main__':
    main()