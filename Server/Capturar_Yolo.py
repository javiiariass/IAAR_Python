import cv2
import numpy as np
import time
import os
import random

from camera import Camera
from motor import tankMotor
from servo import Servo
from ultrasonic import Ultrasonic

# ==========================================
# FUNCIONES PARA EL DATASET YOLO (TODO EL FRAME)
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
                # Ampliamos mucho la tolerancia (de 0.5-2.0 a 0.2-5.0) para detectar 
                # pelotas difuminadas/ovaladas por el movimiento de la cámara
                if 0.2 <= aspect_ratio <= 5.0:
                    bboxes.append((x, y, w, h))
    return (len(bboxes) > 0), 0, bboxes

def guardar_imagen_yolo(frame, detecciones, base_dir="dataset_clasificacion", prefix="auto_yolo"):
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
# FUNCIONES DE MOVIMIENTO (COMPORTAMIENTO REACTIVO)
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

def procesar_captura_yolo(frame_capturado, ultima_foto_time, cooldown_fotos):
    detecciones = []
    bola_detectada, _, bboxes_bola = detectar_bola_roja_yolo(frame_capturado)
    if bola_detectada:
        for bbox in bboxes_bola: detecciones.append((0, bbox))

    linea_detectada, _, bboxes_linea = detectar_linea_verde_yolo(frame_capturado)
    if linea_detectada:
        for bbox in bboxes_linea: detecciones.append((1, bbox))

    tiempo_actual = time.time()
    if detecciones and (tiempo_actual - ultima_foto_time > cooldown_fotos):
        guardar_imagen_yolo(frame_capturado, detecciones, prefix="auto_yolo")
        return tiempo_actual
    return ultima_foto_time

def main():
    print("Script (con GUI) Autónomo: Conducción + YOLO automático...")
    cap = Camera(stream_size=(320, 240), hflip=True, vflip=True)
    sonar = Ultrasonic()
    cap.start_stream()
    
    levantar_gancho()
    time.sleep(1)

    ultima_foto_time = 0
    cooldown_fotos = 2.5 # Aumentado el cooldown a 2.5 segs. para tener variedad entre fotos

    try:
        while True:
            frame_bytes = cap.get_frame()
            if frame_bytes is None: continue
            
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            cv2.imshow("Camara - Modo Autonomo YOLO ('q' para salir, 'c' foto manual)", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Captura manual: forzamos guardar SIEMPRE (incluso sin objetos para imágenes de fondo)
                detecciones_manuales = []
                bola_detectada, _, bboxes_bola = detectar_bola_roja_yolo(frame)
                if bola_detectada:
                    for bbox in bboxes_bola: detecciones_manuales.append((0, bbox))
                
                linea_detectada, _, bboxes_linea = detectar_linea_verde_yolo(frame)
                if linea_detectada:
                    for bbox in bboxes_linea: detecciones_manuales.append((1, bbox))
                    
                guardar_imagen_yolo(frame, detecciones_manuales, prefix="manual_yolo")
                print(f"\n[MANUAL] Captura de fondo/manual guardada con {len(detecciones_manuales)} detecciones.    ")
                ultima_foto_time = time.time()
            
            # 1. EVALUAR YOLO Y GUARDAR AUTOMÁTICAMENTE
            ultima_foto_time = procesar_captura_yolo(frame, ultima_foto_time, cooldown_fotos)
            
            # 2. EVALUAR PELIGRO PARA CONDUCIR (solo ROI y sonar)
            distancia = sonar.get_distance()
            if distancia < 0: distancia = 999.0
            limite_detectado, cantidad_pixeles = evaluar_linea_reactiva(frame)

            # Imprimir constantemente los píxeles (usa \r para no saturar la pantalla)
            estado_sonar_str = f"| Distancia: {distancia:5.1f}cm" if distancia != 999.0 else "| Distancia: Error"
            print(f"Px verdes: {cantidad_pixeles} (Umbral actual: 3000) {estado_sonar_str}      ", end="\r")

            if limite_detectado or (0 <= distancia <= 20):
                if limite_detectado:
                    print(f"\n¡Límite verde detectado! ({cantidad_pixeles} px) Evadiendo...     ")
                else:
                    print(f"\n¡Obstáculo inminente! ({distancia:5.1f} cm) Evadiendo...          ")
                    
                detener()
                time.sleep(0.3)
                girar_aleatorio(retroceder=False)
                detener()
                time.sleep(0.2)
                
            elif 20 < distancia <= 40:
                print(f"\nObstáculo a media distancia ({distancia:5.1f} cm). Giro ligero...    ")
                girar_aleatorio(tiempo_min=0.2, tiempo_max=0.4, retroceder=False, velocidad_giro=1500)
                detener()
                time.sleep(0.1)

            else:
                avanzar()

    finally:
        detener()
        motor.close()
        servo_obj.setServoStop()
        sonar.close()
        cap.stop_stream()
        cap.close()
        cv2.destroyAllWindows()
        print("Robot detenido de forma segura.")

if __name__ == '__main__':
    main()
