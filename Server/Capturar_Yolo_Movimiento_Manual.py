import cv2
import numpy as np
import time
import os

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
    print(" MANTEN 'w' para AVANZAR")
    print(" MANTEN 's' para RETROCEDER")
    print(" PULSA 'c' para TOMAR FOTO (puedes pulsarla mientras conduces)")
    print(" PULSA 'q' para SALIR")
    print("========================================")
    
    cap = Camera(stream_size=(320, 240), hflip=True, vflip=True)
    cap.start_stream()
    
    levantar_gancho()
    
    last_move_time = 0
    is_moving = False

    try:
        while True:
            frame_bytes = cap.get_frame()
            if frame_bytes is None: continue
            
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            cv2.imshow("Conduccion Manual + YOLO ('w','s','c','q')", frame)
            
            # OpenCV waitKey con refresco rápido. Atrapa las teclas repetidas por el OS
            key = cv2.waitKey(20) & 0xFF
            
            # 1. EVALUAR CAPTURA MANUAL
            if key == ord('c'):
                detecciones_manuales = []
                bola_detectada, _, bboxes_bola = detectar_bola_roja_yolo(frame)
                if bola_detectada:
                    for bbox in bboxes_bola: detecciones_manuales.append((0, bbox))
                
                linea_detectada, _, bboxes_linea = detectar_linea_verde_yolo(frame)
                if linea_detectada:
                    for bbox in bboxes_linea: detecciones_manuales.append((1, bbox))
                    
                guardar_imagen_yolo(frame, detecciones_manuales, prefix="movimiento_manual")
                print(f"[FOTO MANUAL] Captura guardada con {len(detecciones_manuales)} detecciones.      ")
                
                # Le damos un margen extra de tiempo al motor para que no dé un tirón
                # al solapar la tecla 'c' con las teclas de conducción
                if is_moving:
                    last_move_time = time.time()
            
            # 2. EVALUAR CONDUCCIÓN
            elif key == ord('w'):
                avanzar()
                last_move_time = time.time()
                is_moving = True
            elif key == ord('s'):
                retroceder()
                last_move_time = time.time()
                is_moving = True
            elif key == ord('q'):
                break

            # 3. DETENER SI EL USUARIO HA SOLTADO LA TECLA
            # Si hace más de 0.15 segundos que OpenCV no registra la tecla 'w' o 's',
            # consideramos que el usuario ha dejado de pulsarla y detenemos al robot.
            if is_moving and (time.time() - last_move_time > 0.15):
                detener()
                is_moving = False

    finally:
        detener()
        motor.close()
        servo_obj.setServoStop()
        cap.stop_stream()
        cap.close()
        cv2.destroyAllWindows()
        print("Robot detenido de forma segura.")

if __name__ == '__main__':
    main()
