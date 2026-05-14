import cv2
import numpy as np
import time
import random
import os  # Necesario para crear las carpetas del dataset
from motor import tankMotor
from camera import Camera
from servo import Servo
from ultrasonic import Ultrasonic

motor = tankMotor()
servo_obj = Servo()

def levantar_gancho():
    """Levanta el gancho/brazo para que no bloquee la cámara."""
    # servo_obj.setServoAngle('0', 90)
    servo_obj.setServoAngle('1', 140)
    time.sleep(0.5)

def detener():
    motor.setMotorModel(0, 0)

# valores de -4095 a 4095
def avanzar(velocidad=1200):
    # Valores negativos para forzar avance hacia adelante si los motores están montados al revés
    factor_correccion = 1.2
    motor.setMotorModel(-velocidad, -velocidad*factor_correccion)

def girar_aleatorio(tiempo_min=0.5, tiempo_max=1.5, retroceder=True, velocidad_giro=2000):
    """Realiza un giro sobre sí mismo de duración aleatoria acotada entre tiempo_min y tiempo_max. Si está atascado, retrocede."""
    velocidad_retroceso = 1200

    if retroceder:
        # Retroceder un poco SIEMPRE antes de girar (para escapar de la esquina físicamente)
        motor.setMotorModel(velocidad_retroceso, velocidad_retroceso)
        time.sleep(0.3)
        detener()
    
    time.sleep(0.1)

    direccion = random.choice(["izquierda", "derecha"])
    if direccion == "derecha":
        motor.setMotorModel(-velocidad_giro, velocidad_giro)
    else:
        motor.setMotorModel(velocidad_giro, -velocidad_giro)

    # El tiempo de giro aleatorio define los grados del giro, acotando entre los parámetros min y max
    tiempo_giro = random.uniform(tiempo_min, tiempo_max)
    time.sleep(tiempo_giro)
    detener()


# --- FUNCIONES DE PERCEPCIÓN ---

def detectar_linea_verde(frame):
    """
    Analiza TODA la imagen para detectar la cinta verde para el dataset,
    pero evalúa solo el tercio inferior para indicar si debe activar el comportamiento reactivo (evasión).
    """
    altura, anchura = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 3. Definir rangos para el color VERDE (ajustar en el laboratorio)
    verde_bajo = np.array([40, 50, 50])
    verde_alto = np.array([85, 255, 255])

    # 4. Crear una máscara que aísle solo el color verde
    mascara = cv2.inRange(hsv, verde_bajo, verde_alto)

    # Evaluar solo el tercio inferior para comportamiento reactivo (frenar)
    mascara_roi = mascara[int(altura * 2 / 3):altura, 0:anchura]
    pixeles_verdes_roi = cv2.countNonZero(mascara_roi)
    UMBRAL_REACTIVO = 3000
    limite_reactivo = (pixeles_verdes_roi > UMBRAL_REACTIVO)

    bboxes = []
    # Encontrar los contornos de la línea verde en la imagen completa para YOLO
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        for contorno in contornos:
            if cv2.contourArea(contorno) > 500: # Umbral de área para líneas individuales
                x, y, w, h = cv2.boundingRect(contorno)
                bboxes.append((x, y, w, h))
                
    return limite_reactivo, pixeles_verdes_roi, bboxes

def detectar_bola_roja(frame):
    """
    Analiza la imagen para detectar una mancha roja y devuelve su Bounding Box.
    Retorna (True, area, bbox) si supera el umbral, (False, 0, None) en caso contrario.
    bbox es una tupla (x, y, ancho, alto) o una lista de tuplas
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Aumentamos la saturación mínima a 150 y acotamos más el tono (Hue) para evitar detectar la piel
    rojo_bajo1 = np.array([0, 150, 100])
    rojo_alto1 = np.array([8, 255, 255])
    
    rojo_bajo2 = np.array([170, 150, 100])
    rojo_alto2 = np.array([180, 255, 255])

    mascara1 = cv2.inRange(hsv, rojo_bajo1, rojo_alto1)
    mascara2 = cv2.inRange(hsv, rojo_bajo2, rojo_alto2)
    mascara_roja = cv2.add(mascara1, mascara2)

    # Encontrar los contornos de la mancha roja
    contornos, _ = cv2.findContours(mascara_roja, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Umbral (Ajustar en el laboratorio según el tamaño de la bola a esa distancia)
    UMBRAL_ROJO = 1500 
    bboxes = []

    if contornos:
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area > UMBRAL_ROJO:
                # Obtener el rectángulo que envuelve a la pelota: (X, Y, Ancho, Alto)
                x, y, w, h = cv2.boundingRect(contorno)
                # La bola formará un cuadro. Excluimos objetos alargados (brazos/manos)
                aspect_ratio = float(w) / h
                if 0.5 <= aspect_ratio <= 2.0:
                    bboxes.append((x, y, w, h))
                
        if bboxes:
            return True, 0, bboxes
            
    return False, 0, []


#BUCLE PRINCIPAL
def main():
    print("INICIANDO COMPORTAMIENTO REACTIVO Y RECOLECCIÓN DE DATASET DUAL...")

    # --- CONFIGURACIÓN DEL DATASET (CLASIFICACIÓN/YOLO) ---
    CARPETA_BASE = "dataset_clasificacion"
    
    # Crear las carpetas automáticamente
    os.makedirs(CARPETA_BASE, exist_ok=True) 
    
    # Encuentra el índice inicial para no sobreescribir imágenes
    def obtener_siguiente_indice(prefijo):
        idx = 0
        while os.path.exists(os.path.join(CARPETA_BASE, f"{prefijo}_{idx:04d}.jpg")):
            idx += 1
        return idx

    contador_roja = obtener_siguiente_indice("yolo_dataset")
    contador_verde = obtener_siguiente_indice("yolo_dataset")  # Usaremos un índice global para los archivos, o simplemente autoincrementar

    indice_global = max(obtener_siguiente_indice("yolo_dataset"), 0)
    
    INTERVALO_FOTOS = 0.5 # Segundos de espera mínima entre fotos
    tiempo_ultima_foto = time.time()

    # Levantar el gancho para que no estorbe a la cámara
    levantar_gancho()

    # Inicializar el sensor de ultrasonidos (sonar)
    sonar = Ultrasonic()

    # Usar la clase Camera definida en el workspace (usa picamera2 internamente)
    cap = Camera(stream_size=(320, 240), hflip=True, vflip=True)
    cap.start_stream()

    # Dar un segundo para que la cámara arranque y exponga correctamente
    time.sleep(1)

    try:
        while True:
            # Obtener frame como bytes
            frame_bytes = cap.get_frame()
            if frame_bytes is None:
                continue

            # Decodificar el frame a una imagen OpenCV
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Error al decodificar la imagen de la cámara")
                continue

            # --- LÓGICA DE RECOLECCIÓN DE DATASET (DOS CLASES) ---
            # Guardaremos imágenes junto con anotaciones para pelota roja (0) y linea verde (1)
            _, _, bboxes_roja = detectar_bola_roja(frame)
            limite_reactivo, cantidad_pixeles, bboxes_verde = detectar_linea_verde(frame)
            
            tiempo_actual = time.time()
            if (tiempo_actual - tiempo_ultima_foto) >= INTERVALO_FOTOS:
                # Si hay alguna detección en la imagen completa, guardamos un archivo de anotación YOLO
                if bboxes_roja or bboxes_verde:
                    nombre_base = f"yolo_dataset_{indice_global:04d}"
                    ruta_imagen = os.path.join(CARPETA_BASE, f"{nombre_base}.jpg")
                    ruta_txt = os.path.join(CARPETA_BASE, f"{nombre_base}.txt")
                    
                    cv2.imwrite(ruta_imagen, frame)
                    alto_img, ancho_img = frame.shape[:2]
                    
                    with open(ruta_txt, "w") as f:
                        if bboxes_roja:
                            for bbox in bboxes_roja:
                                x, y, w, h = bbox
                                x_centro_norm = (x + (w / 2.0)) / ancho_img
                                y_centro_norm = (y + (h / 2.0)) / alto_img
                                ancho_norm = w / float(ancho_img)
                                alto_norm = h / float(alto_img)
                                f.write(f"0 {x_centro_norm:.6f} {y_centro_norm:.6f} {ancho_norm:.6f} {alto_norm:.6f}\n")
                        
                        if bboxes_verde:
                            for bbox in bboxes_verde:
                                x, y, w, h = bbox
                                x_centro_norm = (x + (w / 2.0)) / ancho_img
                                y_centro_norm = (y + (h / 2.0)) / alto_img
                                ancho_norm = w / float(ancho_img)
                                alto_norm = h / float(alto_img)
                                f.write(f"1 {x_centro_norm:.6f} {y_centro_norm:.6f} {ancho_norm:.6f} {alto_norm:.6f}\n")
                            
                    print(f"\n[DATASET] Imagen y {len(bboxes_roja) + len(bboxes_verde)} anotación(es) guardadas: {nombre_base}")
                    indice_global += 1
                    tiempo_ultima_foto = tiempo_actual
            # -----------------------------------------------------------
            
            # Obtener lectura del sonar
            distancia = sonar.get_distance()
            if distancia < 0:
                distancia = 999.0

            # Imprimir constantemente los píxeles (usa \r para no saturar la pantalla)
            estado_sonar_str = f"| Distancia: {distancia:5.1f}cm" if distancia != 999.0 else "| Distancia: Error"
            print(f"Px verdes: {cantidad_pixeles} (Umbral actual: 3000) {estado_sonar_str}      ", end="\r")

            # --- COMPORTAMIENTO REACTIVO ---
            if limite_reactivo or (0 <= distancia <= 20):
                if limite_reactivo:
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

    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario.")

    finally:
        detener()
        motor.close()
        servo_obj.setServoStop()
        sonar.close()
        cap.stop_stream()
        cap.close()
        print("Robot detenido de forma segura.")


if __name__ == '__main__':
    main()
