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
    Analiza la parte inferior de la imagen para detectar la cinta verde.
    Retorna True si la detecta cerca, False en caso contrario.
    """
    altura, anchura = frame.shape[:2]

    # 1. Definir ROI (Región de Interés): Analizamos solo el tercio inferior
    roi = frame[int(altura * 2 / 3):altura, 0:anchura]

    # 2. Convertir a HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 3. Definir rangos para el color VERDE (ajustar en el laboratorio)
    verde_bajo = np.array([40, 50, 50])
    verde_alto = np.array([85, 255, 255])

    # 4. Crear una máscara que aísle solo el color verde
    mascara = cv2.inRange(hsv, verde_bajo, verde_alto)

    # 5. Contar cuántos píxeles verdes hay en nuestra zona cercana
    pixeles_verdes = cv2.countNonZero(mascara)

    # umbral límite de puntos verdes
    UMBRAL = 3000
    if pixeles_verdes > UMBRAL:
        return True, pixeles_verdes
    return False, pixeles_verdes

def detectar_bola_roja(frame):
    """
    Analiza la imagen para detectar una mancha roja y devuelve su Bounding Box.
    Retorna (True, area, bbox) si supera el umbral, (False, 0, None) en caso contrario.
    bbox es una tupla (x, y, ancho, alto)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rangos para el color ROJO (requiere dos rangos en OpenCV)
    rojo_bajo1 = np.array([0, 100, 100])
    rojo_alto1 = np.array([10, 255, 255])
    
    rojo_bajo2 = np.array([160, 100, 100])
    rojo_alto2 = np.array([180, 255, 255])

    mascara1 = cv2.inRange(hsv, rojo_bajo1, rojo_alto1)
    mascara2 = cv2.inRange(hsv, rojo_bajo2, rojo_alto2)
    mascara_roja = cv2.add(mascara1, mascara2)

    # Encontrar los contornos de la mancha roja
    contornos, _ = cv2.findContours(mascara_roja, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Umbral (Ajustar en el laboratorio según el tamaño de la bola a esa distancia)
    UMBRAL_ROJO = 1500 

    if contornos:
        # Quedarnos con el contorno más grande (asumimos que es la pelota)
        contorno_mas_grande = max(contornos, key=cv2.contourArea)
        area = cv2.contourArea(contorno_mas_grande)

        if area > UMBRAL_ROJO:
            # Obtener el rectángulo que envuelve a la pelota: (X, Y, Ancho, Alto)
            x, y, w, h = cv2.boundingRect(contorno_mas_grande)
            return True, area, (x, y, w, h)
            
    return False, 0, None


#BUCLE PRINCIPAL
def main():
    print("INICIANDO COMPORTAMIENTO REACTIVO Y RECOLECCIÓN DE DATASET DUAL...")

    # --- CONFIGURACIÓN DEL DATASET (CLASIFICACIÓN) ---
    CARPETA_BASE = "dataset_clasificacion"
    CARPETA_ROJA = os.path.join(CARPETA_BASE, "pelota_roja")
    CARPETA_FONDO = os.path.join(CARPETA_BASE, "sin_pelota")
    
    # Crear las carpetas automáticamente
    os.makedirs(CARPETA_ROJA, exist_ok=True) 
    os.makedirs(CARPETA_FONDO, exist_ok=True) 
    
    MAX_FOTOS_POR_CLASE = 200
    contador_roja = 0
    contador_fondo = 0
    
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
            # Solo ejecutamos esto si alguna de las dos carpetas aún no ha llegado a 200
            if contador_roja < MAX_FOTOS_POR_CLASE or contador_fondo < MAX_FOTOS_POR_CLASE:
                bola_detectada, area_roja, bbox = detectar_bola_roja(frame)
                tiempo_actual = time.time()
                
                # Comprobar que ha pasado el intervalo de medio segundo
                if (tiempo_actual - tiempo_ultima_foto) >= INTERVALO_FOTOS:
                    
                    # CASO A: VEMOS LA PELOTA Y AÚN FALTAN FOTOS ROJAS
                    if bola_detectada and contador_roja < MAX_FOTOS_POR_CLASE:
                        nombre_base = f"bola_{contador_roja:03d}"
                        
                        # 1. Guardar Imagen JPG
                        ruta_imagen = os.path.join(CARPETA_ROJA, f"{nombre_base}.jpg")
                        cv2.imwrite(ruta_imagen, frame)
                        
                        # 2. Guardar Anotación YOLO en TXT (Mantenemos esto por si decides usar Detección de Objetos en el futuro)
                        x, y, w, h = bbox
                        alto_img, ancho_img = frame.shape[:2]
                        x_centro_norm = (x + (w / 2.0)) / ancho_img
                        y_centro_norm = (y + (h / 2.0)) / alto_img
                        ancho_norm = w / float(ancho_img)
                        alto_norm = h / float(alto_img)
                        
                        ruta_txt = os.path.join(CARPETA_ROJA, f"{nombre_base}.txt")
                        with open(ruta_txt, "w") as f:
                            f.write(f"0 {x_centro_norm:.6f} {y_centro_norm:.6f} {ancho_norm:.6f} {alto_norm:.6f}\n")
                            
                        contador_roja += 1
                        tiempo_ultima_foto = tiempo_actual
                        print(f"\n[DATASET] PELOTA guardada: {contador_roja}/{MAX_FOTOS_POR_CLASE}")

                    # CASO B: NO VEMOS LA PELOTA Y AÚN FALTAN FOTOS DE FONDO
                    elif not bola_detectada and contador_fondo < MAX_FOTOS_POR_CLASE:
                        nombre_base = f"fondo_{contador_fondo:03d}"
                        
                        # Guardar SOLO la imagen JPG (sin archivo de texto)
                        ruta_imagen = os.path.join(CARPETA_FONDO, f"{nombre_base}.jpg")
                        cv2.imwrite(ruta_imagen, frame)
                        
                        contador_fondo += 1
                        tiempo_ultima_foto = tiempo_actual
                        print(f"\n[DATASET] FONDO guardado: {contador_fondo}/{MAX_FOTOS_POR_CLASE}")

                    # COMPROBAR SI HEMOS TERMINADO TODO
                    if contador_roja == MAX_FOTOS_POR_CLASE and contador_fondo == MAX_FOTOS_POR_CLASE:
                        print("\n[DATASET] ¡ÉXITO! Dataset completo con 200 fotos de cada clase.")
            # -----------------------------------------------------------

            # --- LÓGICA ORIGINAL DE PERCEPCIÓN ---
            limite_detectado, cantidad_pixeles = detectar_linea_verde(frame)
            
            # Obtener lectura del sonar
            distancia = sonar.get_distance()
            if distancia < 0:
                distancia = 999.0

            # Imprimir constantemente los píxeles (usa \r para no saturar la pantalla)
            estado_sonar_str = f"| Distancia: {distancia:5.1f}cm" if distancia != 999.0 else "| Distancia: Error"
            print(f"Px verdes: {cantidad_pixeles} (Umbral actual: 3000) {estado_sonar_str}      ", end="\r")

            # --- COMPORTAMIENTO REACTIVO ---
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
