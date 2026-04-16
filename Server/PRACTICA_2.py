import cv2
import numpy as np
import time
import random
from motor import tankMotor
from camera import Camera

motor = tankMotor()

def detener():
    motor.setMotorModel(0, 0)


# valores de -4095 a 4095
def avanzar(velocidad=1500):
    # Valores negativos para forzar avance hacia adelante si los motores están montados al revés
    motor.setMotorModel(-velocidad, -velocidad)

def girar_aleatorio():
    """Realiza un giro sobre sí mismo de duración y sentido aleatorios."""
    velocidad_giro = 2000
    direccion = random.choice(["izquierda", "derecha"])
    # Al estar los motores invertidos, también invertimos la lógica de giro
    # girar derecha -> orugas izquierdas avanzan (negativo) y derechas retroceden (positivo)
    # girar izquierda -> orugas derechas avanzan (negativo) e izquierdas retroceden (positivo)
    if direccion == "derecha":
        motor.setMotorModel(-velocidad_giro, velocidad_giro)
    else:
        # Las orugas izquierdas retroceden (positivo), las derechas avanzan (negativo)
        motor.setMotorModel(velocidad_giro, -velocidad_giro)

    # El tiempo de giro aleatorio (entre 0.5 y 1.5 segundos) define el grado del giro
    tiempo_giro = random.uniform(0.5, 1.5)
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
    # H: Matiz (aprox 40-80 para verde), S: Saturación, V: Brillo
    verde_bajo = np.array([40, 50, 50])
    verde_alto = np.array([85, 255, 255])

    # 4. Crear una máscara que aísle solo el color verde
    mascara = cv2.inRange(hsv, verde_bajo, verde_alto)

    # 5. Contar cuántos píxeles verdes hay en nuestra zona cercana
    pixeles_verdes = cv2.countNonZero(mascara)

    #debug para comprobar
    # cv2.imshow("Mascara Verde", mascara)
    # cv2.waitKey(1)

    # umbral límite de puntos verdes
    UMBRAL = 3000
    if pixeles_verdes > UMBRAL:
        return True
    return False


#BUCLE PRINCIPAL
def main():
    print("INICIANDO COMPORTAMIENTO REACTIVO...")

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

            limite_detectado = detectar_linea_verde(frame)

            #COMPORTAMIENTO REACTIVO
            if limite_detectado:
                print("¡Límite verde detectado! Evadiendo...")
                detener()
                time.sleep(0.2)  # Breve pausa para estabilizar
                girar_aleatorio()
            else:
                avanzar()

    #DETENER PROGRAMA
    except KeyboardInterrupt:
        #CTRL + C
        print("\nPrograma interrumpido por el usuario.")

    finally:
        # Siempre detener motores y liberar cámara al salir
        detener()
        motor.close()
        cap.stop_stream()
        cap.close()
        print("Robot detenido de forma segura.")


if __name__ == '__main__':
    main()