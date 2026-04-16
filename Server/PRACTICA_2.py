import cv2
import numpy as np
import time
import random
from Motor import *  # Importamos la clase Motor del repositorio de Freenove

# Inicializamos el controlador de motores de Freenove
motor = Motor()


# --- FUNCIONES DE MOVIMIENTO ---

def detener():
    """Detiene todos los motores del tanque."""
    motor.setMotorModel(0, 0, 0, 0)


def avanzar(velocidad=1500):
    """Hace que el tanque avance en línea recta."""
    # En Freenove, los valores van típicamente de -4095 a 4095
    motor.setMotorModel(velocidad, velocidad, velocidad, velocidad)


def girar_aleatorio():
    """Realiza un giro sobre sí mismo de duración y sentido aleatorios."""
    velocidad_giro = 2000
    direccion = random.choice(["izquierda", "derecha"])

    if direccion == "derecha":
        # Las orugas izquierdas avanzan, las derechas retroceden
        motor.setMotorModel(velocidad_giro, velocidad_giro, -velocidad_giro, -velocidad_giro)
    else:
        # Las orugas izquierdas retroceden, las derechas avanzan
        motor.setMotorModel(-velocidad_giro, -velocidad_giro, velocidad_giro, velocidad_giro)

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

    # Mostrar por pantalla para depuración (opcional, puedes comentarlo)
    # cv2.imshow("Mascara Verde", mascara)
    # cv2.waitKey(1)

    # 6. Umbral de decisión (si hay más de X píxeles, consideramos que hemos topado con el límite)
    UMBRAL = 2000  # <-- Ajustar este valor según la resolución y tamaño de la cinta

    if pixeles_verdes > UMBRAL:
        return True
    return False


# --- BUCLE PRINCIPAL (COMPORTAMIENTO REACTIVO) ---

def main():
    print("Iniciando Práctica 2: Comportamiento Reactivo...")

    # Inicializar la cámara (0 suele ser la cámara Pi conectada/USB)
    cap = cv2.VideoCapture(0)

    # Bajar la resolución para procesar más rápido
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer la cámara")
                break

            # Percibir el entorno
            limite_detectado = detectar_linea_verde(frame)

            # Actuar en consecuencia (Reflejo simple)
            if limite_detectado:
                print("¡Límite verde detectado! Evadiendo...")
                detener()
                time.sleep(0.2)  # Breve pausa para estabilizar
                girar_aleatorio()
            else:
                avanzar()

    except KeyboardInterrupt:
        # Manejo seguro si presionas Ctrl+C en la consola
        print("\nPrograma interrumpido por el usuario.")

    finally:
        # Siempre detener motores y liberar cámara al salir
        detener()
        cap.release()
        cv2.destroyAllWindows()
        print("Robot detenido de forma segura.")


if __name__ == '__main__':
    main()