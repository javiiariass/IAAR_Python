import cv2
import numpy as np
import time
import random
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
        # Al invertir, enviamos voltaje positivo (+) para ir hacia atrás (según la polarización invertida corregida)
        motor.setMotorModel(velocidad_retroceso, velocidad_retroceso)
        time.sleep(0.3)
        detener()
        time.sleep(0.1)

    direccion = random.choice(["izquierda", "derecha"])
    # Al estar los motores invertidos, también invertimos la lógica de giro
    # girar derecha -> orugas izquierdas avanzan (negativo) y derechas retroceden (positivo)
    # girar izquierda -> orugas derechas avanzan (negativo) e izquierdas retroceden (positivo)
    if direccion == "derecha":
        motor.setMotorModel(-velocidad_giro, velocidad_giro)
    else:
        # Las orugas izquierdas retroceden (positivo), las derechas avanzan (negativo)
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
        return True, pixeles_verdes
    return False, pixeles_verdes


#BUCLE PRINCIPAL
def main():
    print("INICIANDO COMPORTAMIENTO REACTIVO...")

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

            limite_detectado, cantidad_pixeles = detectar_linea_verde(frame)
            
            # Obtener lectura del sonar
            distancia = sonar.get_distance()
            if distancia < 0:
                # Lectura errónea, forzar un valor alto o mantenerse ignorándolo
                distancia = 999.0

            # Imprimir constantemente los píxeles (usa \r para no saturar la pantalla)
            estado_sonar_str = f"| Distancia: {distancia:5.1f}cm" if distancia != 999.0 else "| Distancia: Error"
            print(f"Px verdes: {cantidad_pixeles} (Umbral actual: 3000) {estado_sonar_str}      ", end="\r")

            #COMPORTAMIENTO REACTIVO
            if limite_detectado or (0 <= distancia <= 20):
                if limite_detectado:
                    print(f"\n¡Límite verde detectado! ({cantidad_pixeles} px) Evadiendo...     ")
                else:
                    print(f"\n¡Obstáculo inminente! ({distancia:5.1f} cm) Evadiendo...          ")
                    
                detener()
                time.sleep(0.3)  # Pausa antes de la maniobra de salida
                
                # Retroceder + girar en vez de solo girar sobre su eje apoyado en la esquina
                girar_aleatorio(retroceder=False)
                
                # Devolver el control al flujo normal
                detener()
                time.sleep(0.2)
                
            elif 20 < distancia <= 40:
                print(f"\nObstáculo a media distancia ({distancia:5.1f} cm). Giro ligero...    ")
                # Girar en un ángulo ligero SIN retroceder.
                # Nota: Se usa velocidad alta (2000) porque con 1000 a veces las orugas no logran vencer la fricción,
                # consiguiendo el "giro ligero" a través de un tiempo de giro más reducido.
                girar_aleatorio(tiempo_min=0.2, tiempo_max=0.4, retroceder=False, velocidad_giro=1500)
                
                # Devolver el control al flujo normal
                detener()
                time.sleep(0.1)

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
        servo_obj.setServoStop()
        sonar.close()
        cap.stop_stream()
        cap.close()
        print("Robot detenido de forma segura.")


if __name__ == '__main__':
    main()