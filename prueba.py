import time
import random
import cv2
import numpy as np
from camera import Camera
from motor import tankMotor

def main():
    print("Iniciando Práctica 2: Comportamiento reactivo con cámara (Tanque)...")
    
    # 1. Inicialización de componentes
    motor = tankMotor()
    cam = Camera(stream_size=(400, 300))
    # Arrancamos el stream interno de la cámara para obtener fotos JPEG rápidamente
    cam.start_stream() 
    
    # Rango de color verde en HSV (OpenCV usa H: 0-179, S: 0-255, V: 0-255)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Umbral de píxeles para considerar que la cinta está suficientemente cerca
    AREA_THRESHOLD = 500  
    SPEED = 2500

    try:
        # Dejar que la cámara se estabilice un segundo
        time.sleep(1)
        print("Robot deambulando...")

        while True:
            # --- FASE DE PERCEPCIÓN ---
            frame_bytes = cam.get_frame()
            
            if frame_bytes is None:
                continue
                
            # Decodificar el frame JPEG a una matriz de imagen de OpenCV
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Recortar la imagen para mirar solo la parte inferior (suelo más cercano)
            # La imagen es de 400x300, nos quedamos de la fila 150 a la 300
            height, width = img.shape[:2]
            img_bottom = img[int(height/2):height, 0:width]
            
            # Convertir a espacio de color HSV (mejor para detectar color ignorando luces/sombras)
            hsv = cv2.cvtColor(img_bottom, cv2.COLOR_BGR2HSV)
            
            # Crear una máscara que resalte solo los píxeles dentro del rango del verde
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Contar los píxeles blancos de la máscara (los que son verdes)
            green_pixels = cv2.countNonZero(mask)
            
            # --- FASE DE REACCIÓN ---
            if green_pixels > AREA_THRESHOLD:
                print(f"Límite verde detectado (Píxeles: {green_pixels}). Girando...")
                
                # Detener momentáneamente
                motor.setMotorModel(0, 0)
                time.sleep(0.1)
                
                # Generar tiempo de giro aleatorio entre 0.5 y 2.0 segundos (grado aleatorio)
                turn_time = random.uniform(0.5, 2.0)
                
                # Decidir aleatoriamente entre girar a la izquierda o la derecha
                if random.choice([True, False]):
                    # Girar Derecha
                    motor.setMotorModel(SPEED, -SPEED)
                else:
                    # Girar Izquierda
                    motor.setMotorModel(-SPEED, SPEED)
                
                # Mantener el giro durante el tiempo aleatorio
                time.sleep(turn_time)
                print("Fin del giro. Retomando avance libre.")
                
            else:
                # Si no se detecta la línea, avanzar libremente en línea recta
                motor.setMotorModel(SPEED, SPEED)
                
    except KeyboardInterrupt:
        print("\nPráctica terminada por el usuario. Deteniendo hardware...")
        
    finally:
        # Asegurar un cierre limpio
        motor.setMotorModel(0, 0)
        motor.close()
        cam.stop_stream()
        cam.close()

if __name__ == '__main__':
    main()