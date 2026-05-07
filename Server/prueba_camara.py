import cv2
import numpy as np
import time
from camera import Camera

def main():
    print("Iniciando prueba de cámara pura (sin GUI/Qt)...")
    
    # Iniciamos la cámara igual que en la PRÁCTICA 2, sin usar start_image() que suele invocar librerías gráficas
    cap = Camera(stream_size=(640, 480), hflip=True, vflip=True)
    cap.start_stream()
    
    print("Esperando 2 segundos para que la cámara enfoque/ajuste la luz...")
    time.sleep(2)
    
    print("Capturando frame...")
    frame_bytes = cap.get_frame()
    
    if frame_bytes is None:
        print("ESTADO: ERROR - No se ha podido obtener ningún frame byte de la cámara.")
    else:
        # Decodificamos la imagen
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            nombre_archivo = "foto_prueba.jpg"
            cv2.imwrite(nombre_archivo, frame)
            print(f"ESTADO: ¡ÉXITO! La cámara funciona correctamente. Imagen guardada como '{nombre_archivo}' en la carpeta actual.")
        else:
            print("ESTADO: ERROR - El frame se obtuvo pero OpenCV no pudo decodificarlo.")
            
    print("Cerrando la conexión con la cámara...")
    try:
        cap.stop_stream()
        cap.close()
    except:
        pass
    print("Test finalizado.")

if __name__ == "__main__":
    main()
