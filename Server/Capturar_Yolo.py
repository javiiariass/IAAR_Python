import cv2
import numpy as np
import time
import os
from camera import Camera

def detectar_linea_verde(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    verde_bajo = np.array([40, 50, 50])
    verde_alto = np.array([85, 255, 255])
    mascara = cv2.inRange(hsv, verde_bajo, verde_alto)
    
    bboxes = []
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        for contorno in contornos:
            # Comprobar que realmente tiene un tamaño mínimo
            if cv2.contourArea(contorno) > 500: # Umbral de área para líneas individuales
                x, y, w, h = cv2.boundingRect(contorno)
                bboxes.append((x, y, w, h))
                
    return (len(bboxes) > 0), 0, bboxes

def detectar_bola_roja(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Aumentamos la saturación mínima a 150 y acotamos más el tono (Hue) para evitar detectar la piel
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
                # Calculamos la proporción entre el ancho y la altura, brazos/manos son alargados
                # pero una bola encaja en un cuadrado (proporción cercana a 1.0)
                aspect_ratio = float(w) / h
                if 0.5 <= aspect_ratio <= 2.0:
                    bboxes.append((x, y, w, h))
    if bboxes:
        return True, 0, bboxes
    return False, 0, []

def guardar_imagen_yolo(frame, detecciones, base_dir="dataset_clasificacion", prefix="manual_yolo"):
    """Guarda imagen y etiqueta YOLO asegurando no sobreescribir archivos existentes con múltiples detecciones posibles."""
    os.makedirs(base_dir, exist_ok=True)
    
    # Encontrar siguiente índice disponible
    idx = 0
    while True:
        nombre_base = f"{prefix}_{idx:04d}"
        ruta_imagen = os.path.join(base_dir, f"{nombre_base}.jpg")
        ruta_txt = os.path.join(base_dir, f"{nombre_base}.txt")
        if not os.path.exists(ruta_imagen) and not os.path.exists(ruta_txt):
            break
        idx += 1
        
    cv2.imwrite(ruta_imagen, frame)
    
    # Siempre creamos el archivo de texto, y añadimos todas las detecciones que hayamos encontrado
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

def main():
    print("Script para generar una nueva imagen YOLO sin sobreescribir...")
    cap = Camera(stream_size=(320, 240), hflip=True, vflip=True)
    cap.start_stream()
    time.sleep(1)

    try:
        while True:
            frame_bytes = cap.get_frame()
            if frame_bytes is None: continue
            
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            cv2.imshow("Camara - Presiona 'c' para capturar la imagen, 'q' para salir", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("Capturando imagen y detectando clases...")
                detecciones = []
                
                bola_detectada, _, bboxes_bola = detectar_bola_roja(frame)
                if bola_detectada and bboxes_bola:
                    print(f"- {len(bboxes_bola)} Pelota(s) rojas detectada(s)!")
                    for bbox in bboxes_bola:
                        detecciones.append((0, bbox))
                    
                linea_detectada, _, bboxes_linea = detectar_linea_verde(frame)
                if linea_detectada and bboxes_linea:
                    print(f"- {len(bboxes_linea)} Línea(s) verde(s) detectada(s)!")
                    for bbox in bboxes_linea:
                        detecciones.append((1, bbox))
                    
                if not detecciones:
                    print("- Ningún objeto detectado. Guardando sin anotaciones (como imagen de fondo).")
                    
                guardar_imagen_yolo(frame, detecciones, prefix="yolo_manual")

    finally:
        cap.stop_stream()
        cap.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
