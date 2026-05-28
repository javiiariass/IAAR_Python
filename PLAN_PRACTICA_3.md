# Plan Práctica 3 — Entrenamiento YOLO para el Tanque

**Fecha objetivo:** Lunes 25 de mayo de 2026  
**Entrega final:** 2-4 de junio de 2026  
**Objetivo:** Entrenar un modelo YOLOv8n que detecte `bola_roja` (clase 0) y `linea_verde` (clase 1) y ejecutarlo en tiempo real en la Raspberry Pi para el comportamiento deliberativo del tanque.

---

## Estado actual del proyecto (diagnóstico)

### Lo que ya tenemos

- **183 imágenes** en `Server/dataset_clasificacion/` a resolución 320×240
  - 125 con anotaciones, 58 vacías (negative samples = 31.7%, perfecto)
  - 82 anotaciones de `bola_roja`, 53 de `linea_verde` (en formato labelImg, clase 16/17)
  - 36 anotaciones de `bola_roja`, 32 de `linea_verde` (en formato auto-captura, clase 0/1)
- **3 vídeos de prueba** en `Server/videos_prueba/` (~14 MB total, del 19 de mayo)
- **Scripts de captura** operativos: `Capturar_Yolo.py`, `Capturar_Yolo_Servidor.py`, `Capturar_Yolo_Movimiento_Manual.py`, `Grabar_Video_Autonomo.py`
- **Script de pinza** probado: `Practica_3/prueba_pinza.py`
- **labelImg** clonado en el repo

### Problemas críticos detectados

1. **IDs de clase inconsistentes** — Es el problema más grave:
   - Los scripts automáticos (`PRACTICA_3_DATASET_ANOTACIONES.py`, etc.) escriben clase `0` = bola_roja, `1` = linea_verde
   - labelImg usó `predefined_classes.txt` con 15 clases basura (dog, person, cat...) antes de bola_roja, así que etiquetó `16` = bola_roja, `17` = linea_verde
   - **Resultado:** El dataset mezcla ambos esquemas y NO se puede usar directamente para entrenar
   - El `classes.txt` del dataset confirma la confusión (contiene las 15 clases + las nuestras)

2. **Dataset pequeño** — 183 imágenes es poco para YOLO, aunque con data augmentation y transfer learning (pesos preentrenados COCO) puede funcionar razonablemente con ~300-500 imágenes.

3. **No hay notebook de entrenamiento funcional** — `colab.ipynb` está vacío.

---

## PLAN PASO A PASO

### FASE 0 — Preparación local (hacer HOY, antes de mañana)

#### 0.1 Arreglar los IDs de clase del dataset

Esto es lo primero y más crítico. Hay que unificar TODAS las anotaciones a `0` = bola_roja, `1` = linea_verde.

```python
# Script: fix_class_ids.py — Ejecutar en local o en la RPi
# Convierte clase 16 → 0 (bola_roja) y clase 17 → 1 (linea_verde)
import os, glob

DATASET_DIR = "Server/dataset_clasificacion"
MAPPING = {"16": "0", "17": "1"}  # labelImg → YOLO

for txt_path in glob.glob(os.path.join(DATASET_DIR, "*.txt")):
    if os.path.basename(txt_path) == "classes.txt":
        continue
    
    lines = open(txt_path).readlines()
    new_lines = []
    changed = False
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] in MAPPING:
            parts[0] = MAPPING[parts[0]]
            changed = True
        # Ignorar líneas con texto como "0=bola_roja"
        if "=" in parts[0]:
            changed = True
            continue
        new_lines.append(" ".join(parts) + "\n")
    
    if changed:
        with open(txt_path, "w") as f:
            f.writelines(new_lines)
        print(f"Corregido: {os.path.basename(txt_path)}")

# Reescribir classes.txt limpio
with open(os.path.join(DATASET_DIR, "classes.txt"), "w") as f:
    f.write("bola_roja\nlinea_verde\n")

print("¡Dataset unificado!")
```

**Verificación:** Tras ejecutar, comprobar que solo existen clases 0 y 1:
```bash
grep -h "^[0-9]" Server/dataset_clasificacion/*.txt | awk '{print $1}' | sort | uniq -c
```

#### 0.2 Arreglar predefined_classes.txt de labelImg

```bash
echo -e "bola_roja\nlinea_verde" > labelimg/labelImg-master/data/predefined_classes.txt
```

Así si mañana necesitáis re-etiquetar algo con labelImg, usará clase 0 y 1 directamente.

#### 0.3 Subir el dataset a Google Drive

El entrenamiento se hará en **Google Colab con GPU** (como recomienda el tema 6 de la asignatura). La RPi no tiene potencia para entrenar.

1. Crear una carpeta en Google Drive: `IAAR/practica3_yolo/`
2. Subir toda la carpeta `Server/dataset_clasificacion/` (solo los .jpg y .txt, sin subcarpetas de backup)
3. Subir también los vídeos de `Server/videos_prueba/` (servirán para validación visual)

---

### FASE 1 — Ampliar el dataset en el laboratorio (mañana, ~1h)

El dataset actual tiene ~183 imágenes. Para YOLO con transfer learning, un mínimo razonable es **300-500 imágenes**. Recomiendo llegar a ~400.

#### 1.1 Estrategia de captura

Usar `Capturar_Yolo_Servidor.py` (robot quieto + streaming por la app Freenove) combinado con `Capturar_Yolo.py` (robot autónomo):

| Escenario | Nº fotos objetivo | Script recomendado |
|-----------|-------------------|--------------------|
| Bola sola, distintas distancias | 40-50 | Servidor (estático) |
| Bola parcialmente oculta/ocluida | 20-30 | Servidor (estático) |
| Línea verde desde distintos ángulos | 30-40 | Autónomo |
| Bola + línea juntas en frame | 20-30 | Autónomo / Servidor |
| Fondos vacíos (negative samples) | 20-30 | Servidor (tecla 'd') |
| Robot en movimiento (motion blur) | 30-40 | Autónomo |

**Consejos de las instrucciones del repo (0000_Instrucciones.md):**
- Variar iluminación (ventanas, sombras, bajo luces directas)
- Bolas cortadas por el borde de la imagen (oclusiones parciales)
- Incluir zapatos/manos en algunos fondos para que aprenda a no confundirlos
- Negative samples = 10-15% del total

#### 1.2 Verificación rápida de las nuevas fotos

Tras capturar, hacer un conteo rápido:
```bash
cd Server/dataset_clasificacion
echo "Total imágenes: $(ls *.jpg | wc -l)"
echo "Con anotación: $(find . -name '*.txt' ! -name 'classes.txt' -size +0c | wc -l)"
echo "Vacías (fondo): $(find . -name '*.txt' ! -name 'classes.txt' -empty | wc -l)"
echo "Anotaciones clase 0 (bola): $(grep -h '^0 ' *.txt | wc -l)"
echo "Anotaciones clase 1 (línea): $(grep -h '^1 ' *.txt | wc -l)"
```

#### 1.3 Subir las nuevas fotos a Google Drive

Sincronizar las nuevas imágenes con la carpeta de Drive.

---

### FASE 2 — Organizar el dataset en formato YOLO (Colab, ~15 min)

YOLOv8 (Ultralytics) espera esta estructura exacta:

```
dataset/
├── images/
│   ├── train/    (80% de las imágenes)
│   └── val/      (20% de las imágenes)
├── labels/
│   ├── train/    (los .txt correspondientes)
│   └── val/      (los .txt correspondientes)
└── dataset.yaml
```

#### 2.1 Script de partición (ejecutar en Colab)

```python
import os, shutil, random, glob

# Montar Drive
from google.colab import drive
drive.mount('/content/drive')

SRC = "/content/drive/MyDrive/Colab Notebooks/IAAR/practica3_yolo/dataset"
DST = "/content/dataset"

# Crear estructura
for split in ['train', 'val']:
    os.makedirs(f"{DST}/images/{split}", exist_ok=True)
    os.makedirs(f"{DST}/labels/{split}", exist_ok=True)

# Listar todas las imágenes con su .txt
images = sorted(glob.glob(f"{SRC}/*.jpg"))
random.seed(42)
random.shuffle(images)

split_idx = int(len(images) * 0.8)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

for img_list, split in [(train_imgs, 'train'), (val_imgs, 'val')]:
    for img_path in img_list:
        base = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(SRC, f"{base}.txt")
        
        shutil.copy(img_path, f"{DST}/images/{split}/")
        if os.path.exists(txt_path) and os.path.basename(txt_path) != "classes.txt":
            shutil.copy(txt_path, f"{DST}/labels/{split}/")
        else:
            # Crear .txt vacío para negative samples
            open(f"{DST}/labels/{split}/{base}.txt", 'w').close()

print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)}")
```

#### 2.2 Crear dataset.yaml

```python
yaml_content = """
path: /content/dataset
train: images/train
val: images/val

nc: 2
names: ['bola_roja', 'linea_verde']
"""

with open(f"{DST}/dataset.yaml", 'w') as f:
    f.write(yaml_content)
```

---

### FASE 3 — Entrenar YOLOv8n en Colab (~30-45 min)

#### 3.1 ¿Por qué YOLOv8n?

- **YOLOv8n (nano):** 3.2M de parámetros, arquitectura moderna (2023) con detección anchor-free y cabeza desacoplada. Más preciso que YOLOv5n (~33% más mAP en COCO) con un coste computacional aún aceptable para la RPi.
- **Funciona en CPU** de la Raspberry Pi exportado a ONNX (~2-4 FPS en RPi 4 a 320×240, suficiente para el tanque).
- **Transfer learning** desde pesos COCO preentrenados (80 clases genéricas). Solo hay que ajustar a nuestras 2 clases.
- La librería `ultralytics` unifica entrenamiento, validación y exportación con una API sencilla.

#### 3.2 Notebook de entrenamiento

```python
# Celda 1: Instalar ultralytics (incluye YOLOv8)
!pip install ultralytics

# Celda 2: Entrenar
from ultralytics import YOLO

# Cargar el modelo preentrenado YOLOv8 nano
model = YOLO("yolov8n.pt")

# Entrenar con transfer learning sobre nuestro dataset
results = model.train(
    data="/content/dataset/dataset.yaml",
    epochs=100,
    imgsz=320,           # Resolución = la de la cámara del robot
    batch=32,            # Colab con GPU T4 lo aguanta bien
    patience=20,         # Early stopping: para si no mejora en 20 épocas
    project="/content/drive/MyDrive/Colab Notebooks/IAAR/practica3_yolo/runs",
    name="tanque_v1",
    cache=True,          # Cachea imágenes en RAM para ir más rápido
    device=0,            # Usar GPU
)

# Qué hace el transfer learning aquí:
# El modelo ya viene entrenado con 80 clases de objetos comunes (COCO).
# Las capas convolucionales ya "saben" extraer bordes, formas, colores, etc.
# Al entrenar con nuestras imágenes, YOLOv8 solo necesita aprender:
# "esta forma roja redonda = bola_roja" y "esta franja verde = linea_verde",
# reutilizando todo lo que ya aprendió de COCO. Por eso funciona con pocos datos.
```

#### 3.3 Data Augmentation (ya integrada en YOLOv8)

YOLOv8 aplica augmentation automáticamente durante el entrenamiento: mosaic (mezcla 4 imágenes), flip horizontal/vertical, escala, variación de color HSV, etc. Con un dataset pequeño (~300-500 imgs) esto es fundamental y no necesitas configurar nada extra.

Si quieres personalizar el augmentation (opcional), puedes pasar parámetros adicionales a `model.train()`:

```python
results = model.train(
    data="/content/dataset/dataset.yaml",
    epochs=100,
    imgsz=320,
    batch=32,
    patience=20,
    # --- Augmentation personalizado (opcional) ---
    hsv_h=0.02,      # Variación de tono (importante para rojo/verde)
    hsv_s=0.8,        # Variación de saturación
    hsv_v=0.5,        # Variación de brillo
    degrees=15.0,     # Rotación máxima
    translate=0.2,    # Traslación
    scale=0.6,        # Escala (más agresivo que default)
    shear=5.0,        # Deformación
    flipud=0.1,       # Volteo vertical
    fliplr=0.5,       # Volteo horizontal
    mosaic=1.0,       # Mosaic augmentation
    mixup=0.1,        # MixUp augmentation
    project="/content/drive/MyDrive/Colab Notebooks/IAAR/practica3_yolo/runs",
    name="tanque_v1",
    cache=True,
    device=0,
)
```

#### 3.4 Evaluar el modelo entrenado

```python
# Celda 3: Ver las curvas de entrenamiento (loss, mAP, precision, recall)
from IPython.display import Image, display
display(Image('/content/drive/MyDrive/Colab Notebooks/IAAR/practica3_yolo/runs/tanque_v1/results.png'))

# Celda 4: Validar sobre el conjunto de validación
model = YOLO("/content/drive/MyDrive/Colab Notebooks/IAAR/practica3_yolo/runs/tanque_v1/weights/best.pt")
metrics = model.val(data="/content/dataset/dataset.yaml", imgsz=320)
print(f"mAP@0.5: {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")

# Celda 5: Probar con los vídeos grabados (genera vídeos con las cajas dibujadas)
results = model.predict(
    source="/content/drive/MyDrive/Colab Notebooks/IAAR/practica3_yolo/videos_prueba/",
    imgsz=320,
    conf=0.4,
    save=True,
    save_txt=True,
    project="/content/drive/MyDrive/Colab Notebooks/IAAR/practica3_yolo/inferencia_videos"
)
```

**Métricas objetivo (razonables para este caso):**
- mAP@0.5 ≥ 0.75 para ambas clases
- Precision ≥ 0.80
- Recall ≥ 0.70

Si no se alcanzan: más datos → re-entrenar.

---

### FASE 4 — Exportar el modelo para la Raspberry Pi (~10 min)

#### 4.1 Exportar a ONNX (formato universal, fácil de usar con OpenCV DNN)

```python
# Celda 6: Exportar a ONNX
model = YOLO("/content/drive/MyDrive/Colab Notebooks/IAAR/practica3_yolo/runs/tanque_v1/weights/best.pt")
model.export(format="onnx", imgsz=320, simplify=True)

# El archivo best.onnx se guardará junto al best.pt en la carpeta weights/
```

#### 4.2 (Alternativa más rápida) Exportar a NCNN

```python
# NCNN es el formato más rápido en ARM (Raspberry Pi)
model.export(format="ncnn", imgsz=320)
```

#### 4.3 Descargar y copiar a la RPi

```bash
# Desde tu PC, copiar el modelo a la Raspberry Pi
scp best.onnx practica@<IP_RASPI>:~/Freenove_Tank_Robot_Kit_for_Raspberry_Pi/Code/Server/
```

---

### FASE 5 — Script de inferencia en la RPi (~30 min de desarrollo)

#### 5.1 Script de inferencia YOLOv8 con OpenCV DNN

Ya creado en `Server/yolo_inferencia.py`. Carga el modelo ONNX y lo ejecuta con OpenCV, sin necesidad de instalar PyTorch ni ultralytics en la RPi.

La diferencia clave con YOLOv5 es cómo se lee la tabla de salida del modelo:
- YOLOv5 devuelve `[1, N, 7]` → cada fila: `[cx, cy, w, h, obj_conf, score0, score1]`
- YOLOv8 devuelve `[1, 6, 8400]` → hay que hacer `.T` (transponer) para obtener `[8400, 6]`, y cada fila es: `[cx, cy, w, h, score0, score1]` (sin obj_conf separado)

El script ya maneja esto internamente. Solo se usa `detector.detect(frame)` y devuelve las detecciones listas.

#### 5.2 Script principal integrado: PRACTICA_3_YOLO.py

```python
# PRACTICA_3_YOLO.py — Comportamiento deliberativo con YOLO
import cv2
import numpy as np
import time
import random
from motor import tankMotor
from camera import Camera
from servo import Servo
from ultrasonic import Ultrasonic
from yolo_inferencia import YOLODetector

# --- HARDWARE ---
motor = tankMotor()
servo_obj = Servo()
sonar = Ultrasonic()
detector = YOLODetector("best.onnx", conf_threshold=0.40)

# --- ESTADOS DEL ROBOT ---
ESTADO_BUSCAR = "BUSCAR"          # Explorar buscando bola
ESTADO_ACERCAR = "ACERCAR"        # Bola vista, acercándose
ESTADO_RECOGER = "RECOGER"        # Bola a distancia de pinza
ESTADO_EVADIR = "EVADIR"          # Evadiendo línea/obstáculo
ESTADO_SOLTAR = "SOLTAR"          # Soltando la bola

estado_actual = ESTADO_BUSCAR
bola_recogida = False

# --- FUNCIONES DE MOVIMIENTO (heredadas de Práctica 2) ---
def levantar_gancho():
    servo_obj.setServoAngle('0', 90)   # Pinza abierta
    servo_obj.setServoAngle('1', 140)  # Brazo arriba
    time.sleep(0.5)

def detener():
    motor.setMotorModel(0, 0)

def avanzar(velocidad=1000):
    factor_correccion = 1.2
    motor.setMotorModel(-velocidad, -velocidad * factor_correccion)

def girar_izquierda(velocidad=1500):
    motor.setMotorModel(velocidad, -velocidad)

def girar_derecha(velocidad=1500):
    motor.setMotorModel(-velocidad, velocidad)

def girar_aleatorio(tiempo_min=0.4, tiempo_max=1.0):
    direccion = random.choice(["izq", "der"])
    vel = 1500
    if direccion == "izq":
        motor.setMotorModel(vel, -vel)
    else:
        motor.setMotorModel(-vel, vel)
    time.sleep(random.uniform(tiempo_min, tiempo_max))
    detener()

def retroceder(tiempo=0.4):
    motor.setMotorModel(1200, 1200)
    time.sleep(tiempo)
    detener()

def coger_bola():
    """Secuencia completa: abrir pinza, bajar brazo, cerrar, subir."""
    servo_obj.setServoAngle('0', 90)   # Abrir pinza
    time.sleep(0.3)
    # Bajar brazo despacio
    for angle in range(150, 90, -2):
        servo_obj.setServoAngle('1', angle)
        time.sleep(0.02)
    time.sleep(0.3)
    servo_obj.setServoAngle('0', 135)  # Cerrar pinza
    time.sleep(0.5)
    servo_obj.setServoAngle('1', 140)  # Subir brazo
    time.sleep(0.5)

def soltar_bola():
    servo_obj.setServoAngle('0', 90)   # Abrir pinza
    time.sleep(0.5)

# --- BUCLE PRINCIPAL ---
def main():
    global estado_actual, bola_recogida
    
    print("=== PRÁCTICA 3: COMPORTAMIENTO DELIBERATIVO CON YOLO ===")
    levantar_gancho()
    
    cap = Camera(stream_size=(320, 240), hflip=True, vflip=True)
    cap.start_stream()
    time.sleep(1)
    
    tiempo_sin_bola = time.time()
    TIMEOUT_BUSQUEDA = 8  # Segundos antes de girar al buscar
    
    try:
        while True:
            frame_bytes = cap.get_frame()
            if frame_bytes is None:
                continue
            
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            
            h_frame, w_frame = frame.shape[:2]
            
            # --- DETECCIÓN YOLO ---
            detecciones = detector.detect(frame)
            bola_encontrada, bola_cx, bola_area = detector.get_ball_info(detecciones, w_frame)
            linea_visible, linea_peligro = detector.get_line_detected(detecciones, h_frame)
            
            # --- SONAR ---
            distancia = sonar.get_distance()
            if distancia < 0:
                distancia = 999.0
            
            # --- MÁQUINA DE ESTADOS ---
            
            # PRIORIDAD 1: Evasión (siempre activa)
            if linea_peligro or (0 <= distancia <= 15):
                if estado_actual != ESTADO_EVADIR:
                    print(f"\n⚠ EVASIÓN: {'Línea verde' if linea_peligro else f'Obstáculo {distancia:.0f}cm'}")
                estado_actual = ESTADO_EVADIR
                detener()
                time.sleep(0.2)
                retroceder(0.4)
                girar_aleatorio(0.5, 1.2)
                estado_actual = ESTADO_BUSCAR
                tiempo_sin_bola = time.time()
                continue
            
            # PRIORIDAD 2: Si tenemos bola recogida, soltarla tras avanzar un poco
            if bola_recogida:
                estado_actual = ESTADO_SOLTAR
                print("\n Soltando bola...")
                avanzar(800)
                time.sleep(1.5)
                detener()
                soltar_bola()
                bola_recogida = False
                retroceder(0.5)
                girar_aleatorio(0.8, 1.5)
                estado_actual = ESTADO_BUSCAR
                tiempo_sin_bola = time.time()
                continue
            
            # PRIORIDAD 3: Bola detectada → acercarse
            if bola_encontrada:
                tiempo_sin_bola = time.time()
                
                # ¿Está lo suficientemente cerca para recoger? (sonar ~6-7cm según log)
                if distancia <= 10 or bola_area > 0.08:
                    estado_actual = ESTADO_RECOGER
                    print(f"\n¡Recogiendo bola! (dist={distancia:.1f}cm, area={bola_area:.3f})")
                    detener()
                    time.sleep(0.3)
                    coger_bola()
                    bola_recogida = True
                    continue
                
                # Dirigirse hacia la bola
                estado_actual = ESTADO_ACERCAR
                error = bola_cx - 0.5  # Negativo = bola a la izquierda, positivo = derecha
                
                if abs(error) < 0.12:
                    # Centrada, avanzar recto
                    avanzar(800)
                elif error > 0:
                    # Bola a la derecha, girar derecha suave
                    motor.setMotorModel(-1200, -600)
                else:
                    # Bola a la izquierda, girar izquierda suave
                    motor.setMotorModel(-600, -1200)
                
                print(f"\r→ ACERCAR: cx={bola_cx:.2f} err={error:+.2f} dist={distancia:.0f}cm    ", end="")
            
            else:
                # PRIORIDAD 4: Buscar (explorar)
                estado_actual = ESTADO_BUSCAR
                
                elapsed = time.time() - tiempo_sin_bola
                if elapsed > TIMEOUT_BUSQUEDA:
                    # Llevamos mucho sin ver bola, girar para explorar
                    print(f"\n Girando para explorar ({elapsed:.0f}s sin bola)...")
                    girar_aleatorio(0.6, 1.5)
                    tiempo_sin_bola = time.time()
                else:
                    # Avanzar recto buscando
                    avanzar(800)
                    print(f"\r BUSCAR: {elapsed:.0f}s sin bola | dist={distancia:.0f}cm    ", end="")
    
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    finally:
        detener()
        soltar_bola()
        motor.close()
        servo_obj.setServoStop()
        sonar.close()
        cap.stop_stream()
        cap.close()
        print("Robot detenido de forma segura.")

if __name__ == '__main__':
    main()
```

---

### FASE 6 — Pruebas en el robot (~30 min)

#### 6.1 Checklist de pruebas

1. **Test de inferencia pura** — Sin motores, solo cámara + YOLO:
   ```python
   # test_yolo_solo.py
   from camera import Camera
   from yolo_inferencia import YOLODetector
   import numpy as np, cv2, time

   detector = YOLODetector("best.onnx")
   cap = Camera(stream_size=(320, 240), hflip=True, vflip=True)
   cap.start_stream()
   time.sleep(1)

   try:
       while True:
           frame_bytes = cap.get_frame()
           np_arr = np.frombuffer(frame_bytes, np.uint8)
           frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
           
           t0 = time.time()
           dets = detector.detect(frame)
           fps = 1.0 / (time.time() - t0)
           
           for d in dets:
               print(f"  {d[1]}: {d[2]:.2f} @ ({d[3]},{d[4]},{d[5]},{d[6]})")
           print(f"FPS: {fps:.1f} | Detecciones: {len(dets)}     ", end="\r")
   except KeyboardInterrupt:
       cap.stop_stream()
       cap.close()
   ```

2. **Verificar FPS** — Debería dar >2 FPS con ONNX en RPi. Si es <1 FPS, considerar reducir `img_size` a 256 o usar NCNN.

3. **Test con movimiento** — Ejecutar `PRACTICA_3_YOLO.py` con una bola visible para comprobar que se acerca.

4. **Test completo** — Poner bola detrás de obstáculo, comprobar que explora, encuentra, recoge y suelta.

---

## Resumen de archivos a crear/modificar

| Archivo | Ubicación | Propósito |
|---------|-----------|-----------|
| `fix_class_ids.py` | Server/ | Corregir IDs de clase del dataset |
| `dataset.yaml` | Colab | Config del dataset para YOLOv8 |
| `yolo_inferencia.py` | Server/ | Módulo de detección ONNX |
| `PRACTICA_3_YOLO.py` | Server/ | Script principal del robot |
| `test_yolo_solo.py` | Server/ | Test de inferencia sin motores |

## Cronograma para mañana

| Hora | Tarea | Duración |
|------|-------|----------|
| Llegada | Ejecutar `fix_class_ids.py`, verificar | 10 min |
| +10 min | Capturar ~150-200 fotos adicionales | 45-60 min |
| +1h10 | Subir todo a Drive, montar Colab | 15 min |
| +1h25 | Entrenar YOLOv8n (GPU Colab) | 30-45 min |
| +2h10 | Evaluar métricas + probar con vídeos | 15 min |
| +2h25 | Exportar ONNX, copiar a RPi | 10 min |
| +2h35 | Test de inferencia (sin motores) | 15 min |
| +2h50 | Test con movimiento completo | 30 min |
| +3h20 | Ajustar umbrales/velocidades | 20 min |
| **Total** | | **~3.5-4 horas** |

## Notas importantes

- **Los motores están invertidos:** valores negativos = avanzar, positivos = retroceder. El motor derecho necesita `factor_correccion = 1.2`.
- **Servos de la pinza:** Canal 0 = pinza (90°=abierta, 135°=cerrada), Canal 1 = brazo (140°=arriba, 90°=abajo).
- **Distancia sonar para recoger bola:** ~6-7 cm según el log del repo.
- **Resolución de cámara:** Siempre usar 320×240 (es la del stream, y la de entrenamiento).
- **El gancho debe estar levantado** (`servo '1' → 140`) para que no bloquee la cámara.
- **Cerrar con Ctrl+C limpiamente** para que los motores se detengan y la cámara se libere.
