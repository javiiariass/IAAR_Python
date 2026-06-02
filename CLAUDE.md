# CLAUDE.md — Contexto y Planning del Proyecto IAAR Práctica 3

## Qué es este proyecto

Práctica 3 de la asignatura IAAR (Inteligencia Artificial Aplicada a la Robótica). Un tanque Freenove con Raspberry Pi 4 que usa YOLOv8n para detectar bolas rojas (recogerlas) y líneas verdes (bordes de mesa, evitar caerse). YOLO26n se probó pero dio peores resultados con nuestro dataset pequeño. El robot opera sobre una mesa con líneas verdes pintadas en los bordes, debe recoger múltiples bolas rojas en ~1 minuto, esquivar una caja obstáculo y no caerse.

## Hardware

- **Raspberry Pi 4** (NO es la 5)
- Motores invertidos: valores negativos = avanzar. Motor derecho más lento → FACTOR_CORRECCION = 1.2
- Motor API: `tankMotor.setMotorModel(duty_izq, duty_der)` — rango 0-4095
- Servo ch0 = pinza (90° abierta, 135° cerrada), ch1 = brazo (140° arriba, 90° abajo)
- Cámara 320×240 con hflip=True, vflip=True
- Sonar ultrasónico: `Ultrasonic.get_distance()` → cm, -1 si error
- 3 sensores infrarrojos frontales: `Infrared.read_one_infrared(1/2/3)` → 0 o 1
- La pinza tapa parcialmente la cámara al bajar el brazo

## Modelo YOLO

- Entrenado en Google Colab con GPU. Notebook: `entrenar_modelo.ipynb`
- 2 clases: 0 = bola_roja, 1 = linea_verde
- Entrenado con imgsz=320, dataset en Google Drive (`/content/drive/MyDrive/Colab Notebooks/IAAR/practica3_yolo/`)
- Resultados v8n: mAP50 = 0.951
- Output tensor con imgsz=320: shape (1, 6, 2100), NO 8400 (8400 es con 640)
- Tras transponer: [2100, 6] → cada fila: [cx, cy, w, h, score_bola, score_linea]
- Exportado a ONNX y NCNN. Ambos funcionan en la RPi.
- NCNN blob names: leer del .param file (ya resuelto, estuvimos haciendo inferencia)

## Archivos del proyecto

### Scripts principales (4 variantes, misma lógica):
- `Server/PRACTICA_3_YOLO.py` — ONNX, sin streaming
- `Server/PRACTICA_3_YOLO_STREAM.py` — ONNX, con streaming Freenove
- `Server/PRACTICA_3_YOLO_NCNN.py` — NCNN, sin streaming
- `Server/PRACTICA_3_YOLO_STREAM_NCNN.py` — NCNN, con streaming Freenove

### Módulos de inferencia:
- `Server/yolo_inferencia.py` — ONNX con OpenCV DNN. detect() vectorizado con numpy. cv2.setNumThreads(4). Umbrales de confianza por clase (conf_bola, conf_linea).
- `Server/yolo_inferencia_ncnn.py` — NCNN con ncnn.Net(). num_threads=4. Misma interfaz que ONNX. detect() vectorizado. Umbrales por clase.

### Visualización (sin motores, solo ver detecciones):
- `Server/Ver_YOLO_en_vivo.py` — ONNX
- `Server/Ver_YOLO_en_vivo_NCNN.py` — NCNN

### Scripts de la Práctica 2 (referencia para comportamiento reactivo):
- `Server/PRACTICA_2_solo_vision.py` — Detección de línea verde con HSV (saturación). Funciona bien.
- `Server/PRACTICA_2_solo_infrarrojos.py` — Test de los 3 sensores IR frontales.
- `Server/PRACTICA_2.py` — Versión completa con ambos.
- `Server/PRACTICA_2_con_sonar.py` — Con sonar añadido.

### Hardware APIs:
- `Server/motor.py` — tankMotor.setMotorModel(duty1, duty2), GPIO 24,23 (izq), 5,6 (der)
- `Server/servo.py` — Servo.setServoAngle(channel_str, angle)
- `Server/ultrasonic.py` — Ultrasonic.get_distance() → cm, -1 si error
- `Server/infrared.py` — Infrared.read_one_infrared(1/2/3)
- `Server/camera.py` — Camera(stream_size, hflip, vflip), .start_stream(), .get_frame(), .stop_stream(), .close()
- `Server/server.py` — TankServer TCP. Puerto 5003 (comandos), 8003 (vídeo). IP desde wlan0. Protocolo: 4 bytes longitud (little-endian) + datos JPEG.

### Otros:
- `entrenar_modelo.ipynb` — Notebook de Colab para entrenar YOLOv8n
- `entrenar_modelo_yolo26n.ipynb` — Notebook para YOLO26n (descartado, peores resultados)
- `PLAN_PRACTICA_3.md` — Plan original de la práctica
- `tanque_v1-2/` — Resultados del entrenamiento v8n (best.pt, métricas, matrices de confusión)
- `Server/0000_Instrucciones.md` — Instrucciones para capturar dataset y probar modelo

## Instrucciones para el modelo YOLOv8n

### Formato de salida

YOLOv8n exportado a ONNX con `imgsz=320` produce un tensor de forma `(1, 6, 2100)`:
- Eje 0: batch (siempre 1)
- Eje 1: 4 coordenadas (cx, cy, w, h) + 2 scores de clase (bola_roja, linea_verde)
- Eje 2: 2100 candidatos (las "cajas" que el modelo propone)
- Tras transponer queda `[2100, 6]`: cada fila es `[cx, cy, w, h, score_bola_roja, score_linea_verde]`
- A diferencia de YOLOv5, NO hay columna separada de obj_conf; los scores de clase ya son la confianza directa
- 2100 candidatos es con imgsz=320. Con 640 serían 8400. Los comentarios en el código deben reflejar esto.

### Postprocesado actual (vectorizado con numpy)

1. Transponer el tensor: `data = outputs[0].T` → shape `[2100, 6]`
2. Extraer scores: `class_scores = data[:, 4:]` → `[2100, 2]`
3. Para cada candidato: `class_id = argmax(scores)`, `confidence = max(scores)`
4. Prefiltro rápido con `min(conf_bola, conf_linea)` para descartar la mayoría
5. Filtro fino: cada candidato debe superar el umbral de SU clase (`conf_por_clase[class_id]`)
6. Convertir coordenadas `(cx, cy, w, h)` → `(x, y, w, h)` escaladas al tamaño original del frame
7. Aplicar NMS con `cv2.dnn.NMSBoxes()` para eliminar duplicados
8. Retornar lista de tuplas: `(class_id, class_name, confidence, x, y, w, h)`

### Interfaz de los módulos de inferencia

Tanto `YOLODetector` (ONNX) como `YOLODetectorNCNN` (NCNN) exponen la misma interfaz:

```python
detector = YOLODetector("best.onnx", conf_threshold=0.40, conf_bola=0.35, conf_linea=0.50)

# Detección
detecciones = detector.detect(frame)  # → [(class_id, name, conf, x, y, w, h), ...]

# Info de bola (la más grande = más cercana)
encontrada, centro_x, area = detector.get_ball_info(detecciones, frame_width)
# centro_x: 0.0=izq, 0.5=centro, 1.0=der
# area: (w*h) / (frame_width^2), cuanto mayor más cerca

# Info de línea
line_info = detector.get_line_info(detecciones, frame_width, frame_height)
# Retorna dict: detectada, peligro, posicion_x, posicion_y, esquina, num_peligrosas
# peligro=True solo si la línea está en el tercio inferior (frame_height * 2/3)
```

### Cómo entrenar / reentrenar

1. Abrir `entrenar_modelo.ipynb` en Google Colab (necesita GPU)
2. Las primeras celdas montan Drive y dividen el dataset (80/20 con seed 42)
3. El dataset está en `/content/drive/MyDrive/Colab Notebooks/IAAR/practica3_yolo/`
4. Entrenar: `model.train(data=..., epochs=100, imgsz=320, batch=32, patience=20)`
5. Validar: comparar mAP50, precision, recall
6. Probar con vídeos: `model.predict(source=".../videos_prueba/", ...)`
7. Exportar a ONNX: `model.export(format="onnx", imgsz=320, simplify=True)`
8. Exportar a NCNN: `model.export(format="ncnn", imgsz=320)`
9. Copiar `best.onnx` y `best_ncnn_model/` a `Server/` en la RPi

## Estado actual de los 4 scripts principales

Todos sincronizados con la misma lógica:
- `tiempo_ultima_bola`: grace period de 1s para no confundir bola con obstáculo en sonar
- Lógica combinada sonar+cámara para recogida (ambos deben confirmar, no OR)
- Retroceder si bola demasiado cerca (area > AREA_RECOGER * 3)
- Supresión de evasión de línea cuando hay bola cerca (posicion_y > 0.80)
- Giro proporcional al error de centrado (factor_lenta varía con el error)
- Frenar si cualquiera de los sensores indica cercanía
- Versiones STREAM esperan conexión del cliente antes de arrancar

### Constantes actuales (en los 4 scripts):
```
VEL_EXPLORAR = 900
VEL_ACERCAR = 800
VEL_FRENADO = 350
VEL_GIRO = 1200
FACTOR_CORRECCION = 1.2
DIST_RECOGER = 7.0 cm
DIST_FRENAR = 15.0 cm
DIST_OBSTACULO = 15.0 cm
DIST_OBSTACULO_LEJOS = 30.0 cm
AREA_RECOGER = 0.05
TIMEOUT_BUSQUEDA = 6 s
PAUSA_TRAS_SOLTAR = 1.5 s
PINZA_ABIERTA = 90, PINZA_CERRADA = 135
BRAZO_ARRIBA = 140, BRAZO_ABAJO = 90
```

## Ya implementado

- Vectorización de detect() con numpy en ambos módulos (ONNX y NCNN) — eliminado el for-loop sobre 2100 candidatos
- cv2.setNumThreads(4) en ONNX, num_threads=4 en NCNN
- Umbrales de confianza por clase: conf_bola, conf_linea (constructor de YOLODetector y YOLODetectorNCNN)
- Giro proporcional al error de centrado (factor_lenta = max(0.0, 0.65 - abs(error) * 1.5))
- Lógica combinada sonar+cámara para recogida
- Umbral de línea peligrosa: tercio inferior del frame (frame_height * 2/3)
- Las 4 variantes de script sincronizadas

## Problemas observados en lab

1. **Robot no esquivaba línea verde** → Corregido cambiando umbral. Se revertió a 2/3 porque 1/2 era demasiado agresivo.
2. **Robot evitaba bolas rojas** → Sonar confundía bola con obstáculo. Corregido con tiempo_ultima_bola (grace period 1s).
3. **Giro suave demasiado lento** → Factor 0.3 insuficiente. Cambiado a 0.55 y luego a giro proporcional.
4. **Robot no cogía bola** → Condición OR (sonar o cámara) causaba falsos positivos. Cambiado a AND con fallback.
5. **Bola en esquina, robot evade línea** → Supresión de evasión cuando hay bola cerca.
6. **Centrado de bola muy lento** → Umbral 0.10 muy estricto y giro fijo. Cambiado a 0.15 + giro proporcional.
7. **YOLO no detecta línea en algún frame → robot casi se cae** → Motivó el plan de añadir capa reactiva HSV.
8. **Ctrl+C tarda mucho** → Operaciones bloqueantes (sonar, sleeps en secuencias de evasión). Planificado handler SIGINT + flag.
9. **~5 FPS con NCNN** → Vectorización ya aplicada. Pendiente medir mejora.
10. **4 scripts duplicados** → Cada cambio hay que aplicarlo 4 veces. Planificado unificar.

## PLANNING — Reestructuración pendiente

### 1. ~~Probar YOLO26n~~ — DESCARTADO

Probado. Con nuestro dataset pequeño (146 train), YOLO26n da peores resultados que v8n:
- mAP50: 0.862 (v8n: 0.951), precision: 0.756 (v8n: 0.972)
- Seguimos con YOLOv8n. Notebook de prueba: `entrenar_modelo_yolo26n.ipynb`

### 2. Arquitectura de tres capas

Separar seguridad de estrategia. La seguridad NO debe depender de YOLO.

**Capa 0 — IR (hilo propio):**
- Los 3 infrarrojos frontales. Si alguno salta → marcha atrás inmediata.
- No se anula NUNCA bajo ninguna circunstancia. Prioridad absoluta.
- API: `Infrared.read_one_infrared(1/2/3)` — 1 = línea detectada.

**Capa 1 — HSV reactiva (hilo propio):**
- Saturación de verde en el tercio inferior del frame (código de Práctica 2).
- Barata (microsegundos), determinista, independiente de YOLO.
- Detecta "hay verde peligroso sí/no". Si sí → frenar y retroceder.
- Se puede suprimir PARCIALMENTE cuando YOLO detecta bola cerca de línea (intersección de bboxes).
- Comunicación con hilo principal: flag `suprimir_linea` (True/False).
- Código de referencia en `PRACTICA_2_solo_vision.py`: HSV range [40,50,50]-[85,255,255], umbral 3000 píxeles.

**Capa 2 — Deliberativa (hilo principal):**
- YOLO + sonar. Máquina de estados: BUSCAR → ACERCAR → RECOGER → EVADIR.
- YOLO para: identificar bola/línea, posición, tamaño, intersección de bboxes.
- Sonar para: distancia a objetos.

### 3. Prioridades dentro de la capa deliberativa

1. **Obstáculo (sonar):** esquivar suavemente si lejos (DIST_OBSTACULO_LEJOS), parada de emergencia si cerca (DIST_OBSTACULO). Solo si no hay bola visible/reciente.
2. **Bola (YOLO + sonar combinados):**
   - Sonar para distancia, YOLO para identificar y centrar
   - Giro proporcional al error
   - Retroceder si demasiado cerca
   - Recoger cuando ambos confirman
   - Frenar si cualquiera indica cercanía
3. **Nada** → explorar avanzando, girar si >6s sin ver bola

### 4. Bola cerca de línea (intersección de bboxes)

- Usar intersección/solapamiento de bboxes de bola y línea para determinar si la bola está en el borde de la mesa.
- Si hay intersección: suprimir parcialmente la reactividad HSV verde → permitir acercarse lento.
- Cuanto más grande la bola (más cerca del robot), más se puede suprimir.
- Se comunica del hilo principal al hilo reactivo con flag `suprimir_linea`.

### 5. Secuencia de recogida

- Activar flag `recogiendo = True` que deshabilita todo EXCEPTO IR.
- Al bajar el brazo la pinza tapa la cámara → YOLO ve basura, HSV cambia. Ignorar todo.
- Verificación post-recogida:
  - Guardar `bola_area` y `bola_cx` justo antes de bajar pinza.
  - Tras subir brazo, hacer un frame y comparar.
  - Si YOLO no ve bola (o el área bajó drásticamente) → éxito, la pinza se la llevó.
  - Si sigue viendo bola grande en la misma posición → fallo, reintentar.
  - Ojo: si hay otra bola de fondo, YOLO la verá. Comparar ÁREA y POSICIÓN, no solo presencia.

### 6. Memoria de bola tras evasión

- Si el robot estaba en ACERCAR y tiene que evadir (línea/obstáculo), guardar último `bola_cx`.
- Después de evadir, en vez de buscar aleatoriamente, girar hacia donde la vio por última vez.

### 7. Timeout de aproximación

- Si lleva >X segundos en estado ACERCAR sin llegar a RECOGER, algo va mal.
- Volver a BUSCAR para no quedarse atascado indefinidamente.

### 8. Ctrl+C rápido

- Handler de `signal.SIGINT` que pone flag global `running = False`.
- Todos los `time.sleep()` se reemplazan por sleeps cortos en bucle que comprueban el flag.
- Las funciones de evasión y recogida comprueban el flag entre pasos.
- Los hilos reactivos también comprueban el flag para terminar limpiamente.
- Problema actual: sonar.get_distance() puede bloquear (busy-wait esperando eco). Poner timeout agresivo.

### 9. Script único

- Unificar los 4 scripts en uno solo con parámetros: `--ncnn`, `--stream`.
- O al menos extraer la lógica común (movimiento, servo, evasión, bucle principal) a un módulo compartido.
- Evitar aplicar cada cambio 4 veces.

### 10. Streaming Freenove

- Protocolo TCP: puerto 5003 (comandos), 8003 (vídeo).
- Envío: 4 bytes longitud (little-endian `struct.pack('<I', len)`) + datos JPEG.
- Versiones STREAM esperan conexión del cliente antes de arrancar el robot.
- `TankServer` de `server.py` maneja los sockets.

## Notas de usuario

- Javi es estudiante de ingeniería informática. Programa en C++, Java, SQL. Algo de ensamblador 8086.
- Prefiere planificar antes de codificar. No cambiar código "a lo loco".
- El robot debe funcionar en la RPi 4, no la 5.
- El dataset y vídeos de prueba están en Google Drive (ruta en el notebook).

