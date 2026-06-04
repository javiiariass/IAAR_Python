# CLAUDE.md — Contexto y Planning del Proyecto IAAR Práctica 3

## Qué es este proyecto

Práctica 3 de la asignatura IAAR (Inteligencia Artificial Aplicada a la Robótica). Un tanque Freenove con Raspberry Pi 4 que usa YOLOv8n para detectar bolas rojas (recogerlas) y líneas verdes (bordes de mesa, evitar caerse). YOLO26n se probó pero dio peores resultados con nuestro dataset pequeño. El robot opera sobre una mesa con líneas verdes pintadas en los bordes, debe recoger múltiples bolas rojas en ~1 minuto, esquivar una caja obstáculo y no caerse.

> **Registro de cambios:** el log de cambios del proyecto se va añadiendo en
> `cambiosPrac3.md` (raíz del repo, lo más reciente abajo). Este `CLAUDE.md` mantiene el
> ESTADO y el PLANNING; `cambiosPrac3.md` mantiene el HISTÓRICO de qué se tocó y por qué.
> **Para ponerte al día, lee primero `cambiosPrac3.md` y el código de `Server/PRACTICA_3.py`.**
>
> **Fase actual: CALIBRACIÓN de valores en el robot real** (la arquitectura ya está hecha).
> Guía de medidas en curso: `Server/LAB_TAREAS_COMPANERO.md`.

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

### Script principal (ACTUAL):
- `Server/PRACTICA_3.py` — **script único** por capas (subsumption), solo NCNN, con flag `--stream`. **Es el que se usa.** Incluye modos de prueba: `--test-motor`, `--test-ir`, `--test-hsv`, `--test-percepcion`, y `--sin-motor` / `--log`.
- `Server/LAB_TAREAS_COMPANERO.md` — guía de la sesión de calibración en curso (qué medir y cómo).

### Scripts antiguos (LEGACY — anteriores a la unificación, ya no se usan):
- `Server/PRACTICA_3_YOLO.py`, `_STREAM.py`, `_NCNN.py`, `_STREAM_NCNN.py` — las 4 variantes que se unificaron en `PRACTICA_3.py`. Se conservan solo por referencia.

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

## Estado actual — `PRACTICA_3.py` (script único por capas)

Arquitectura de 3 capas YA implementada (ver PLANNING → YA HECHO). Resumen de la lógica
deliberativa (Capa 2):
- `tiempo_ultima_bola`: grace period de 1s para no confundir bola con obstáculo en el sonar.
- Recogida por **ÁREA de YOLO** (el sonar quedó solo para el obstáculo/caja).
- Centrado por **taps de pivote** (duty fijo `VEL_GIRO_BOLA`, duración proporcional al error) + **pausa** para que YOLO reevalúe.
- Aproximación final a **pulsos** (P-AVANCE) para no subirse a la bola a ~5 FPS.
- Retroceso fino/pulsado si la bola está demasiado cerca.
- Supresión del frenado por verde (`suprimir_linea`) cuando hay bola en aproximación/borde.
- `TIMEOUT_ACERCAR` anti-atasco. La Capa 0 (IR) siempre activa como backstop anti-caída.

### Constantes actuales (en `PRACTICA_3.py`) — EN CALIBRACIÓN
```
# Velocidades (duty 0-4095; NEGATIVO = avanzar)
VEL_EXPLORAR = 850, VEL_ACERCAR = 700, VEL_FRENADO = 400, VEL_GIRO = 1000, VEL_RETROCESO = 900
FACTOR_CORRECCION = 1.2
# Sonar (solo obstáculo)
DIST_RECOGER = 6.0, DIST_FRENAR = 20.0, DIST_OBSTACULO = 25.0, DIST_OBSTACULO_LEJOS = 35.0
# Bola (YOLO)
AREA_RECOGER = 0.150        # calibrado (~distancia de pinza)
BOLA_CENTRADA = 0.15        # margen de centrado
VEL_GIRO_BOLA = 1200        # duty del tap de pivote
RATIO_APROX_FINA = 0.6, PULSO_AVANCE = 0.10
PULSO_GIRO = 0.30, PULSO_GIRO_MAX = 0.50, PULSO_PAUSA = 0.20
VEL_RETROCESO_FINO = 800, TIMEOUT_ACERCAR = 7
# Línea (HSV): rango [40,50,50]-[85,255,255], ROI = tercio inferior
HSV_UMBRAL_PIXELES = 3000, RATIO_SUPRIMIR_LINEA = 0.5
# Tiempos / servo
TIMEOUT_BUSQUEDA = 20, PAUSA_TRAS_SOLTAR = 2, IR_PERIODO = 0.02, IR_RETROCESO_EXTRA = 0.25
PINZA_ABIERTA = 90, PINZA_CERRADA = 135, BRAZO_ARRIBA = 140, BRAZO_ABAJO = 90
```

## Ya implementado

- Arquitectura de **3 capas** (subsumption) + árbitro `MotorSeguro` por prioridad (IR > HSV > deliberativa).
- **Script único** `PRACTICA_3.py` (NCNN), con modos de prueba y `--stream` / `--sin-motor` / `--log`.
- **Ctrl+C rápido**: handler SIGINT + flag `running` + `dormir()` troceado + hilos daemon.
- **Recogida por área** de YOLO; sonar solo para el obstáculo.
- **Centrado por taps** de pivote + aproximación a pulsos; **timeout anti-atasco** en ACERCAR.
- Supresión de línea por bola en borde (hoy **booleana**, pendiente pasar a umbral dinámico).
- Vectorización de `detect()` con numpy; `num_threads=3` en NCNN (bajado de 4 para dejar un core al PWM por software). Umbrales de confianza por clase (`conf_bola=0.5`, `conf_linea=0.25`).

## Estado de pruebas y fallos actuales (en calibración)

**Estamos probando/ajustando valores en el robot real.** Fallos abiertos y su origen probable:

1. **Se queda "pillado" antes de avanzar/girar** (zumba como aplicando fuerza pero no se
   mueve; al rato arranca). Origen probable: **fricción estática (stiction)** — los duties de
   arranque desde parado son bajos (avance 700/400, taps a 1200) y a veces no rompen la
   fricción. Se suma posible **PWM por software inestable bajo carga**: gpiozero genera el PWM
   por software y con NCNN ocupando los cores el waveform tiembla → baja el par efectivo.
   Mitigado en parte alargando el tap mínimo de giro (`PULSO_GIRO` 0.10→0.30) y **bajando NCNN
   de 4 a 3 hilos** (deja un core al PWM). Pendiente: medir el duty de arranque real (Tarea 3
   del lab), confirmar si bajar hilos ayudó, y decidir subir duty / pigpio.
2. **Se aleja de la bola cuando podría cogerla.** Origen: la ventana de recogida era muy
   estrecha (retrocedía con `area > AREA_RECOGER*1.1` antes de poder coger). Mitigado subiendo
   el umbral de retroceso a 1.2× y ensanchando la recogida (bypass de centrado 1.5×→1.2×).
   Nota clave: **el área NO es monótona** — cerca BAJA porque la cámara está alta y la bola se
   sale por abajo del campo; por eso el punto de recogida se fija en área≈0.150 (no en "bola debajo").
3. **Bola en esquina con dos bboxes de línea que no solapan la bola.** YOLO marca esquina y el
   robot evade/abandona la bola aunque el camino recto esté libre. Sin resolver del todo; la
   idea es el **umbral de verde dinámico** (pendiente #1) en vez de la supresión por solape.
4. **Centrado lento / errático.** Es consecuencia del fallo #1 (stiction): los taps no arrancan
   fiable. Se atacará junto con la decisión de motor, NO tocando el algoritmo de centrado en sí.

## PLANNING — Estado y trabajo pendiente

> **Fase actual: CALIBRACIÓN de valores en el robot real.** La arquitectura ya está montada;
> ahora se miden/ajustan constantes (área de recogida, umbrales de verde, duty de arranque
> del motor) y se pulen fallos de control. Medidas en curso en `Server/LAB_TAREAS_COMPANERO.md`.

### YA HECHO

- **YOLO26n DESCARTADO**: con el dataset pequeño (146 train) da peor que v8n (mAP50 0.862 vs 0.951; precision 0.756 vs 0.972). Seguimos con v8n. Notebook: `entrenar_modelo_yolo26n.ipynb`.
- **Arquitectura de 3 capas (subsumption)** en `PRACTICA_3.py`:
  - Capa 0 — IR (hilo, prioridad ABSOLUTA): marcha atrás refleja, nunca se anula. `Infrared.read_one_infrared(1/2/3)`.
  - Capa 1 — HSV (hilo, dueño de la cámara): verde en el tercio inferior → frena/retrocede. Publica el frame para que YOLO no redecodifique. Rango [40,50,50]-[85,255,255].
  - Capa 2 — deliberativa (hilo principal): YOLO + sonar, FSM BUSCAR→ACERCAR→RECOGER→EVADIR.
  - Árbitro `MotorSeguro` por prioridad IR > HSV > deliberativa.
- **Script único** `PRACTICA_3.py`, solo NCNN, `--stream` opcional + modos de prueba.
- **Ctrl+C rápido**: SIGINT + flag `running` + `dormir()` troceado + hilos daemon. (El sonar de gpiozero en RPi 4 NO hace busy-wait; el bloqueo temido era de la RPi 5 con lgpio.)
- **Recogida por ÁREA de YOLO** (no por sonar: rebota mal en bolas pequeñas). Sonar solo para el obstáculo.
- **Centrado por taps de pivote** + aproximación final a pulsos (P-AVANCE) para no subirse a la bola a ~5 FPS.
- **Timeout de aproximación**: `TIMEOUT_ACERCAR` → si se atasca en ACERCAR, retrocede y re-busca.
- **Supresión de línea por bola en borde** (versión actual): `bola_en_borde` (intersección de bboxes) + `RATIO_SUPRIMIR_LINEA` (por área). OJO: hoy es **booleana** (apaga el HSV del todo) → a mejorar (pendiente #1).

### PENDIENTE

1. **Umbral de verde dinámico (rediseño de la supresión de línea).** Hoy `suprimir_linea` es un bool que pone `peligro=False` (apaga el HSV entero). Cambiar a **dos umbrales**: normal (~3000 px) y con-bola (~10-12k px, a medir en lab). El robot frena si verde > umbral, según haya o no bola en aproximación. **Sin** "techo de emergencia" por píxeles: el verde CAE a 0 cuando la línea se sale de la cámara (cámara alta), así que no se dispara — el backstop real es el **IR**. Bloqueado hasta medir el umbral con-bola (Tarea 2 del lab). Resuelve el fallo de la bola en esquina.
2. **Decisión motor / fricción (stiction).** Ya hecho: tap mínimo de giro más largo y **NCNN bajado de 4 a 3 hilos** (un core libre para el PWM). Tras medir el duty de arranque (Tarea 3), confirmar si bajar hilos ayudó y decidir subir duties (fuerza, p.ej. `VEL_GIRO_BOLA`) o no. pigpio (PWM por DMA, inmune a la carga de CPU) solo si sigue temblando.
3. **Verificación post-recogida.** Guardar `bola_area`/`bola_cx` antes de bajar la pinza; tras subir el brazo, comparar. Si el área bajó mucho / ya no se ve esa bola → éxito; si sigue grande en la misma posición → reintentar. Comparar ÁREA y POSICIÓN, no solo presencia (puede haber otra bola de fondo). (Durante la recogida la pinza tapa la cámara → ignorar percepción salvo IR.)
4. **Memoria de bola tras evasión.** Si estaba en ACERCAR y tiene que evadir, guardar el último `bola_cx` y, tras evadir, girar hacia ahí en vez de re-buscar a ciegas. El "escape de línea" actual puede abandonar una bola buena cerca del borde.

### Referencia

- **Prioridades en la Capa 2**: (1) evasión de línea YOLO / obstáculo sonar (solo si no hay bola visible/reciente), (2) bola: centrar con taps + recoger por área, retroceder si demasiado cerca, (3) explorar.
- **Streaming Freenove**: TCP 5003 (comandos) / 8003 (vídeo). Envío: `struct.pack('<I', len)` + JPEG. `TankServer` en `server.py`. Con `--stream` espera al cliente antes de arrancar.
- **Año anterior** (referencia cruda que aprobó): `Server/Pepe el marismeño/`. Corrían motores a ~2000 (sin problema de stiction), centrado **bang-bang de 3 cubos** (izq/centro/der a velocidad fija), recogida por sonar. Mucho más simple que lo nuestro — útil para no sobre-ingenierizar.

## Notas de usuario

- Javi es estudiante de ingeniería informática. Programa en C++, Java, SQL. Algo de ensamblador 8086.
- Prefiere planificar antes de codificar. No cambiar código "a lo loco".
- El robot debe funcionar en la RPi 4, no la 5.
- El dataset y vídeos de prueba están en Google Drive (ruta en el notebook).

