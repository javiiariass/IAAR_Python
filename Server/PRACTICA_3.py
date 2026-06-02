"""
PRACTICA_3.py — Arquitectura de 3 capas (subsumption) con YOLO NCNN.

Separa SEGURIDAD de ESTRATEGIA. La seguridad NO depende de YOLO.

  Capa 0 — IR (hilo propio, prioridad ABSOLUTA):
      Los 3 infrarrojos frontales. Si alguno salta → marcha atrás refleja.
      Nunca se anula. Reacciona en ~ms, independiente de YOLO.

  Capa 1 — HSV reactiva (hilo propio, dueño de la cámara):
      Saturación de verde en el tercio inferior del frame (código Práctica 2).
      Barata y determinista. Si hay verde peligroso → frena y retrocede.
      Se puede suprimir (flag suprimir_linea) cuando la bola está en el borde.

  Capa 2 — Deliberativa (hilo principal):
      YOLO + sonar. Máquina de estados BUSCAR → ACERCAR → RECOGER → EVADIR.
      Solo conduce cuando NO hay emergencia de una capa superior.

El acceso al motor lo arbitra MotorSeguro por prioridad (IR > HSV > deliberativa):
una capa de seguridad puede frenar/retroceder DIRECTO con baja latencia, mientras
los comandos de la deliberativa quedan en no-op hasta que se despeje el peligro.

---------------------------------------------------------------------
USO (normal):
  sudo python PRACTICA_3.py                  → 3 capas, sin streaming
  sudo python PRACTICA_3.py --bolas 5        → para tras recoger 5 bolas
  sudo python PRACTICA_3.py --stream         → con vídeo al cliente Freenove

MODOS DE PRUEBA (para validar cada parte por separado en el laboratorio):
  sudo python PRACTICA_3.py --test-motor       → cableado motor + árbitro de prioridad
  sudo python PRACTICA_3.py --test-ir          → los 3 IR + marcha atrás refleja
  sudo python PRACTICA_3.py --test-hsv         → detección de verde HSV + frenado
  sudo python PRACTICA_3.py --test-percepcion  → YOLO + decisiones, SIN mover (tuneo)
  (cualquiera con --stream para ver el vídeo; --sin-motor para no mover el motor)

Requisitos en la RPi:
  pip install ncnn --break-system-packages
Archivos del modelo en Server/:
  best_ncnn_model/model.ncnn.param
  best_ncnn_model/model.ncnn.bin
"""
import cv2
import numpy as np
import time
import random
import argparse
import struct
import signal
import threading

from motor import tankMotor
from servo import Servo
from ultrasonic import Ultrasonic
from infrared import Infrared
from camera import Camera
# YOLODetectorNCNN y TankServer se importan de forma PEREZOSA (dentro de las
# funciones que los necesitan) para que los modos --test-ir / --test-motor
# funcionen aunque ncnn o el servidor no estén disponibles.

# =====================================================================
# CONSTANTES — Ajustar según pruebas en el robot real
# =====================================================================

# Velocidades del motor (duty cycle, rango 0-4095)
# Nota: valores NEGATIVOS = avanzar (motores invertidos)
VEL_EXPLORAR = 850       # Velocidad al buscar bola (no muy alta para que la cámara no vibre)
VEL_ACERCAR = 700        # Velocidad al dirigirse hacia una bola vista
VEL_FRENADO = 600        # Velocidad de aproximación final (sonar < 15cm)
VEL_GIRO = 1200          # Velocidad de giro sobre sí mismo
VEL_RETROCESO = 900      # Velocidad de marcha atrás (reflejos de seguridad y maniobras)

# Factor de corrección del motor derecho (el derecho gira más lento)
FACTOR_CORRECCION = 1.2

# Sonar
DIST_RECOGER = 10.0      # Distancia (cm) a la que la bola está al alcance de la pinza
DIST_FRENAR = 15.0       # Distancia (cm) a la que empezar a reducir velocidad
DIST_OBSTACULO = 20.0    # Distancia (cm) para parada de emergencia por obstáculo
DIST_OBSTACULO_LEJOS = 35.0  # Distancia (cm) para empezar a esquivar suavemente

# Bola (detección YOLO)
AREA_RECOGER = 0.06      # Área relativa de la bola para considerar "suficientemente cerca"

# Línea verde (HSV, Capa 1) — del PRACTICA_2_solo_vision.py
HSV_VERDE_BAJO = (40, 50, 50)
HSV_VERDE_ALTO = (85, 255, 255)
HSV_UMBRAL_PIXELES = 3000    # Píxeles verdes en el ROI para considerar "peligro"
HSV_ROI_DESDE = 2.0 / 3.0    # ROI = tercio inferior del frame

# Tiempos
TIMEOUT_BUSQUEDA = 10    # Segundos sin ver bola antes de girar para explorar
PAUSA_TRAS_SOLTAR = 2    # Segundos de espera después de soltar bola
IR_PERIODO = 0.02        # Periodo de muestreo del hilo IR (~50 Hz)
IR_RETROCESO_EXTRA = 0.25  # Retroceso adicional al despejarse el IR (separa de la línea)

# Servo ángulos
PINZA_ABIERTA = 90
PINZA_CERRADA = 135
BRAZO_ARRIBA = 140   # probar con 150
BRAZO_ABAJO = 90


# =====================================================================
# ESTADO COMPARTIDO ENTRE HILOS
# =====================================================================

class Estado:
    """Flags compartidos entre las 3 capas, protegidos por un Lock.

    En CPython la lectura/escritura de un bool es atómica por el GIL, así que
    los flags se usan sin lock; el Lock protege solo el frame (objeto grande).
    """
    def __init__(self):
        self.running = True          # SIGINT lo pone a False → todos los bucles terminan
        self.peligro_ir = False      # Capa 0 → todas
        self.peligro_hsv = False     # Capa 1 → deliberativa
        self.suprimir_linea = False  # Capa 2 → Capa 1 (bola en el borde: no frenes)
        self._frame = None           # último frame BGR publicado por el hilo HSV
        self._frame_lock = threading.Lock()

    def set_frame(self, frame):
        with self._frame_lock:
            self._frame = frame

    def get_frame(self):
        with self._frame_lock:
            return self._frame


# =====================================================================
# ÁRBITRO DEL MOTOR POR PRIORIDAD (IR > HSV > deliberativa)
# =====================================================================

class MotorSeguro:
    """Arbitra el acceso al motor entre los 3 hilos por prioridad.

    Protocolo:
      - Las capas de seguridad (ir, hsv) hacen tomar() antes de mandar y
        soltar() al despejarse el peligro.
      - La deliberativa nunca toma: solo llama a mover("deliberativa", ...),
        que queda en no-op mientras una capa superior tenga el control.

    Con activo=False (modo --sin-motor) arbitra igual pero no toca el hardware,
    útil para probar percepción/lógica sin que el robot se mueva.
    """
    PRIORIDAD = {"deliberativa": 1, "hsv": 2, "ir": 3}

    def __init__(self, motor, activo=True):
        self._motor = motor
        self._activo = activo
        self._lock = threading.Lock()
        self._dueno = None  # capa con el control de emergencia, o None

    def _prio(self, capa):
        return self.PRIORIDAD.get(capa, 0)

    def _puede(self, capa):
        return self._dueno is None or self._prio(capa) >= self._prio(self._dueno)

    def mover(self, capa, izq, der):
        """Aplica un comando solo si nadie de mayor prioridad manda. Devuelve True si se aplicó."""
        with self._lock:
            if not self._puede(capa):
                return False
            if self._activo and self._motor is not None:
                self._motor.setMotorModel(izq, der)
            return True

    def tomar(self, capa):
        """Una capa de seguridad reclama el control exclusivo. True si lo consigue."""
        with self._lock:
            if not self._puede(capa):
                return False
            self._dueno = capa
            return True

    def soltar(self, capa):
        """Libera el control si esta capa lo tenía."""
        with self._lock:
            if self._dueno == capa:
                self._dueno = None

    def dueno(self):
        with self._lock:
            return self._dueno

    def parar_forzado(self):
        """Parada incondicional (para el cierre limpio)."""
        if self._motor is not None:
            self._motor.setMotorModel(0, 0)


# =====================================================================
# UTILIDAD: dormir interrumpible (para Ctrl+C rápido — problema #8)
# =====================================================================

def dormir(segundos, estado):
    """Como time.sleep() pero comprueba estado.running en trozos cortos."""
    fin = time.time() + segundos
    while estado.running:
        restante = fin - time.time()
        if restante <= 0:
            break
        time.sleep(min(0.02, restante))


# =====================================================================
# FUNCIONES DE MOVIMIENTO (todas pasan por el árbitro: capa + MotorSeguro)
# =====================================================================

def avanzar(motor, capa, velocidad):
    """Avanzar recto. Velocidad positiva = avanzar (se invierte internamente)."""
    motor.mover(capa, -velocidad, int(-velocidad * FACTOR_CORRECCION))


def girar_izquierda(motor, capa, velocidad=VEL_GIRO):
    motor.mover(capa, velocidad, -velocidad)


def girar_derecha(motor, capa, velocidad=VEL_GIRO):
    motor.mover(capa, -velocidad, velocidad)


def girar_suave_izquierda(motor, capa, velocidad=VEL_ACERCAR):
    motor.mover(capa, -int(velocidad * 0.5), -int(velocidad * FACTOR_CORRECCION))


def girar_suave_derecha(motor, capa, velocidad=VEL_ACERCAR):
    motor.mover(capa, -velocidad, -int(velocidad * 0.5 * FACTOR_CORRECCION))


def detener(motor, capa):
    motor.mover(capa, 0, 0)


def retroceder(motor, capa, estado, tiempo=0.4):
    """Retroceder un poco (valores positivos = marcha atrás)."""
    motor.mover(capa, VEL_RETROCESO, int(VEL_RETROCESO * FACTOR_CORRECCION))
    dormir(tiempo, estado)
    detener(motor, capa)


# =====================================================================
# FUNCIONES DE SERVO (PINZA Y BRAZO)
# =====================================================================

def levantar_gancho(servo):
    """Posición inicial: pinza abierta, brazo arriba."""
    servo.setServoAngle('0', PINZA_ABIERTA)
    servo.setServoAngle('1', BRAZO_ARRIBA)
    time.sleep(0.5)


def coger_bola(servo, estado):
    """
    Secuencia de recogida: abrir pinza → bajar brazo despacio → cerrar → subir.
    IMPORTANTE: la pinza tapa la cámara al bajar, así que el robot debe estar
    bien posicionado ANTES de llamar a esta función. La Capa 0 (IR) sigue activa.
    """
    servo.setServoAngle('0', PINZA_ABIERTA)
    dormir(0.3, estado)

    for angle in range(BRAZO_ARRIBA, BRAZO_ABAJO, -2):
        servo.setServoAngle('1', angle)
        time.sleep(0.02)
    dormir(0.3, estado)

    servo.setServoAngle('0', PINZA_CERRADA)
    dormir(0.5, estado)

    servo.setServoAngle('1', BRAZO_ARRIBA)
    dormir(0.5, estado)


def soltar_bola(servo):
    """Abrir pinza para soltar la bola."""
    servo.setServoAngle('0', PINZA_ABIERTA)
    time.sleep(0.5)


# =====================================================================
# PERCEPCIÓN HSV (Capa 1) — del PRACTICA_2_solo_vision.py
# =====================================================================

def contar_verde(frame):
    """Cuenta píxeles verdes en el tercio inferior del frame. Devuelve (peligro, n_pixeles)."""
    altura, anchura = frame.shape[:2]
    roi = frame[int(altura * HSV_ROI_DESDE):altura, 0:anchura]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, np.array(HSV_VERDE_BAJO), np.array(HSV_VERDE_ALTO))
    n = cv2.countNonZero(mascara)
    return n > HSV_UMBRAL_PIXELES, n


# =====================================================================
# LÓGICA DE EVASIÓN DE LÍNEA (Capa 2, estratégica — YOLO)
# La Capa 1 (HSV) ya hace el reflejo de no caerse; esto da el giro de salida.
# =====================================================================

def evadir_linea(motor, capa, estado, line_info):
    """Retrocede y gira en dirección OPUESTA a la línea según line_info de YOLO."""
    pos_x = line_info["posicion_x"]
    esquina = line_info["esquina"]

    detener(motor, capa)
    dormir(0.1, estado)
    retroceder(motor, capa, estado, 0.5)

    if esquina:
        # Esquina: líneas a ambos lados → girar mucho hacia donde haya más espacio
        retroceder(motor, capa, estado, 0.3)
        if pos_x > 0.5:
            girar_izquierda(motor, capa, VEL_GIRO)
        else:
            girar_derecha(motor, capa, VEL_GIRO)
        dormir(random.uniform(0.8, 1.3), estado)
    elif pos_x > 0.6:
        girar_izquierda(motor, capa, VEL_GIRO)
        dormir(random.uniform(0.4, 0.8), estado)
    elif pos_x < 0.4:
        girar_derecha(motor, capa, VEL_GIRO)
        dormir(random.uniform(0.4, 0.8), estado)
    else:
        if random.random() > 0.5:
            girar_izquierda(motor, capa, VEL_GIRO)
        else:
            girar_derecha(motor, capa, VEL_GIRO)
        dormir(random.uniform(0.5, 1.0), estado)

    detener(motor, capa)


def bola_en_borde(detecciones, bola_encontrada, bola_area):
    """¿La bola más grande solapa (bbox) con alguna línea verde? → está en el borde.

    Se usa para activar suprimir_linea: si la bola está justo en el borde de la
    mesa, dejamos que la Capa 2 se acerque despacio en vez de que la Capa 1 frene.
    Solo si la bola ya es razonablemente cercana (área).
    """
    if not bola_encontrada or bola_area <= AREA_RECOGER * 0.5:
        return False
    bolas = [d for d in detecciones if d[0] == 0]
    lineas = [d for d in detecciones if d[0] == 1]
    if not bolas or not lineas:
        return False
    _, _, _, bx, by, bw, bh = max(bolas, key=lambda d: d[5] * d[6])
    for _, _, _, lx, ly, lw, lh in lineas:
        ix = min(bx + bw, lx + lw) - max(bx, lx)
        iy = min(by + bh, ly + lh) - max(by, ly)
        if ix > 0 and iy > 0:
            return True
    return False


# =====================================================================
# STREAMING AL CLIENTE FREENOVE (solo con --stream)
# =====================================================================

COLOR_BOLA = (0, 0, 255)      # Rojo
COLOR_LINEA = (0, 255, 0)     # Verde
COLOR_ESTADO = (0, 255, 255)  # Amarillo
COLOR_TEXTO = (255, 255, 255) # Blanco


def dibujar_detecciones(frame, detecciones, estado_fsm, distancia, bolas_recogidas,
                        estado, motor):
    """Dibuja bounding boxes + estado del robot + flags de las capas sobre el frame."""
    for class_id, nombre, confianza, x, y, w, h in detecciones:
        color = COLOR_BOLA if class_id == 0 else COLOR_LINEA
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        etiqueta = f"{nombre} {confianza:.0%}"
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(frame, etiqueta, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXTO, 1)

    cv2.putText(frame, f"{estado_fsm} | Sonar:{distancia:.0f}cm | Bolas:{bolas_recogidas}",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_ESTADO, 1)
    flags = (f"IR:{int(estado.peligro_ir)} HSV:{int(estado.peligro_hsv)} "
             f"SUP:{int(estado.suprimir_linea)} dueno:{motor.dueno()}")
    cv2.putText(frame, flags, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_ESTADO, 1)
    return frame


def enviar_frame(tcp_server, frame):
    """Envía un frame anotado al cliente de Freenove si está conectado."""
    if not tcp_server.isVideoServerConnected():
        return
    try:
        _, jpeg = cv2.imencode('.jpg', frame)
        datos = jpeg.tobytes()
        tcp_server.sendDataToVideoClient(struct.pack('<I', len(datos)))
        tcp_server.sendDataToVideoClient(datos)
    except Exception:
        pass


# =====================================================================
# CAPA 0 — HILO INFRARROJOS (prioridad absoluta)
# =====================================================================

def hilo_ir(estado, motor, infrared):
    """Si cualquier IR frontal salta → marcha atrás refleja. Nunca se anula."""
    while estado.running:
        try:
            disparo = (infrared.read_one_infrared(1) or
                       infrared.read_one_infrared(2) or
                       infrared.read_one_infrared(3))
        except Exception:
            disparo = 0

        if disparo:
            estado.peligro_ir = True
            motor.tomar("ir")
            motor.mover("ir", VEL_RETROCESO, int(VEL_RETROCESO * FACTOR_CORRECCION))
        elif estado.peligro_ir:
            # Acabamos de despejar: un poco más de retroceso para separarnos y soltar
            motor.mover("ir", VEL_RETROCESO, int(VEL_RETROCESO * FACTOR_CORRECCION))
            dormir(IR_RETROCESO_EXTRA, estado)
            motor.mover("ir", 0, 0)
            motor.soltar("ir")
            estado.peligro_ir = False

        time.sleep(IR_PERIODO)

    motor.soltar("ir")


# =====================================================================
# CAPA 1 — HILO HSV REACTIVO (además, dueño de la cámara)
# =====================================================================

def hilo_hsv(estado, motor, camera):
    """Lee la cámara, publica el frame y frena/retrocede si hay verde peligroso.

    Es el ÚNICO lector de la cámara: publica estado.frame para que la Capa 2
    (YOLO) consuma el último frame sin bloquearse ni redecodificar.
    """
    while estado.running:
        fb = camera.get_frame()
        if fb is None:
            continue
        frame = cv2.imdecode(np.frombuffer(fb, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        estado.set_frame(frame)

        peligro, _ = contar_verde(frame)
        if estado.suprimir_linea:
            peligro = False  # Capa 2 pidió no frenar (bola en el borde)
        estado.peligro_hsv = peligro

        if peligro:
            # Reflejo: retroceder mientras siga viéndose verde abajo. Sin sleeps
            # largos para seguir siendo reactivo y ceder al IR si hace falta.
            if motor.tomar("hsv"):
                motor.mover("hsv", VEL_RETROCESO, int(VEL_RETROCESO * FACTOR_CORRECCION))
        else:
            motor.soltar("hsv")

    motor.soltar("hsv")


# =====================================================================
# CAPA 2 — DELIBERATIVA (hilo principal): YOLO + sonar, máquina de estados
# =====================================================================

def capa2_deliberativa(estado, motor, servo, sonar, detector, tcp_server, total_bolas):
    bolas_recogidas = 0
    estado_fsm = "BUSCAR"
    tiempo_sin_bola = time.time()
    tiempo_ultima_bola = 0          # Última vez que YOLO vio bola (grace period sonar)
    dir_esquiva_obstaculo = None    # "izq"/"der", se fija al detectar obstáculo
    en_seguridad = False            # Para imprimir el cambio a SEGURIDAD una sola vez

    while estado.running:
        if total_bolas > 0 and bolas_recogidas >= total_bolas:
            print(f"\n¡Objetivo cumplido! {bolas_recogidas}/{total_bolas} bolas recogidas.")
            break

        frame = estado.get_frame()
        if frame is None:
            dormir(0.02, estado)
            continue

        h_frame, w_frame = frame.shape[:2]

        # --- Percepción ---
        detecciones = detector.detect(frame)
        bola_encontrada, bola_cx, bola_area = detector.get_ball_info(detecciones, w_frame)
        line_info = detector.get_line_info(detecciones, w_frame, h_frame)

        distancia = sonar.get_distance()
        if distancia < 0:
            distancia = 999.0

        # Bola en el borde → pedir a la Capa 1 que NO frene (acercamiento lento)
        estado.suprimir_linea = bola_en_borde(detecciones, bola_encontrada, bola_area)

        # --- Streaming ---
        if tcp_server is not None:
            frame_anotado = dibujar_detecciones(
                frame.copy(), detecciones, estado_fsm, distancia,
                bolas_recogidas, estado, motor)
            enviar_frame(tcp_server, frame_anotado)

        # ==========================================================
        # CEDER A SEGURIDAD: si una capa superior tiene una emergencia,
        # la deliberativa no conduce (sus mover() serían no-op de todos modos).
        # ==========================================================
        if estado.peligro_ir or estado.peligro_hsv:
            if not en_seguridad:
                cual = "IR" if estado.peligro_ir else "HSV"
                print(f"\n■ SEGURIDAD ({cual}): cediendo control...")
                en_seguridad = True
            estado_fsm = "SEGURIDAD"
            dormir(0.02, estado)
            continue
        en_seguridad = False

        # ==========================================================
        # PRIORIDAD 1: EVASIÓN DE LÍNEA (YOLO, giro estratégico de salida)
        # ==========================================================
        if line_info["peligro"]:
            linea_inminente = line_info["posicion_y"] > 0.80
            bola_cerca = bola_encontrada and bola_area > AREA_RECOGER * 0.5
            if bola_cerca:
                evadir_ahora = linea_inminente
            else:
                evadir_ahora = not bola_encontrada or linea_inminente

            if evadir_ahora:
                if estado_fsm != "EVADIR":
                    esquina_txt = " (ESQUINA)" if line_info["esquina"] else ""
                    print(f"\n⚠ EVADIR: línea en x={line_info['posicion_x']:.2f} "
                          f"y={line_info['posicion_y']:.2f}{esquina_txt}")
                estado_fsm = "EVADIR"
                evadir_linea(motor, "deliberativa", estado, line_info)
                estado_fsm = "BUSCAR"
                tiempo_sin_bola = time.time()
                continue

        # ==========================================================
        # PRIORIDAD 1b: OBSTÁCULO POR SONAR (caja). Solo si no hay bola.
        # ==========================================================
        bola_reciente = (time.time() - tiempo_ultima_bola) < 1.0
        if not bola_encontrada and not bola_reciente and 0 < distancia < DIST_OBSTACULO_LEJOS:
            if dir_esquiva_obstaculo is None:
                dir_esquiva_obstaculo = random.choice(["izq", "der"])

            if distancia < DIST_OBSTACULO:
                if estado_fsm != "EVADIR":
                    print(f"\n⚠ OBSTÁCULO CERCA: sonar={distancia:.1f}cm")
                estado_fsm = "EVADIR"
                detener(motor, "deliberativa")
                dormir(0.1, estado)
                retroceder(motor, "deliberativa", estado, 0.3)
                if dir_esquiva_obstaculo == "izq":
                    girar_izquierda(motor, "deliberativa", VEL_GIRO)
                else:
                    girar_derecha(motor, "deliberativa", VEL_GIRO)
                dormir(random.uniform(0.5, 1.0), estado)
                detener(motor, "deliberativa")
                dir_esquiva_obstaculo = None
                estado_fsm = "BUSCAR"
                tiempo_sin_bola = time.time()
                continue
            else:
                estado_fsm = "EVADIR"
                if dir_esquiva_obstaculo == "izq":
                    girar_suave_izquierda(motor, "deliberativa", VEL_ACERCAR)
                else:
                    girar_suave_derecha(motor, "deliberativa", VEL_ACERCAR)
                print(f"\r⚠ ESQUIVANDO: sonar={distancia:.1f}cm → "
                      f"curvando {dir_esquiva_obstaculo}    ", end="")
                continue
        else:
            dir_esquiva_obstaculo = None

        # ==========================================================
        # PRIORIDAD 2: BOLA DETECTADA (cámara=qué, sonar=distancia)
        # ==========================================================
        if bola_encontrada:
            tiempo_sin_bola = time.time()
            tiempo_ultima_bola = time.time()

            sonar_cerca = 0 < distancia <= DIST_RECOGER
            bola_grande = bola_area > AREA_RECOGER

            # --- ¿Demasiado cerca? La pinza no llega si está pegada ---
            bola_enorme = bola_area > AREA_RECOGER * 3
            if bola_enorme or (sonar_cerca and distancia < DIST_RECOGER * 0.5):
                estado_fsm = "RETROCEDER"
                print(f"\r← MUY CERCA: area={bola_area:.3f} dist={distancia:.1f}cm → retrocediendo", end="")
                retroceder(motor, "deliberativa", estado, 0.2)
                continue

            # --- ¿Recoger? Ambos sensores confirman, o la bola es obvia ---
            if (sonar_cerca and bola_grande) or bola_area > AREA_RECOGER * 2.5:
                estado_fsm = "RECOGER"
                print(f"\n✓ RECOGER: dist={distancia:.1f}cm area={bola_area:.3f}")

                detener(motor, "deliberativa")
                dormir(0.3, estado)
                coger_bola(servo, estado)

                bolas_recogidas += 1
                print(f"  Bola #{bolas_recogidas} recogida. Soltando...")

                soltar_bola(servo)
                levantar_gancho(servo)

                print(f"  Esperando {PAUSA_TRAS_SOLTAR}s...")
                dormir(PAUSA_TRAS_SOLTAR, estado)

                retroceder(motor, "deliberativa", estado, 0.3)

                estado_fsm = "BUSCAR"
                tiempo_sin_bola = time.time()
                continue

            # --- Acercarse a la bola: giro proporcional al error de centrado ---
            estado_fsm = "ACERCAR"
            error = bola_cx - 0.5

            if (0 < distancia < DIST_FRENAR) or bola_grande:
                vel = VEL_FRENADO
            else:
                vel = VEL_ACERCAR

            if abs(error) < 0.15:
                avanzar(motor, "deliberativa", vel)
            else:
                factor_lenta = max(0.0, 0.65 - abs(error) * 1.5)
                vel_rapida = int(vel)
                vel_lenta = int(vel * factor_lenta)
                if error > 0:
                    motor.mover("deliberativa", -vel_rapida, -int(vel_lenta * FACTOR_CORRECCION))
                else:
                    motor.mover("deliberativa", -vel_lenta, -int(vel_rapida * FACTOR_CORRECCION))

            print(f"\r→ ACERCAR: cx={bola_cx:.2f} err={error:+.2f} "
                  f"area={bola_area:.3f} dist={distancia:.0f}cm vel={vel}    ", end="")
            continue

        # ==========================================================
        # PRIORIDAD 3: BUSCAR (no hay bola visible)
        # ==========================================================
        estado_fsm = "BUSCAR"
        elapsed = time.time() - tiempo_sin_bola

        if elapsed > TIMEOUT_BUSQUEDA:
            print(f"\n↻ BUSCAR: girando ({elapsed:.0f}s sin bola)...")
            if random.random() > 0.5:
                girar_izquierda(motor, "deliberativa", VEL_GIRO)
            else:
                girar_derecha(motor, "deliberativa", VEL_GIRO)
            dormir(random.uniform(0.5, 1.2), estado)
            detener(motor, "deliberativa")
            tiempo_sin_bola = time.time()
        else:
            avanzar(motor, "deliberativa", VEL_EXPLORAR)
            print(f"\r○ BUSCAR: {elapsed:.0f}s sin bola | dist={distancia:.0f}cm    ", end="")

    return bolas_recogidas


# =====================================================================
# MODOS DE PRUEBA (validar cada capa por separado en el laboratorio)
# =====================================================================

def modo_test_motor(estado, motor):
    """Verifica el cableado del motor y el árbitro de prioridad IR>HSV>delib."""
    print("\n=== TEST MOTOR + ÁRBITRO DE PRIORIDAD ===")
    print("Cada paso imprime si el comando se APLICÓ (True) o quedó en no-op (False).\n")

    print("1) deliberativa avanza 1s")
    print("   aplicado:", motor.mover("deliberativa", -VEL_EXPLORAR, -VEL_EXPLORAR))
    dormir(1.0, estado)
    detener(motor, "deliberativa")
    dormir(0.5, estado)
    if not estado.running:
        return

    print("2) HSV toma el control y retrocede 1s (mientras la deliberativa lo intenta)")
    motor.tomar("hsv")
    print("   HSV retrocede aplicado:", motor.mover("hsv", VEL_RETROCESO, VEL_RETROCESO))
    print("   deliberativa avanzar (debe ser False):", motor.mover("deliberativa", -VEL_EXPLORAR, -VEL_EXPLORAR))
    dormir(1.0, estado)
    motor.soltar("hsv")
    detener(motor, "hsv")
    dormir(0.5, estado)
    if not estado.running:
        return

    print("3) IR toma el control (mayor prioridad) sobre HSV")
    motor.tomar("hsv")
    motor.tomar("ir")
    print("   IR retrocede aplicado:", motor.mover("ir", VEL_RETROCESO, VEL_RETROCESO))
    print("   HSV avanzar (debe ser False):", motor.mover("hsv", -VEL_EXPLORAR, -VEL_EXPLORAR))
    print("   dueño actual:", motor.dueno())
    dormir(1.0, estado)
    motor.soltar("ir")
    motor.soltar("hsv")
    motor.parar_forzado()
    print("\nFin del test de motor.")


def modo_test_ir(estado, motor, infrared):
    """Imprime los 3 IR en bucle y retrocede si alguno salta."""
    print("\n=== TEST IR (Capa 0) ===")
    print("Pasa una línea/borde por delante de cada sensor. Ctrl+C para salir.\n")
    while estado.running:
        try:
            v1 = infrared.read_one_infrared(1)
            v2 = infrared.read_one_infrared(2)
            v3 = infrared.read_one_infrared(3)
        except Exception as e:
            print("Error leyendo IR:", e)
            break
        disparo = v1 or v2 or v3
        if disparo:
            motor.tomar("ir")
            motor.mover("ir", VEL_RETROCESO, int(VEL_RETROCESO * FACTOR_CORRECCION))
        else:
            motor.soltar("ir")
            detener(motor, "ir")
        print(f"\rIR1:{v1} IR2:{v2} IR3:{v3} | {'RETROCEDIENDO' if disparo else 'libre        '}", end="")
        dormir(0.1, estado)
    motor.parar_forzado()


def modo_test_hsv(estado, motor, camera):
    """Detección de verde HSV en bucle; retrocede si hay verde peligroso."""
    print("\n=== TEST HSV (Capa 1) ===")
    print(f"Umbral actual: {HSV_UMBRAL_PIXELES} px verdes en el tercio inferior. Ctrl+C para salir.\n")
    while estado.running:
        fb = camera.get_frame()
        if fb is None:
            continue
        frame = cv2.imdecode(np.frombuffer(fb, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        peligro, n = contar_verde(frame)
        if peligro:
            motor.tomar("hsv")
            motor.mover("hsv", VEL_RETROCESO, int(VEL_RETROCESO * FACTOR_CORRECCION))
        else:
            motor.soltar("hsv")
            detener(motor, "hsv")
        print(f"\rpíxeles verdes: {n:6d} | {'PELIGRO retrocede' if peligro else 'vía libre        '}", end="")
    motor.parar_forzado()


def modo_test_percepcion(estado, camera, detector, tcp_server):
    """YOLO + lógica de decisión SIN mover el motor. Para tunear confianzas/áreas."""
    print("\n=== TEST PERCEPCIÓN (YOLO, sin mover) ===")
    print("Mira qué detecta y qué DECIDIRÍA la máquina de estados. Ctrl+C para salir.\n")
    while estado.running:
        fb = camera.get_frame()
        if fb is None:
            continue
        frame = cv2.imdecode(np.frombuffer(fb, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        h_frame, w_frame = frame.shape[:2]

        detecciones = detector.detect(frame)
        bola_encontrada, bola_cx, bola_area = detector.get_ball_info(detecciones, w_frame)
        line_info = detector.get_line_info(detecciones, w_frame, h_frame)
        verde, n_verde = contar_verde(frame)

        if tcp_server is not None:
            anotado = dibujar_detecciones(frame.copy(), detecciones, "PERCEPCION",
                                          0.0, 0, estado, MotorSeguro(None, activo=False))
            enviar_frame(tcp_server, anotado)

        partes = [f"dets:{len(detecciones)}", f"HSV_verde:{n_verde}({'PELIGRO' if verde else 'ok'})"]
        if bola_encontrada:
            partes.append(f"BOLA cx={bola_cx:.2f} area={bola_area:.3f}")
        if line_info["detectada"]:
            partes.append(f"LINEA {'PELIGRO' if line_info['peligro'] else 'lejos'} "
                          f"y={line_info['posicion_y']:.2f}")
        print("\r" + " | ".join(partes) + "        ", end="")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Práctica 3 — Robot por capas (NCNN)")
    parser.add_argument("--bolas", type=int, default=-1,
                        help="Número de bolas a recoger (-1 = indefinido)")
    parser.add_argument("--stream", action="store_true",
                        help="Enviar vídeo con detecciones al cliente Freenove")
    parser.add_argument("--sin-motor", dest="sin_motor", action="store_true",
                        help="No mover el motor (modo seco para probar percepción/lógica)")
    parser.add_argument("--test-motor", dest="test_motor", action="store_true",
                        help="Probar cableado del motor + árbitro de prioridad")
    parser.add_argument("--test-ir", dest="test_ir", action="store_true",
                        help="Probar los 3 infrarrojos (Capa 0)")
    parser.add_argument("--test-hsv", dest="test_hsv", action="store_true",
                        help="Probar la detección de verde HSV (Capa 1)")
    parser.add_argument("--test-percepcion", dest="test_percepcion", action="store_true",
                        help="Probar YOLO + decisiones sin mover (tuneo)")
    args = parser.parse_args()

    estado = Estado()

    # Ctrl+C rápido: el handler solo baja el flag; los bucles terminan solos.
    def handler(sig, frm):
        estado.running = False
    signal.signal(signal.SIGINT, handler)

    print("=" * 52)
    print(" PRÁCTICA 3 — ARQUITECTURA POR CAPAS (NCNN)")
    if args.sin_motor:
        print(" [modo seco: el motor NO se moverá]")
    print("=" * 52)

    # --- Hardware básico (siempre) ---
    motor_hw = tankMotor()
    # En --test-motor forzamos el motor activo (si no, no se podría probar).
    motor_activo = (not args.sin_motor) or args.test_motor
    motor = MotorSeguro(motor_hw, activo=motor_activo)

    servo = None
    sonar = None
    infrared = None
    camera = None
    tcp_server = None
    detector = None
    hilos = []

    try:
        # --- Streaming opcional (import perezoso) ---
        if args.stream:
            from server import TankServer
            tcp_server = TankServer()
            tcp_server.startTcpServer()
            print("Servidor TCP iniciado. Esperando cliente Freenove...")
            while estado.running and not tcp_server.isVideoServerConnected():
                time.sleep(0.5)
            if estado.running:
                print("Cliente conectado.")

        # =====================================================
        # MODOS DE PRUEBA
        # =====================================================
        if args.test_motor:
            modo_test_motor(estado, motor)
            return

        if args.test_ir:
            infrared = Infrared()
            modo_test_ir(estado, motor, infrared)
            return

        if args.test_hsv:
            camera = Camera(stream_size=(320, 240), hflip=True, vflip=True)
            camera.start_stream()
            time.sleep(1)
            modo_test_hsv(estado, motor, camera)
            return

        if args.test_percepcion:
            from yolo_inferencia_ncnn import YOLODetectorNCNN
            detector = YOLODetectorNCNN("best_ncnn_model", conf_threshold=0.40)
            camera = Camera(stream_size=(320, 240), hflip=True, vflip=True)
            camera.start_stream()
            time.sleep(1)
            modo_test_percepcion(estado, camera, detector, tcp_server)
            return

        # =====================================================
        # MODO NORMAL: 3 capas
        # =====================================================
        from yolo_inferencia_ncnn import YOLODetectorNCNN
        servo = Servo()
        sonar = Ultrasonic()
        infrared = Infrared()
        detector = YOLODetectorNCNN("best_ncnn_model", conf_threshold=0.40)

        levantar_gancho(servo)

        camera = Camera(stream_size=(320, 240), hflip=True, vflip=True)
        camera.start_stream()
        time.sleep(1)

        if args.bolas > 0:
            print(f"Objetivo: recoger {args.bolas} bolas")
        else:
            print("Modo indefinido: Ctrl+C para salir")
        print("Arrancando capas: IR (Capa 0) + HSV (Capa 1) + deliberativa (Capa 2)\n")

        # Hilos de seguridad (daemon: no bloquean el cierre del proceso)
        t_ir = threading.Thread(target=hilo_ir, args=(estado, motor, infrared), daemon=True)
        t_hsv = threading.Thread(target=hilo_hsv, args=(estado, motor, camera), daemon=True)
        t_ir.start()
        t_hsv.start()
        hilos = [t_ir, t_hsv]

        # Esperar al primer frame del hilo HSV antes de deliberar
        while estado.running and estado.get_frame() is None:
            time.sleep(0.05)

        # Capa 2 en el hilo principal
        bolas = capa2_deliberativa(estado, motor, servo, sonar, detector, tcp_server, args.bolas)
        print(f"\nBolas recogidas: {bolas}")

    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario.")
    finally:
        estado.running = False
        motor.parar_forzado()
        for t in hilos:
            t.join(timeout=1.0)
        if servo is not None:
            try:
                soltar_bola(servo)
                levantar_gancho(servo)
            except Exception:
                pass
        if tcp_server is not None:
            try:
                tcp_server.stopTcpServer()
            except Exception:
                pass
        if camera is not None:
            try:
                camera.stop_stream()
                camera.close()
            except Exception:
                pass
        if sonar is not None:
            try:
                sonar.close()
            except Exception:
                pass
        if infrared is not None:
            try:
                infrared.close()
            except Exception:
                pass
        print("Robot detenido de forma segura.")


if __name__ == '__main__':
    main()
