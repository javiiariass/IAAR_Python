"""
PRACTICA_3_YOLO_STREAM.py — Comportamiento deliberativo con YOLO + streaming.

Igual que PRACTICA_3_YOLO.py pero además envía el vídeo con los
bounding boxes dibujados al cliente de Freenove para verlo en tiempo real.

Instrucciones:
  1. Ejecutar en la RPi: sudo python PRACTICA_3_YOLO_STREAM.py
  2. Abrir la app Freenove en el PC, poner la IP de la RPi y Connect
  3. Se verá el vídeo con las detecciones y el estado del robot

Uso:
  sudo python PRACTICA_3_YOLO_STREAM.py           → busca indefinidamente
  sudo python PRACTICA_3_YOLO_STREAM.py --bolas 5 → para tras recoger 5 bolas
"""
import cv2
import numpy as np
import time
import random
import argparse
import struct

from motor import tankMotor
from camera import Camera
from servo import Servo
from ultrasonic import Ultrasonic
from server import TankServer
from yolo_inferencia import YOLODetector

# =====================================================================
# CONSTANTES — Ajustar según pruebas en el robot real
# =====================================================================

# Velocidades del motor (duty cycle, rango 0-4095)
# Nota: valores NEGATIVOS = avanzar (motores invertidos)
VEL_EXPLORAR = 900       # Velocidad al buscar bola (no muy alta para que la cámara no vibre)
VEL_ACERCAR = 800        # Velocidad al dirigirse hacia una bola vista
VEL_FRENADO = 350        # Velocidad de aproximación final (sonar < 15cm)
VEL_GIRO = 1200          # Velocidad de giro sobre sí mismo

# Factor de corrección del motor derecho (el derecho gira más lento)
FACTOR_CORRECCION = 1.2

# Sonar
DIST_RECOGER = 7.0       # Distancia (cm) a la que la bola está al alcance de la pinza
DIST_FRENAR = 15.0       # Distancia (cm) a la que empezar a reducir velocidad
DIST_OBSTACULO = 15.0    # Distancia (cm) para parada de emergencia por obstáculo
DIST_OBSTACULO_LEJOS = 30.0  # Distancia (cm) para empezar a esquivar suavemente

# Bola (detección YOLO)
AREA_RECOGER = 0.05      # Área relativa de la bola para considerar "suficientemente cerca"

# Tiempos
TIMEOUT_BUSQUEDA = 6     # Segundos sin ver bola antes de girar para explorar
PAUSA_TRAS_SOLTAR = 1.5  # Segundos de espera después de soltar bola

# Servo ángulos
PINZA_ABIERTA = 90
PINZA_CERRADA = 135
BRAZO_ARRIBA = 140   # probar con 150
BRAZO_ABAJO = 90


# =====================================================================
# FUNCIONES DE MOVIMIENTO
# =====================================================================

def avanzar(motor, velocidad):
    """Avanzar recto. Velocidad positiva = avanzar (se invierte internamente)."""
    motor.setMotorModel(-velocidad, int(-velocidad * FACTOR_CORRECCION))


def girar_izquierda(motor, velocidad=VEL_GIRO):
    """Girar sobre sí mismo hacia la izquierda."""
    motor.setMotorModel(velocidad, -velocidad)


def girar_derecha(motor, velocidad=VEL_GIRO):
    """Girar sobre sí mismo hacia la derecha."""
    motor.setMotorModel(-velocidad, velocidad)


def girar_suave_izquierda(motor, velocidad=VEL_ACERCAR):
    """Avanzar girando suavemente a la izquierda (rueda izq más lenta)."""
    motor.setMotorModel(-int(velocidad * 0.3), -int(velocidad * FACTOR_CORRECCION))


def girar_suave_derecha(motor, velocidad=VEL_ACERCAR):
    """Avanzar girando suavemente a la derecha (rueda der más lenta)."""
    motor.setMotorModel(-velocidad, -int(velocidad * 0.3 * FACTOR_CORRECCION))


def detener(motor):
    motor.setMotorModel(0, 0)


def retroceder(motor, tiempo=0.4):
    """Retroceder un poco (valores positivos = marcha atrás)."""
    motor.setMotorModel(900, int(900 * FACTOR_CORRECCION))
    time.sleep(tiempo)
    detener(motor)


# =====================================================================
# FUNCIONES DE SERVO (PINZA Y BRAZO)
# =====================================================================

def levantar_gancho(servo):
    """Posición inicial: pinza abierta, brazo arriba."""
    servo.setServoAngle('0', PINZA_ABIERTA)
    servo.setServoAngle('1', BRAZO_ARRIBA)
    time.sleep(0.5)


def coger_bola(servo):
    """
    Secuencia de recogida: abrir pinza → bajar brazo despacio → cerrar → subir.
    IMPORTANTE: la pinza tapa la cámara al bajar, así que el robot
    debe estar bien posicionado ANTES de llamar a esta función.
    """
    # 1. Asegurar pinza abierta
    servo.setServoAngle('0', PINZA_ABIERTA)
    time.sleep(0.3)

    # 2. Bajar brazo despacio (de 140° a 90°, de 2 en 2)
    for angle in range(BRAZO_ARRIBA, BRAZO_ABAJO, -2):
        servo.setServoAngle('1', angle)
        time.sleep(0.02)
    time.sleep(0.3)

    # 3. Cerrar pinza
    servo.setServoAngle('0', PINZA_CERRADA)
    time.sleep(0.5)

    # 4. Subir brazo con la bola
    servo.setServoAngle('1', BRAZO_ARRIBA)
    time.sleep(0.5)


def soltar_bola(servo):
    """Abrir pinza para soltar la bola."""
    servo.setServoAngle('0', PINZA_ABIERTA)
    time.sleep(0.5)


# =====================================================================
# LÓGICA DE EVASIÓN DE LÍNEA
# =====================================================================

def evadir_linea(motor, line_info):
    """
    Ejecuta la maniobra de evasión según dónde esté la línea.
    Siempre retrocede primero y luego gira en dirección OPUESTA a la línea.

    Args:
        motor: instancia de tankMotor
        line_info: dict devuelto por YOLODetector.get_line_info()
    """
    pos_x = line_info["posicion_x"]
    esquina = line_info["esquina"]

    # Primero: frenar y retroceder
    detener(motor)
    time.sleep(0.1)
    retroceder(motor, 0.5)

    if esquina:
        # Esquina: líneas a ambos lados → girar 180° (retroceder más y girar mucho)
        retroceder(motor, 0.3)
        # Girar en la dirección donde haya más espacio (donde la línea esté más lejos)
        if pos_x > 0.5:
            girar_izquierda(motor, VEL_GIRO)
        else:
            girar_derecha(motor, VEL_GIRO)
        time.sleep(random.uniform(0.8, 1.3))
    elif pos_x > 0.6:
        # Línea a la derecha → girar a la izquierda
        girar_izquierda(motor, VEL_GIRO)
        time.sleep(random.uniform(0.4, 0.8))
    elif pos_x < 0.4:
        # Línea a la izquierda → girar a la derecha
        girar_derecha(motor, VEL_GIRO)
        time.sleep(random.uniform(0.4, 0.8))
    else:
        # Línea centrada (delante) → girar a un lado aleatorio
        if random.random() > 0.5:
            girar_izquierda(motor, VEL_GIRO)
        else:
            girar_derecha(motor, VEL_GIRO)
        time.sleep(random.uniform(0.5, 1.0))

    detener(motor)


# =====================================================================
# STREAMING AL CLIENTE FREENOVE
# =====================================================================

# Colores BGR para dibujar bounding boxes
COLOR_BOLA = (0, 0, 255)      # Rojo
COLOR_LINEA = (0, 255, 0)     # Verde
COLOR_ESTADO = (0, 255, 255)  # Amarillo
COLOR_TEXTO = (255, 255, 255) # Blanco


def dibujar_detecciones(frame, detecciones, estado, distancia, bolas_recogidas):
    """Dibuja bounding boxes, estado del robot y distancia sonar sobre el frame."""
    for class_id, nombre, confianza, x, y, w, h in detecciones:
        color = COLOR_BOLA if class_id == 0 else COLOR_LINEA
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        etiqueta = f"{nombre} {confianza:.0%}"
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(frame, etiqueta, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXTO, 1)

    # Info del estado en la parte superior
    cv2.putText(frame, f"{estado} | Sonar:{distancia:.0f}cm | Bolas:{bolas_recogidas}",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_ESTADO, 1)
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
# BUCLE PRINCIPAL
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Práctica 3 — Robot con YOLO")
    parser.add_argument("--bolas", type=int, default=-1,
                        help="Número de bolas a recoger (-1 = indefinido)")
    args = parser.parse_args()

    total_bolas = args.bolas
    bolas_recogidas = 0

    print("=" * 50)
    print(" PRÁCTICA 3: COMPORTAMIENTO DELIBERATIVO CON YOLO")
    print("=" * 50)
    if total_bolas > 0:
        print(f"Objetivo: recoger {total_bolas} bolas")
    else:
        print("Modo indefinido: Ctrl+C para salir")
    print()

    # --- Inicializar hardware ---
    motor = tankMotor()
    servo = Servo()
    sonar = Ultrasonic()
    detector = YOLODetector("best.onnx", conf_threshold=0.40)

    # --- Inicializar servidor TCP para streaming al cliente Freenove ---
    tcp_server = TankServer()
    tcp_server.startTcpServer()
    print("Servidor TCP iniciado. Esperando conexión del cliente Freenove...")
    while not tcp_server.isVideoServerConnected():
        time.sleep(0.5)
    print("Cliente conectado. Iniciando robot.")

    levantar_gancho(servo)

    cap = Camera(stream_size=(320, 240), hflip=True, vflip=True)
    cap.start_stream()
    time.sleep(1)

    # --- Variables de estado ---
    estado = "BUSCAR"
    tiempo_sin_bola = time.time()
    dir_esquiva_obstaculo = None  # "izq" o "der", se fija al detectar obstáculo

    try:
        while True:
            # ¿Hemos terminado?
            if total_bolas > 0 and bolas_recogidas >= total_bolas:
                print(f"\n¡Objetivo cumplido! {bolas_recogidas}/{total_bolas} bolas recogidas.")
                break

            # --- Capturar frame ---
            frame_bytes = cap.get_frame()
            if frame_bytes is None:
                time.sleep(0.01)
                continue

            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            h_frame, w_frame = frame.shape[:2]

            # --- Detección YOLO ---
            detecciones = detector.detect(frame)
            bola_encontrada, bola_cx, bola_area = detector.get_ball_info(detecciones, w_frame)
            line_info = detector.get_line_info(detecciones, w_frame, h_frame)

            # --- Sonar ---
            distancia = sonar.get_distance()
            if distancia < 0:
                distancia = 999.0  # Error de lectura → asumir lejos

            # --- Streaming: dibujar detecciones y enviar al cliente ---
            frame_anotado = dibujar_detecciones(
                frame.copy(), detecciones, estado, distancia, bolas_recogidas)
            enviar_frame(tcp_server, frame_anotado)

            # ==========================================================
            # PRIORIDAD 1: EVASIÓN DE LÍNEA VERDE
            # Solo si la línea está en el tercio inferior (peligro real)
            # ==========================================================
            if line_info["peligro"]:
                if estado != "EVADIR":
                    esquina_txt = " (ESQUINA)" if line_info["esquina"] else ""
                    print(f"\n⚠ EVADIR: línea en x={line_info['posicion_x']:.2f}{esquina_txt}")
                estado = "EVADIR"
                evadir_linea(motor, line_info)
                estado = "BUSCAR"
                tiempo_sin_bola = time.time()
                continue

            # ==========================================================
            # PRIORIDAD 1b: OBSTÁCULO DETECTADO POR SONAR (caja)
            # Solo si NO hay bola delante (si hay bola, el sonar la ve)
            #
            # Dos niveles:
            #  - Lejos (DIST_OBSTACULO < d < DIST_OBSTACULO_LEJOS):
            #    esquivar suavemente, avanzar curvando hacia un lado
            #  - Cerca (d < DIST_OBSTACULO):
            #    parada de emergencia, retroceder y girar fuerte
            # ==========================================================
            if not bola_encontrada and 0 < distancia < DIST_OBSTACULO_LEJOS:

                # Fijar dirección de esquiva al primer contacto
                # y mantenerla mientras siga detectando obstáculo
                if dir_esquiva_obstaculo is None:
                    dir_esquiva_obstaculo = random.choice(["izq", "der"])

                if distancia < DIST_OBSTACULO:
                    # --- CERCA: parada de emergencia ---
                    if estado != "EVADIR":
                        print(f"\n⚠ OBSTÁCULO CERCA: sonar={distancia:.1f}cm")
                    estado = "EVADIR"
                    detener(motor)
                    time.sleep(0.1)
                    retroceder(motor, 0.3)
                    if dir_esquiva_obstaculo == "izq":
                        girar_izquierda(motor, VEL_GIRO)
                    else:
                        girar_derecha(motor, VEL_GIRO)
                    time.sleep(random.uniform(0.5, 1.0))
                    detener(motor)
                    dir_esquiva_obstaculo = None  # Resetear para el próximo obstáculo
                    estado = "BUSCAR"
                    tiempo_sin_bola = time.time()
                    continue
                else:
                    # --- LEJOS: esquivar suavemente curvando ---
                    estado = "EVADIR"
                    if dir_esquiva_obstaculo == "izq":
                        girar_suave_izquierda(motor, VEL_ACERCAR)
                    else:
                        girar_suave_derecha(motor, VEL_ACERCAR)
                    print(f"\r⚠ ESQUIVANDO: sonar={distancia:.1f}cm → "
                          f"curvando {dir_esquiva_obstaculo}    ", end="")
                    continue
            else:
                # No hay obstáculo → resetear dirección de esquiva
                dir_esquiva_obstaculo = None

            # ==========================================================
            # PRIORIDAD 2: BOLA DETECTADA
            # ==========================================================
            if bola_encontrada:
                tiempo_sin_bola = time.time()

                # --- ¿Suficientemente cerca para recoger? ---
                if distancia <= DIST_RECOGER or bola_area > AREA_RECOGER:
                    estado = "RECOGER"
                    print(f"\n✓ RECOGER: dist={distancia:.1f}cm area={bola_area:.3f}")

                    detener(motor)
                    time.sleep(0.3)
                    coger_bola(servo)

                    bolas_recogidas += 1
                    print(f"  Bola #{bolas_recogidas} recogida. Soltando...")

                    # Soltar la bola donde está
                    soltar_bola(servo)

                    # Levantar brazo para despejar la cámara
                    levantar_gancho(servo)

                    # Esperar a que alguien retire la bola
                    print(f"  Esperando {PAUSA_TRAS_SOLTAR}s...")
                    time.sleep(PAUSA_TRAS_SOLTAR)

                    # Retroceder un poco para no re-detectar
                    retroceder(motor, 0.3)

                    estado = "BUSCAR"
                    tiempo_sin_bola = time.time()
                    continue

                # --- Acercarse a la bola ---
                estado = "ACERCAR"
                error = bola_cx - 0.5  # Negativo=izquierda, positivo=derecha

                # Decidir velocidad según distancia sonar
                if 0 < distancia < DIST_FRENAR:
                    vel = VEL_FRENADO
                else:
                    vel = VEL_ACERCAR

                # Dirigirse: centrar la bola en el frame
                if abs(error) < 0.10:
                    # Bola centrada → avanzar recto
                    avanzar(motor, vel)
                elif error > 0:
                    # Bola a la derecha → girar suave a la derecha
                    girar_suave_derecha(motor, vel)
                else:
                    # Bola a la izquierda → girar suave a la izquierda
                    girar_suave_izquierda(motor, vel)

                print(f"\r→ ACERCAR: cx={bola_cx:.2f} err={error:+.2f} "
                      f"dist={distancia:.0f}cm vel={vel}    ", end="")
                continue

            # ==========================================================
            # PRIORIDAD 3: BUSCAR (no hay bola visible)
            # ==========================================================
            estado = "BUSCAR"
            elapsed = time.time() - tiempo_sin_bola

            if elapsed > TIMEOUT_BUSQUEDA:
                # Mucho tiempo sin ver bola → girar para explorar
                print(f"\n↻ BUSCAR: girando ({elapsed:.0f}s sin bola)...")
                if random.random() > 0.5:
                    girar_izquierda(motor, VEL_GIRO)
                else:
                    girar_derecha(motor, VEL_GIRO)
                time.sleep(random.uniform(0.5, 1.2))
                detener(motor)
                tiempo_sin_bola = time.time()
            else:
                # Avanzar recto buscando
                avanzar(motor, VEL_EXPLORAR)
                print(f"\r○ BUSCAR: {elapsed:.0f}s sin bola | "
                      f"dist={distancia:.0f}cm    ", end="")

    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario.")
    finally:
        detener(motor)
        soltar_bola(servo)
        levantar_gancho(servo)
        tcp_server.stopTcpServer()
        cap.stop_stream()
        cap.close()
        sonar.close()
        print(f"Bolas recogidas: {bolas_recogidas}")
        print("Robot detenido de forma segura.")


if __name__ == '__main__':
    main()
