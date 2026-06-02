"""
PRACTICA_3_YOLO_NCNN.py — Comportamiento deliberativo con YOLO (motor NCNN, sin streaming).

Versión que usa NCNN en vez de ONNX para la inferencia.
NCNN está optimizado para ARM (Raspberry Pi) y suele dar más FPS.

El comportamiento es idéntico a PRACTICA_3_YOLO.py, solo cambia
el motor de inferencia.

Requisitos extra en la RPi:
  pip install ncnn --break-system-packages

Archivos del modelo necesarios en Server/:
  best_ncnn_model/model.ncnn.param
  best_ncnn_model/model.ncnn.bin

Uso:
  sudo python PRACTICA_3_YOLO_NCNN.py           → busca indefinidamente
  sudo python PRACTICA_3_YOLO_NCNN.py --bolas 5 → para tras recoger 5 bolas
"""
import cv2
import numpy as np
import time
import random
import argparse

from motor import tankMotor
from camera import Camera
from servo import Servo
from ultrasonic import Ultrasonic
from yolo_inferencia_ncnn import YOLODetectorNCNN

# =====================================================================
# CONSTANTES — Ajustar según pruebas en el robot real
# =====================================================================

VEL_EXPLORAR = 900
VEL_ACERCAR = 800
VEL_FRENADO = 350
VEL_GIRO = 1200

FACTOR_CORRECCION = 1.2

DIST_RECOGER = 7.0
DIST_FRENAR = 15.0
DIST_OBSTACULO = 15.0
DIST_OBSTACULO_LEJOS = 30.0

AREA_RECOGER = 0.05

TIMEOUT_BUSQUEDA = 6
PAUSA_TRAS_SOLTAR = 1.5

PINZA_ABIERTA = 90
PINZA_CERRADA = 135
BRAZO_ARRIBA = 140
BRAZO_ABAJO = 90


# =====================================================================
# FUNCIONES DE MOVIMIENTO
# =====================================================================

def avanzar(motor, velocidad):
    motor.setMotorModel(-velocidad, int(-velocidad * FACTOR_CORRECCION))


def girar_izquierda(motor, velocidad=VEL_GIRO):
    motor.setMotorModel(velocidad, -velocidad)


def girar_derecha(motor, velocidad=VEL_GIRO):
    motor.setMotorModel(-velocidad, velocidad)


def girar_suave_izquierda(motor, velocidad=VEL_ACERCAR):
    motor.setMotorModel(-int(velocidad * 0.55), -int(velocidad * FACTOR_CORRECCION))


def girar_suave_derecha(motor, velocidad=VEL_ACERCAR):
    motor.setMotorModel(-velocidad, -int(velocidad * 0.55 * FACTOR_CORRECCION))


def detener(motor):
    motor.setMotorModel(0, 0)


def retroceder(motor, tiempo=0.4):
    motor.setMotorModel(900, int(900 * FACTOR_CORRECCION))
    time.sleep(tiempo)
    detener(motor)


# =====================================================================
# FUNCIONES DE SERVO (PINZA Y BRAZO)
# =====================================================================

def levantar_gancho(servo):
    servo.setServoAngle('0', PINZA_ABIERTA)
    servo.setServoAngle('1', BRAZO_ARRIBA)
    time.sleep(0.5)


def coger_bola(servo):
    servo.setServoAngle('0', PINZA_ABIERTA)
    time.sleep(0.3)

    for angle in range(BRAZO_ARRIBA, BRAZO_ABAJO, -2):
        servo.setServoAngle('1', angle)
        time.sleep(0.02)
    time.sleep(0.3)

    servo.setServoAngle('0', PINZA_CERRADA)
    time.sleep(0.5)

    servo.setServoAngle('1', BRAZO_ARRIBA)
    time.sleep(0.5)


def soltar_bola(servo):
    servo.setServoAngle('0', PINZA_ABIERTA)
    time.sleep(0.5)


# =====================================================================
# LÓGICA DE EVASIÓN DE LÍNEA
# =====================================================================

def evadir_linea(motor, line_info):
    pos_x = line_info["posicion_x"]
    esquina = line_info["esquina"]

    detener(motor)
    time.sleep(0.1)
    retroceder(motor, 0.5)

    if esquina:
        retroceder(motor, 0.3)
        if pos_x > 0.5:
            girar_izquierda(motor, VEL_GIRO)
        else:
            girar_derecha(motor, VEL_GIRO)
        time.sleep(random.uniform(0.8, 1.3))
    elif pos_x > 0.6:
        girar_izquierda(motor, VEL_GIRO)
        time.sleep(random.uniform(0.4, 0.8))
    elif pos_x < 0.4:
        girar_derecha(motor, VEL_GIRO)
        time.sleep(random.uniform(0.4, 0.8))
    else:
        if random.random() > 0.5:
            girar_izquierda(motor, VEL_GIRO)
        else:
            girar_derecha(motor, VEL_GIRO)
        time.sleep(random.uniform(0.5, 1.0))

    detener(motor)


# =====================================================================
# BUCLE PRINCIPAL
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Práctica 3 — Robot con YOLO (NCNN)")
    parser.add_argument("--bolas", type=int, default=-1,
                        help="Número de bolas a recoger (-1 = indefinido)")
    args = parser.parse_args()

    total_bolas = args.bolas
    bolas_recogidas = 0

    print("=" * 50)
    print(" PRÁCTICA 3: COMPORTAMIENTO DELIBERATIVO (NCNN)")
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
    detector = YOLODetectorNCNN("best_ncnn_model", conf_threshold=0.40)

    levantar_gancho(servo)

    cap = Camera(stream_size=(320, 240), hflip=True, vflip=True)
    cap.start_stream()
    time.sleep(1)

    # --- Variables de estado ---
    estado = "BUSCAR"
    tiempo_sin_bola = time.time()
    tiempo_ultima_bola = 0
    dir_esquiva_obstaculo = None

    try:
        while True:
            if total_bolas > 0 and bolas_recogidas >= total_bolas:
                print(f"\n¡Objetivo cumplido! {bolas_recogidas}/{total_bolas} bolas recogidas.")
                break

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
                distancia = 999.0

            # ==========================================================
            # PRIORIDAD 1: EVASIÓN DE LÍNEA VERDE
            # ==========================================================
            if line_info["peligro"]:
                linea_inminente = line_info["posicion_y"] > 0.80
                bola_cerca = bola_encontrada and bola_area > AREA_RECOGER * 0.5
                if bola_cerca:
                    evadir_ahora = linea_inminente
                else:
                    evadir_ahora = not bola_encontrada or linea_inminente

                if evadir_ahora:
                    if estado != "EVADIR":
                        esquina_txt = " (ESQUINA)" if line_info["esquina"] else ""
                        print(f"\n⚠ EVADIR: línea en x={line_info['posicion_x']:.2f} "
                              f"y={line_info['posicion_y']:.2f}{esquina_txt}")
                    estado = "EVADIR"
                    evadir_linea(motor, line_info)
                    estado = "BUSCAR"
                    tiempo_sin_bola = time.time()
                    continue

            # ==========================================================
            # PRIORIDAD 1b: OBSTÁCULO DETECTADO POR SONAR
            # ==========================================================
            bola_reciente = (time.time() - tiempo_ultima_bola) < 1.0
            if not bola_encontrada and not bola_reciente and 0 < distancia < DIST_OBSTACULO_LEJOS:

                if dir_esquiva_obstaculo is None:
                    dir_esquiva_obstaculo = random.choice(["izq", "der"])

                if distancia < DIST_OBSTACULO:
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
                    dir_esquiva_obstaculo = None
                    estado = "BUSCAR"
                    tiempo_sin_bola = time.time()
                    continue
                else:
                    estado = "EVADIR"
                    if dir_esquiva_obstaculo == "izq":
                        girar_suave_izquierda(motor, VEL_ACERCAR)
                    else:
                        girar_suave_derecha(motor, VEL_ACERCAR)
                    print(f"\r⚠ ESQUIVANDO: sonar={distancia:.1f}cm → "
                          f"curvando {dir_esquiva_obstaculo}    ", end="")
                    continue
            else:
                dir_esquiva_obstaculo = None

            # ==========================================================
            # PRIORIDAD 2: BOLA DETECTADA
            # ==========================================================
            if bola_encontrada:
                tiempo_sin_bola = time.time()
                tiempo_ultima_bola = time.time()

                sonar_cerca = 0 < distancia <= DIST_RECOGER
                bola_grande = bola_area > AREA_RECOGER

                # --- ¿Demasiado cerca? ---
                bola_enorme = bola_area > AREA_RECOGER * 3
                if bola_enorme or (sonar_cerca and distancia < DIST_RECOGER * 0.5):
                    estado = "RETROCEDER"
                    print(f"\r← MUY CERCA: area={bola_area:.3f} dist={distancia:.1f}cm → retrocediendo", end="")
                    retroceder(motor, 0.2)
                    continue

                # --- ¿Recoger? ---
                if (sonar_cerca and bola_grande) or bola_area > AREA_RECOGER * 2.5:
                    estado = "RECOGER"
                    print(f"\n✓ RECOGER: dist={distancia:.1f}cm area={bola_area:.3f}")

                    detener(motor)
                    time.sleep(0.3)
                    coger_bola(servo)

                    bolas_recogidas += 1
                    print(f"  Bola #{bolas_recogidas} recogida. Soltando...")

                    soltar_bola(servo)
                    levantar_gancho(servo)

                    print(f"  Esperando {PAUSA_TRAS_SOLTAR}s...")
                    time.sleep(PAUSA_TRAS_SOLTAR)

                    retroceder(motor, 0.3)

                    estado = "BUSCAR"
                    tiempo_sin_bola = time.time()
                    continue

                # --- Acercarse ---
                estado = "ACERCAR"
                error = bola_cx - 0.5

                if (0 < distancia < DIST_FRENAR) or bola_grande:
                    vel = VEL_FRENADO
                else:
                    vel = VEL_ACERCAR

                if abs(error) < 0.15:
                    avanzar(motor, vel)
                else:
                    factor_lenta = max(0.0, 0.65 - abs(error) * 1.5)
                    vel_rapida = int(vel)
                    vel_lenta = int(vel * factor_lenta)
                    if error > 0:
                        motor.setMotorModel(-vel_rapida, -int(vel_lenta * FACTOR_CORRECCION))
                    else:
                        motor.setMotorModel(-vel_lenta, -int(vel_rapida * FACTOR_CORRECCION))

                print(f"\r→ ACERCAR: cx={bola_cx:.2f} err={error:+.2f} "
                      f"area={bola_area:.3f} dist={distancia:.0f}cm vel={vel}    ", end="")
                continue

            # ==========================================================
            # PRIORIDAD 3: BUSCAR
            # ==========================================================
            estado = "BUSCAR"
            elapsed = time.time() - tiempo_sin_bola

            if elapsed > TIMEOUT_BUSQUEDA:
                print(f"\n↻ BUSCAR: girando ({elapsed:.0f}s sin bola)...")
                if random.random() > 0.5:
                    girar_izquierda(motor, VEL_GIRO)
                else:
                    girar_derecha(motor, VEL_GIRO)
                time.sleep(random.uniform(0.5, 1.2))
                detener(motor)
                tiempo_sin_bola = time.time()
            else:
                avanzar(motor, VEL_EXPLORAR)
                print(f"\r○ BUSCAR: {elapsed:.0f}s sin bola | "
                      f"dist={distancia:.0f}cm    ", end="")

    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario.")
    finally:
        detener(motor)
        soltar_bola(servo)
        levantar_gancho(servo)
        cap.stop_stream()
        cap.close()
        sonar.close()
        print(f"Bolas recogidas: {bolas_recogidas}")
        print("Robot detenido de forma segura.")


if __name__ == '__main__':
    main()
