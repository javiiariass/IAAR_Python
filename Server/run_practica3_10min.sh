#!/usr/bin/env bash
#
# run_practica3_10min.sh
# -----------------------
# Lanza PRACTICA_3.py al encender el robot, lo deja correr 10 minutos y APAGA la Raspberry.
#
#   - El corte a los 10 min se hace con SIGINT (timeout --signal=INT), que dispara el
#     handler de Ctrl+C de PRACTICA_3.py -> parada LIMPIA de motores, servos y cámara.
#   - Si no cerrara en 15 s, se fuerza con SIGKILL (--kill-after=15).
#   - Pase lo que pase (fin por tiempo, fin normal o crash) se ejecuta 'poweroff'.
#   - La salida se guarda en un log para poder revisarla en el siguiente arranque.
#
# Para DESACTIVAR el autoarranque (modo desarrollo):
#     sudo systemctl disable practica3.service
#
set -u

# Carpeta de este script (= Server/). Así funciona esté donde esté el repo en la Pi.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1

DURACION=600          # segundos que corre el MODELO (600 = 10 min)
ESPERA_INICIAL=8      # s tras encender para colocar el robot (sube/baja a gusto)
PY=python3            # intérprete con cv2/ncnn instalados (cambia a 'python' si hiciera falta)
LOG="$DIR/ultimo_arranque.log"


trap 'echo "Interrupción detectada (Ctrl+C). Abortando el script..."; exit 130' INT TERM

{
  echo "===== Arranque $(date) ====="
  echo "Esperando ${ESPERA_INICIAL}s para colocar el robot..."
  sleep "$ESPERA_INICIAL"

  echo "Lanzando PRACTICA_3.py durante ${DURACION}s..."
  # 124 = se acabó el tiempo (es lo NORMAL aquí). 130 = salió por SIGINT.
  timeout --kill-after=15 --signal=INT "$DURACION" "$PY" PRACTICA_3.py
  echo "PRACTICA_3.py terminó (código $?). Apagando la Raspberry..."
} >> "$LOG" 2>&1

# Apagar la Pi pase lo que pase.
/sbin/poweroff
