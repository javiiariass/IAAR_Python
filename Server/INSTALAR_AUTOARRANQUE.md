# Autoarranque: correr el modelo 10 min al encender y apagar la Pi

Hace que, **al dar corriente a la Raspberry**, el robot ejecute `PRACTICA_3.py` durante
**10 minutos** y luego **se apague solo** (`poweroff`). Son 2 archivos:

- `run_practica3_10min.sh` — el lanzador (corre 10 min con corte limpio por SIGINT, luego apaga).
- `practica3.service` — el servicio de systemd que lanza lo anterior en cada arranque.

---

## ⚠️ Aviso importante

Una vez **activado**, **CADA vez que enciendas la Pi** el robot arrancará solo, se moverá
10 minutos y apagará la Pi. Para desarrollo normal **desactívalo** (ver más abajo) o el
robot se pondrá en marcha cada vez que arranques.

Las capas de seguridad (IR anti-caída) siguen activas, pero si algo se descontrola, **corta
la corriente al robot a mano**.

---

## Instalación (una sola vez, en la Raspberry)

Desde la carpeta `Server/`:

```bash
# 1) Permiso de ejecución al lanzador
chmod +x run_practica3_10min.sh

# 2) Averigua la ruta REAL del lanzador y ponla en el .service
realpath run_practica3_10min.sh
#   -> copia esa ruta y edita la línea ExecStart de practica3.service si no coincide

# 3) Instala el servicio
sudo cp practica3.service /etc/systemd/system/
sudo systemctl daemon-reload

# 4) Actívalo para que arranque en cada boot
sudo systemctl enable practica3.service
```

---

## Probar SIN arriesgar (recomendado antes de confiar en él)

Antes de dejarlo en automático con `poweroff` real:

1. En `run_practica3_10min.sh`, baja `DURACION=600` a `DURACION=30` y **comenta** la última
   línea (`/sbin/poweroff` → `# /sbin/poweroff`).
2. Lánzalo a mano con el robot sobre una caja:
   ```bash
   sudo ./run_practica3_10min.sh
   ```
3. Comprueba que arranca, corre los 30 s, para limpio y (con poweroff comentado) NO apaga.
4. Mira el log: `cat ultimo_arranque.log`.
5. Si todo bien, **descomenta** `poweroff` y **devuelve** `DURACION=600`.

---

## Uso normal

- **Encender el robot** → arranca solo, corre 10 min, se apaga. No hay que tocar nada.
- **Ver qué pasó en el último arranque**: `cat Server/ultimo_arranque.log`
  (código `124` = se acabó el tiempo, es lo normal; `130` = salió por SIGINT).

## Desactivar (modo desarrollo)

```bash
sudo systemctl disable practica3.service   # no arranca en el próximo boot
sudo systemctl stop practica3.service      # si estuviera corriendo ahora
```

## Ajustes rápidos (en `run_practica3_10min.sh`)

- `DURACION=600` → segundos que corre el modelo (600 = 10 min).
- `ESPERA_INICIAL=8` → segundos tras encender para colocar el robot antes de que se mueva.
- `PY=python3` → cámbialo a `python` si los paquetes (cv2/ncnn) están en ese intérprete.
