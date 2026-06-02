# Laboratorio Práctica 3 — Paso a paso (3 capas)

Guía para validar `Server/PRACTICA_3.py` (arquitectura por capas) en la sesión de lab.
Cada paso es **arrancar un comando y mirar**, sin tocar código. Si algo falla, en
cada paso tienes qué mirar y qué constante ajustar.

> Recuerda la idea: **la seguridad (IR + HSV) no depende de YOLO**. IR > HSV > deliberativa.
> Todos los comandos se ejecutan en la RPi, dentro de `Server/`, con `sudo`.

---

## 0. Preparación (5 min)

- [ ] Conectar por SSH a la RPi y entrar en la carpeta:
  ```bash
  cd ~/IAAR_Python/Server      # ajusta la ruta a tu repo
  ```
- [ ] Traer el nuevo `PRACTICA_3.py` a la RPi (elige una opción):
  ```bash
  git pull                     # si has hecho push de la rama feat/YOLO
  # o copiarlo a mano con scp desde el PC:
  # scp Server/PRACTICA_3.py pi@<IP_RPi>:~/IAAR_Python/Server/
  ```
- [ ] Comprobar que está el modelo NCNN:
  ```bash
  ls best_ncnn_model/          # debe verse model.ncnn.param y model.ncnn.bin
  ```
- [ ] Comprobar que `ncnn` está instalado:
  ```bash
  python -c "import ncnn; print('ncnn ok')"
  # si falla: pip install ncnn --break-system-packages
  ```
- [ ] Robot sobre la mesa, ruedas al aire o con sitio para moverse sin caerse.

**Salir de cualquier prueba:** `Ctrl+C` (ahora corta rápido; el robot frena y suelta la pinza).

---

## 1. Motor + árbitro de prioridad (5 min) — `--test-motor`

Valida el cableado de los motores y que **IR gana a HSV, y HSV gana a deliberativa**.

```bash
sudo python PRACTICA_3.py --test-motor
```

**Qué debe pasar (mira la consola y las ruedas):**
1. `deliberativa avanza 1s` → el robot avanza. `aplicado: True`.
2. `HSV toma el control y retrocede` → retrocede. La línea
   `deliberativa avanzar (debe ser False)` → **False** (la deliberativa NO puede pisar a HSV).
3. `IR toma el control` → retrocede. `HSV avanzar (debe ser False)` → **False**,
   `dueño actual: ir`.

- ✅ Si los "debe ser False" salen `False` → el árbitro de prioridad funciona.
- ⚠️ Si el robot avanza cuando crees que debería retroceder → motores invertidos al revés.
  Recuerda: en este proyecto **negativo = avanzar**, positivo = marcha atrás.
- ⚠️ Si una rueda va más lenta → es lo esperado (`FACTOR_CORRECCION = 1.2` para la derecha).

---

## 2. Infrarrojos — Capa 0 (5 min) — `--test-ir`

Valida los 3 IR frontales y la marcha atrás refleja.

```bash
sudo python PRACTICA_3.py --test-ir
```

Verás en bucle: `IR1:0 IR2:0 IR3:0 | libre`.

**Qué probar:**
- [ ] Pasa la mano / una línea negra-verde bajo cada sensor y mira que el dígito
      correspondiente pasa a `1`.
- [ ] Con cualquier `1`, debe poner `RETROCEDIENDO` y el robot retrocede.

- ⚠️ Si un sensor siempre da `1` o siempre `0` → revisa cableado / altura del sensor.
- 📌 La Capa 0 es la red de seguridad definitiva: **nunca se anula**. Si esto va, no te caes.

---

## 3. Línea verde HSV — Capa 1 (5 min) — `--test-hsv`

Valida la detección de verde por saturación (independiente de YOLO) y el frenado.

```bash
sudo python PRACTICA_3.py --test-hsv
```

Verás en bucle: `píxeles verdes: NNNN | vía libre`.

**Qué probar:**
- [ ] Acerca la línea verde al **tercio inferior** de la cámara → el contador sube.
- [ ] Al pasar de `3000` px (umbral) → `PELIGRO retrocede` y el robot retrocede.

**Ajuste si hace falta** (en `PRACTICA_3.py`, sección CONSTANTES):
- Frena demasiado tarde / no detecta el verde → baja `HSV_UMBRAL_PIXELES` (p. ej. 2000).
- Salta con cualquier cosa verde de lejos → súbelo (p. ej. 4000) o ajusta
  `HSV_VERDE_BAJO/ALTO` (rango de color).
- Quieres que mire una franja más alta → cambia `HSV_ROI_DESDE` (2/3 = tercio inferior).

> Si tienes `--sin-motor`, este test solo imprime sin mover (útil para calibrar el umbral
> tranquilamente): `sudo python PRACTICA_3.py --test-hsv --sin-motor`

---

## 4. Percepción YOLO — Capa 2 (10 min) — `--test-percepcion`

Valida que **YOLO ve bolas y líneas** y qué decidiría la máquina de estados, **sin moverse**.
Ideal para tunear confianzas y áreas sin sustos.

```bash
sudo python PRACTICA_3.py --test-percepcion
# con vídeo en el PC (app Freenove → IP de la RPi → Connect):
sudo python PRACTICA_3.py --test-percepcion --stream
```

Verás: `dets:N | HSV_verde:NNN(ok) | BOLA cx=0.52 area=0.031 | LINEA lejos y=0.40`

**Qué probar:**
- [ ] Pon una bola roja delante → debe aparecer `BOLA` con `cx` (0=izq, 0.5=centro, 1=der)
      y `area` (sube al acercar la bola).
- [ ] Pon la línea verde → `LINEA` y `peligro/lejos` según esté en el tercio inferior.
- [ ] Con `--stream`, mira los bounding boxes y la barra de estado (incluye flags
      `IR / HSV / SUP` y el dueño del motor).

**Ajuste de umbrales** (constructor del detector y CONSTANTES):
- No detecta bolas → baja `conf_bola` en `yolo_inferencia_ncnn.py` (o `conf_threshold=0.40` en el script).
- Detecta bolas fantasma → súbelo.
- Recoge demasiado pronto/tarde → ajusta `AREA_RECOGER` (0.06) y `DIST_RECOGER` (10 cm).

📌 Anota el `area` típico de una bola cuando está **a distancia de recoger**: ese número
es el que debe rondar `AREA_RECOGER`.

---

## 5. Run completo — 3 capas juntas (resto de la sesión)

```bash
sudo python PRACTICA_3.py                 # indefinido
sudo python PRACTICA_3.py --bolas 5       # para tras 5 bolas
sudo python PRACTICA_3.py --stream        # viendo el vídeo en el PC
```

Arranca: IR (hilo) + HSV (hilo, dueño de la cámara) + deliberativa (principal).

**Qué observar en consola** (estados de la máquina):
- `○ BUSCAR` → explorando.
- `→ ACERCAR` → bola vista, centrándose (mira `err`, debe tender a 0).
- `✓ RECOGER` → baja pinza, recoge, suelta, retrocede.
- `⚠ EVADIR` → línea/obstáculo (giro estratégico de YOLO/sonar).
- `■ SEGURIDAD (IR/HSV)` → una capa de seguridad tomó el control; la deliberativa cede.

**Secuencia de tuning recomendada si algo va raro:**

| Síntoma | Constante a tocar |
|---|---|
| Se acerca pero no recoge | `AREA_RECOGER` ↓, `DIST_RECOGER` ↑ |
| Choca con la bola (muy cerca) | ya retrocede solo; si no, `AREA_RECOGER`*3 |
| Gira poco/mucho al centrar | factor en `ACERCAR` (`0.65 - error*1.5`) |
| Evita bolas (cree que son obstáculo) | grace period `bola_reciente` (1.0 s) |
| Se acerca a la línea con bola al borde | `bola_en_borde` / `suprimir_linea` |
| Frena tarde ante el borde | `HSV_UMBRAL_PIXELES` ↓ (Capa 1, paso 3) |
| Explora muy rápido y vibra | `VEL_EXPLORAR` ↓ |

---

## Chuleta de comandos

```bash
sudo python PRACTICA_3.py --test-motor          # paso 1: motor + prioridad
sudo python PRACTICA_3.py --test-ir             # paso 2: infrarrojos
sudo python PRACTICA_3.py --test-hsv            # paso 3: verde HSV
sudo python PRACTICA_3.py --test-percepcion --stream   # paso 4: YOLO sin mover
sudo python PRACTICA_3.py --stream              # paso 5: todo junto, con vídeo
sudo python PRACTICA_3.py --bolas 5             # objetivo de 5 bolas

# flags combinables:
#   --stream        ver vídeo en el PC (app Freenove)
#   --sin-motor     no mover el motor (probar percepción/lógica en seco)
#   --bolas N       parar tras N bolas
```

## Dónde están los ajustes

Todo en `Server/PRACTICA_3.py`, bloque **CONSTANTES** (arriba del archivo):
velocidades, distancias de sonar, `AREA_RECOGER`, umbrales HSV, tiempos.
Umbrales de confianza de YOLO: constructor `YOLODetectorNCNN(...)` y `yolo_inferencia_ncnn.py`.

## Si el robot se descontrola

`Ctrl+C` → frena, suelta la pinza, sube el brazo y libera el hardware.
La Capa 0 (IR) sigue activa hasta el cierre: aunque la lógica falle, no debería caerse.
