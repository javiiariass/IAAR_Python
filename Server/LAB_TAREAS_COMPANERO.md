# Tareas de laboratorio (mientras Javi no está)

---

## Cambios realizados después de la última sesión de laboratorio

En la inferencia `yolo_inferencia_ncnn.py` le he quitado 1 hilo (de 4 a 3) para que ese hilo lo use el script principal y no se sature tanto. También he subido la confianza necesaria de la bola (de 0.45 a 0.5) -> se podría incluso subir un poco más si sigue detectando bolas donde no hay

Cambios hechos en `PRACTICA_3.py` desde la última sesión (qué y por qué):

| # | Cambio | Antes → Después | Motivo |
|---|---|---|---|
| 1 | `BOLA_CENTRADA` | 0.12 → **0.15** | La bola se da por "centrada" con más margen. De cerca el `cx` tiembla mucho; con el umbral antiguo se quedaba intentando centrar sin parar. Con más margen recoge / avanza recto antes. |
| 2 | `PULSO_GIRO` (tap de giro mínimo) | 0.10 → **0.30 s** | Los taps de centrado cortos no rompían la fricción del motor ("zumba pero no gira"). Un mínimo más largo le da tiempo a arrancar desde parado. |
| 3 | `PULSO_GIRO_MAX` (tap de giro máximo) | 0.30 → **0.50 s** | Acompaña al mínimo más largo, para que el giro proporcional siga teniendo recorrido cuando el error de centrado es grande. |
| 4 | Umbral RETROCEDER por bola muy cerca | `AREA_RECOGER * 1.1` → **`* 1.2`** | No echarse atrás tan pronto. Se tolera que la bola esté un poco más cerca de lo ideal antes de retroceder (preferimos intentar coger). |
| 5 | Recogida sin centrado perfecto | bypass `AREA_RECOGER * 1.5` → **`* 1.2`** | Ensanchar la ventana de recogida: coger la bola aunque no esté perfectamente centrada en cuanto es suficientemente grande, porque de cerca el `cx` es ruidoso. |
| 6 | `modo_test_motor`: pasos nuevos | +**"4) ROTAR derecha"** y **"5) ROTAR izquierda"** (giro en sitio); árbitro renumerado a 6) y 7) | Poder medir el arranque del **giro en sitio** (pivote), que es el que más sufre la fricción, no solo el avance recto. |

> Nota: 1-5 buscan que **centre y recoja mejor** (menos quedarse pillado, menos alejarse de
> la bola). El cambio 6 es solo para medir mejor en el test.

---

## Objetivo de esta sesión

Aún quedan 3 cosas por medir/observar en el robot real. NO hace falta tocar la lógica;
solo correr los modos de prueba que ya existen y **apuntar valores** en la tabla del final.

Todo se ejecuta desde la carpeta `Server/` en la Raspberry:
```
cd ~/.../Server
sudo python PRACTICA_3.py <opciones>
```
`Ctrl+C` para salir de cualquier modo. Si tarda en cerrar, espera unos segundos.

> El flag `--stream` es opcional: muestra el vídeo con las cajas en el cliente Freenove
> (hay que tener el cliente abierto y conectado). Si no, **los valores también salen por
> consola**, que es lo único que necesitamos.

Orden recomendado: primero las dos de **percepción** (el robot no se mueve → seguras de
hacer solo); la del **motor** la última (sí se mueve: robot sobre una caja con las orugas
al aire, o sujétalo).

---

## TAREA 1 — Variación del `cx` de la bola

El área de recogida **ya está calibrada** (`AREA_RECOGER = 0.150`, ~distancia de pinza).
Lo que falta medir es **cuánto tiembla el `cx`** de la bola frame a frame, sobre todo de
cerca. Eso nos dice si `BOLA_CENTRADA = 0.15` es el margen correcto o se queda corto/largo.

**Comando** (no mueve el motor):
```
sudo python PRACTICA_3.py --test-percepcion
```
(o `--test-percepcion --stream` para ver las cajas). Verás líneas tipo:
```
dets:1 | HSV_verde:120(ok) | BOLA cx=0.50 area=0.148
```

**Qué hacer y anotar:**

1. Pon una bola **quieta y centrada** delante del robot, **a distancia de recogida**
   (área ≈ 0.15). Sin tocar nada, observa el `cx` unos segundos y apunta **entre qué dos
   valores baila** (p.ej. 0.47–0.53). → la mitad de ese rango debería caber en
   `BOLA_CENTRADA`.

2. Repite con la bola **un poco más lejos** (área ≈ 0.08–0.10) y apunta el baile del `cx`
   ahí. → para comparar si tiembla más de cerca que de lejos.

3. Si el `cx` se sale de ±0.15 estando la bola visualmente centrada, anótalo: significa que
   `BOLA_CENTRADA` se queda corto y habría que subirlo más.

---

## TAREA 2 — Umbral de verde CON BOLA cerca

El umbral **normal** (3000 px) **ya está bien**, no hace falta tocarlo. Lo que falta es el
**umbral máximo cuando hay una bola pegada a la línea**: cuánto verde tolerar para poder
acercarse a esa bola sin caerse. Lo estimábamos en **10-12k px**.

**Comando** (seguro, no se mueve gracias a `--sin-motor`):
```
sudo python PRACTICA_3.py --test-hsv --sin-motor
```
(añade `--stream` para ver el ROI y el contador en vídeo). Verás:
```
píxeles verdes:  11300 | PELIGRO retrocede
```

**Qué hacer y anotar** (mueve la línea / el robot **a mano**, sin que ande):

1. Acerca la línea verde hasta **lo más cerca que nos atreveríamos con una bola delante**,
   justo antes de que la raya empiece a salirse por abajo de la imagen. Apunta el conteo.
   → **candidato a umbral CON-BOLA** (esperábamos 10-12k).

2. **Importante (el verde NO es monótono):** sigue acercando la línea y observa el conteo.
   Anota:
   - el conteo **máximo (pico)** que llega a marcar,
   - confirma que al acercarla aún más el conteo **CAE hacia 0**, porque la raya se sale de
     la cámara (cámara alta). Apunta a qué distancia empieza a caer.

   Esto es clave: el umbral CON-BOLA tiene que estar **por debajo de ese pico**, o no
   saltará nunca y el robot se caería. (El sensor IR es el último seguro físico.)

3. (Opcional) Confirma de paso que el **normal** (3000) salta a la distancia cómoda de
   siempre, para verificar que sigue bien.

---

## TAREA 3 — Duty de arranque del motor (avance Y giro en sitio)

**Qué buscamos:** el duty más bajo al que las orugas **arrancan limpio desde parado**, sin
quedarse "zumbando sin moverse". **Ahora medimos los dos**: avance recto Y giro en sitio
(con los pasos de rotación nuevos).

**Seguridad:** **CON ESTE TEST SI SE MUEVE EL ROBOT**

Ejecútalo sin `--stream` si quieres porque vas a tener que estar ejecutando test, cambiando valor y volviendo a ejecutar. Si tienes que estar conectando desde el cliente va a ser un coñazo

**Comando:**
```
sudo python PRACTICA_3.py --test-motor
```
Pasos que hace (cada uno mueve **1.2 s desde parado**, luego 0.4 s quieto):
```
1) SOLO oruga izquierda    2) SOLO oruga derecha    3) AMBAS avanzan
4) ROTAR derecha (pivote)  5) ROTAR izquierda (pivote)
6) y 7) pruebas del árbitro de prioridad (HSV/IR)
```

**Qué hacer y anotar:**

1. Con el valor de fábrica (`VEL = 1500`): ¿arrancan **limpio e inmediato** en avance (paso
   3) y en giro en sitio (pasos 4-5)? ¿alguna oruga **zumba y arranca tarde** o no arranca?
   Apunta cuál y en qué paso.

2. Para encontrar el umbral, **edita UNA sola línea** en `PRACTICA_3.py`, dentro de
   `def modo_test_motor`:
   ```python
   VEL = 1500   # <-- cambia este número
   ```
   Probar con **800, 900, 1000, 1100, 1200, 1300, 1400** (relanza el comando cada vez). Para cada
   valor apunta por separado:
   - **AVANCE** (paso 3): ¿arranca inmediato / tarde-zumbando / no se mueve?
   - **GIRO en sitio** (pasos 4-5): igual. *(El giro necesita más fuerza: las orugas raspan
     de lado. Es normal que su umbral sea más alto que el de avance.)*

   Busca el **mínimo que arranca siempre y al instante** para cada caso.

   

---

## Tabla para rellenar

| Medida | Valor(es) | Notas |
|---|---|---|
| `cx` baila (bola cerca, área≈0.15) | entre ___ y ___ | ¿cabe en ±0.15? |
| `cx` baila (bola lejos, área≈0.08) | entre ___ y ___ | |
| **Umbral verde CON-BOLA** (px) | | esperado 10-12k |
| Pico máximo de verde y dónde empieza a caer | | |
| Motor 1500: ¿avance limpio? ¿giro limpio? | | pasos 3 y 4-5 |
| **Duty mínimo arranque — AVANCE** | | probar 1000-1400 |
| **Duty mínimo arranque — GIRO en sitio** | | suele ser mayor |
| Orugas/observaciones raras | | |

---

## Si sobra tiempo
- Repetir TAREA 1 con la bola en **esquina/lateral** y apuntar si YOLO detecta líneas verdes
  a la vez y dónde (cx de la línea). Útil para el tema de "bola en esquina".
- Grabar un par de vídeos cortos con `--stream` de la bola acercándose, para revisarlos juntos.

**No hace falta cambiar la lógica del robot** (salvo la línea `VEL` del test, que se
devuelve a 1500). Solo medir y apuntar.
