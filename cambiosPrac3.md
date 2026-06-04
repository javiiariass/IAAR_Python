# Resumen de cambios respecto al PLANNING (CLAUDE.md) — PRACTICA_3.py

## 1. Estructura y arquitectura
| Plan | Qué hicimos | Por qué |
|---|---|---|
| #9 Script único | Un solo `PRACTICA_3.py`, solo NCNN, con flag `--stream` | Solo usas NCNN → fuera ONNX. Evita mantener 4 scripts y aplicar cada cambio 4 veces |
| #2 Tres capas | Capa 0 IR (hilo), Capa 1 HSV (hilo, dueño de la cámara), Capa 2 deliberativa (hilo principal) | Separar seguridad de estrategia: no caerse NO depende de YOLO (problema #7) |
| (no estaba) | `MotorSeguro`: árbitro por prioridad IR > HSV > deliberativa | Con 3 hilos tocando el motor hay carreras; el árbitro da reacción de baja latencia a seguridad y deja en no-op a la deliberativa durante una emergencia |
| (no estaba) | HSV es el único lector de la cámara y publica el frame | `camera.get_frame()` bloquea; con un solo lector el HSV va a FPS de cámara (seguridad rápida) y YOLO muestrea el último frame sin redecodificar |

## 2. Seguridad y robustez
| Plan | Qué hicimos | Por qué |
|---|---|---|
| #8 Ctrl+C rápido | Handler SIGINT + flag `running` + `dormir()` que trocea los sleeps + hilos daemon | Los `time.sleep()` largos hacían el Ctrl+C lentísimo |
| #8 sonar bloqueante | Confirmado que en RPi 4 el sonar (gpiozero) NO hace busy-wait | El bloqueo temido era del path de la RPi 5 (lgpio), que no usas |

## 3. Aproximación y recogida (los cambios más gordos, todos del lab)
| Plan decía | Qué hicimos | Por qué |
|---|---|---|
| #3 Recogida con sonar+cámara combinados | Recoger solo por ÁREA de YOLO (fuera el sonar de la decisión) | El sonar rebota fatal en una bola pequeña → paradas imprecisas (a veces cerca, a veces lejos) |
| #3 Giro proporcional al error | Centrado por "taps" de pivote: duty alto (rompe fricción) + duración corta y proporcional al error | El giro suave/arco se quedaba pillado cerca; el pivote continuo se pasaba "a lo loco"; a poco duty se calaba (stiction) |
| (no estaba) | Aproximación final a PULSOS (P-AVANCE) | A ~5 FPS, a velocidad constante el robot se subía encima de la bola; a pasitos para y mira → para en el punto |
| (no estaba) | Retroceso fino lento y pulsado (`VEL_RETROCESO_FINO`) cuando está muy cerca | El retroceso de seguridad (900) daba un tirón brusco |
| #7 Timeout de aproximación | `TIMEOUT_ACERCAR`: si lleva >7s en ACERCAR sin recoger → retrocede y re-busca | El log mostró 18s clavado intentando centrar |

## 4. Bola junto a línea verde
| Plan decía | Qué hicimos | Por qué |
|---|---|---|
| #4 Suprimir línea por intersección de bboxes | Implementado (`bola_en_borde`) + añadido `RATIO_SUPRIMIR_LINEA` (suprime por área de la bola) | YOLO no siempre detecta la caja de línea, pero el HSV sí ve verde → la bola al borde no se podía coger. El umbral por área lo resuelve |
| (no estaba) | Escape de línea: al salir de seguridad por línea, retrocede Y gira | El HSV solo retrocede recto → bucle avanza/retrocede infinito; girar rompe el bucle |

## 5. Añadidos para el laboratorio (no estaban en el plan)
- Modos de prueba por flags: `--test-motor`, `--test-ir`, `--test-hsv`, `--test-percepcion`, `--sin-motor`. → Con 3h de lab, validar cada capa por separado sin tocar código.
- `--log` + exponer `hsv_verde` (verde en consola y overlay). → Calibrar umbrales (`AREA_RECOGER`, `HSV_UMBRAL_PIXELES`) con datos reales.
- `LAB_PASO_A_PASO.md`: guía de calibración paso a paso.

## 6. Pendiente del plan (no implementado aún)
- #5 Verificación post-recogida (comparar área/posición antes y después de bajar la pinza para saber si cogió la bola).
- #6 Memoria de bola tras evasión (girar hacia donde se vio por última vez en vez de re-buscar a ciegas). El "escape de línea" actual puede abandonar una bola buena cerca de un borde.
- #4 la intersección de bboxes está, pero en la práctica se manda con el umbral por área (`RATIO_SUPRIMIR_LINEA`).

## Idea de fondo que cambió respecto al plan
El plan confiaba mucho en el SONAR para la distancia y en combinar sensores. En el lab vimos que el sonar es poco fiable para bolas, así que la distancia a la bola la lleva el ÁREA de YOLO, y el sonar quedó solo para el obstáculo (caja). Y casi toda la dificultad real resultó ser control de motores con fricción/stiction a ~5 FPS, que motivó los "taps" y la aproximación pulsada — algo que el plan no anticipaba.

---

## Ajustes post-lab (sesión de calibración) — centrado, recogida y test de motor

> A partir de aquí, el registro de cambios se irá añadiendo en este archivo (lo más
> reciente abajo). `CLAUDE.md` mantiene el estado/planning; aquí va el histórico de qué se
> tocó y por qué.

Cambios sobre `PRACTICA_3.py` tras revisar los fallos del último lab:

| # | Cambio | Antes → Después | Motivo |
|---|---|---|---|
| 1 | `BOLA_CENTRADA` | 0.12 → **0.15** | De cerca el `cx` tiembla mucho; con el umbral antiguo se quedaba intentando centrar sin parar. Más margen → recoge / avanza recto antes. |
| 2 | `PULSO_GIRO` (tap de giro mínimo) | 0.10 → **0.30 s** | Los taps cortos no rompían la fricción del motor ("zumba pero no gira"). Un mínimo más largo da tiempo a arrancar desde parado. |
| 3 | `PULSO_GIRO_MAX` (tap de giro máximo) | 0.30 → **0.50 s** | Acompaña al mínimo más largo, para que el giro proporcional siga teniendo recorrido con error grande. |
| 4 | Umbral RETROCEDER por bola muy cerca | `AREA_RECOGER*1.1` → **`*1.2`** | No echarse atrás tan pronto; tolerar que la bola esté algo más cerca de lo ideal antes de retroceder (preferimos intentar coger). |
| 5 | Recogida sin centrado perfecto | bypass `AREA_RECOGER*1.5` → **`*1.2`** | Ensanchar la ventana de recogida: coger aunque no esté perfectamente centrada en cuanto es suficientemente grande (de cerca el `cx` es ruidoso). |
| 6 | `modo_test_motor`: pasos nuevos | +**"4) ROTAR derecha"** y **"5) ROTAR izquierda"** (pivote); árbitro renumerado a 6) y 7) | Poder medir el arranque del giro en sitio (el que más sufre la fricción), no solo el avance recto. |

**Fallos que motivaron estos cambios y su origen probable:**
- *Le cuesta centrar / se queda pillado antes de avanzar o girar* (zumba como aplicando
  fuerza pero no se mueve, al rato arranca) → origen probable: **fricción estática (stiction)**
  porque el duty de arranque desde parado es bajo, sumado a posible **PWM por software
  inestable bajo carga** (gpiozero + NCNN a 4 hilos ocupando los 4 cores). Mitigado en parte
  con #2/#3; **pendiente** medir el duty de arranque real y decidir subir duty / bajar hilos / pigpio.
- *Se aleja de la bola cuando podría cogerla* → la ventana de recogida era demasiado estrecha
  (retrocedía antes de poder coger). Mitigado con #4 y #5. Nota: el área NO es monótona —
  cerca BAJA (cámara alta, la bola se sale por abajo), por eso el punto de recogida es área≈0.150.
- *Bola en esquina con dos bboxes de línea que no solapan la bola* → aún abierto; la idea es
  el **umbral de verde dinámico** (ver pendientes en `CLAUDE.md`) en vez de la supresión por solape.

**En calibración** (valores a medir en el robot real; guía en `Server/LAB_TAREAS_COMPANERO.md`):
- Variación del `cx` de la bola (validar `BOLA_CENTRADA = 0.15`).
- Umbral de verde CON bola cerca (~10-12k px; el normal de 3000 se queda).
- Duty de arranque del motor (avance y giro en sitio).
