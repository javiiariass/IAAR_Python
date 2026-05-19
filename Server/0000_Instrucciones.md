# Info para final
- distancia sonar para pelota [6-7cm]


### Resumen de Cambios

1.  **Algoritmos de detección unificados:** Los tres scripts usan ahora el mismo umbral de área para la línea verde (`> 700`) y las mismas tolerancias de proporción (`aspect_ratio` entre `0.2` y `5.0`) para detectar bolas difuminadas.
2.  **Soporte Multi-Objeto:** Todos extraen y deconstruyen tantas cajas (pelotas o líneas) como existan simultáneamente en pantalla y las guardan bajo el mismo archivo `.txt`.
3.  **Captura de fondos (Negative Samples) implementada:** Pulsar la tecla manual en cualquiera de los scripts fuerza el guardado de la foto con un `.txt` vacío si no detecta objetos, esencial para reducir falsos positivos en YOLO.
4.  **Diferenciación de Prefijos:** Se ha ajustado el patrón de guardado por defecto para saber el origen de los ficheros (`movimiento_auto`, `movimiento_manual` y `estatico_manual`).

---

### Descripción de Archivos y Outputs

*   **Capturar_Yolo.py (Modo Autónomo):** El robot conduce solo evadiendo la línea reactiva y captura fotos automáticamente (cada 2.5s) si hay objetos. Permite fotos manuales.
    *   **Imprime:** `movimiento_auto_000x` y `movimiento_manual_000x`.

*   **Capturar_Yolo_Movimiento_Manual.py (Modo Teclas):** Tú conduces el robot manteniendo 'W' y 'S' desde la ventana de la cámara, pulsando 'd' para disparar capturas mientras estás en movimiento.
    *   **Imprime:** `movimiento_manual_000x`.

*   **Capturar_Yolo_Servidor.py (Modo CLI/Estático):** Script quieto que envía la señal a la app de VNC/PC. Genera fotos al pulsar la tecla 'd' y Enter desde la consola SSH.
    *   **Imprime:** `estatico_manual_000x`.

*   **Grabar_Video_Autonomo.py (Modo Grabación Autónomo):** Mantiene el comportamiento reactivo y la conducción autónoma, pero en lugar de capturar imágenes para el dataset, graba frame a frame todo el recorrido limpio (sin cajas dibujadas encima) en vídeo. Es ***importantísimo*** cerrar el proceso dándole a la tecla `q` para que el guardado se cierre correctamente sin corromperse.
    *   **Imprime:** Archivos de vídeo dentro de la carpeta `videos_prueba/test_yolo_FECHA.avi`.


### QUÉ HACER
- **importante**. Las fotos "vacías" (sin pelota y sin linea) solo se toman en manual (pulsando la 'd'). Deberiamos tener entre un 10 y un 15% de fotos de este tipo para evitar falsos positivos

### Consideraciones clave para tomar el dataset en una sola sesión:

1. **Variabilidad de iluminación:** Haz fotos en diferentes partes de la pista (cerca de ventanas, bajo luces directas, en zonas de sombra). YOLO es muy sensible a cambios de luz si no los ha visto antes.
2. **Diferentes ángulos y distancias:** Asegúrate de tener capturas de las pelotas y las líneas desde muy lejos, a media distancia y muy cerca. También fotos desde distintos ángulos de acercamiento.
3. **Imágenes de fondo ricas (Negative Samples):** Toma fotos (con la captura manual) del suelo vacío, paredes, esquinas, zócalos, o tus propios zapatos si suelen salir en el frame, *asegurándote de que no hay pelotas ni línea verde*. 
4. **Cuidado con las oclusiones:** Capta intencionadamente pelotas parcialmente tapadas por la pared o cortadas por el borde de la imagen, para que el modelo aprenda a detectar objetos a medias.

### Escenario para el video
Created [](file:///c%3A/Users/javie/Documents/Github/IAAR_Python/Server/Grabar_Video_Autonomo.py)

Acabo de crear el archivo Grabar_Video_Autonomo.py. 

El script mantiene todo el comportamiento reactivo (conducción autónoma, corrección de deriva y evasiones de pared/línea) y, a su vez, inicializa un `cv2.VideoWriter`. Frame a frame, sin dibujar ninguna caja encima (para que el modelo luego tenga un vídeo real y puro sobre el que trabajar), graba todo el recorrido.
Quedan guardados dentro de la carpeta `videos_prueba/test_yolo_FECHA.avi`, para que nunca sobreescribas un vídeo anterior y tengas varios intentos guardados limpios. Ten en cuenta que **es crítico salir pulsando la tecla 'q'** para que el vídeo se cierre y procese correctamente (si cierras de golpe el terminal, OpenCV podría corromper el fichero .avi).

---

### Cómo montar el escenario para probar el modelo con vídeo:

1.  **Bola solitaria:** Pon una bola lejos del resto, preferiblemente en el centro del pasillo o en un extremo libre. Esto te dará tomas de la bola acercándose desde lejos a los ojos del robot hasta tenerla delante, confirmando que YOLO logra traquearla ininterrumpidamente al aumentar su tamaño.
2.  **Oclusiones con la caja:** 
    *   Sitúa la caja ladeada respecto al avance predecible del robot.
    *   Mete una bola *medio tapada* asomando por detrás de una esquina de la caja. El robot tiene que poder detectar que ese arco rojo es "una bola", comprobando si la red aprendió a tolerar oclusiones (gracias al reajuste a `0.2` en el formato del dataset que le hicimos, lo aprenderá seguro).
3.  **Clustering / Cargas juntas:** Coloca la 3ª bola cerquita de la pelota ocluida de la caja. Si el robot consigue encuadrar dos bounding boxes separadas (una para la pelota oculta y otra para la cercana a la caja) significa que el modelo es muy sobresaliente separando instancias del mismo color.
4.  **Encuentros con la línea:** Dispón los obstáculos de forma que el robot, al evitarlos, se aproxime oblicuamente a la frontera verde por lo menos un un par de veces, para que en tu video quede grabado cómo se acerca a la cinta desde diferentes ángulos e inclinaciones.
