import time
import random

def bucle_vision(estado_compartido):
    print("📷 Simulación de detecciones iniciada...")
    try:
        while True:
            for clase in range(5):
                num_detecciones = random.randint(0, 2)
                for _ in range(num_detecciones):
                    cx = random.randint(100, 540)
                    cy = random.randint(100, 380)
                    score = round(random.uniform(0.5, 0.99), 2)
                    estado_compartido.agregar(cx, cy, clase, score)
            time.sleep(1.5)
    except KeyboardInterrupt:
        print("📷 Simulación detenida.")
