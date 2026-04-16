import time
from infrared import Infrared

def main():
    print("INICIANDO PRUEBA DE INFRARROJOS (Sin motores)...")
    print("Mueve un folio blanco o una cinta negra por debajo del frontal del robot.\n")
    
    # Inicializar el módulo de infrarrojos (tiene 3 sensores)
    ir = Infrared()
    
    # Dar un pequeño respiro al iniciar
    time.sleep(1)

    try:
        while True:
            # Leer los valores individuales
            # Normalmente: Canal 1 (Izq), Canal 2 (Centro), Canal 3 (Der)
            izq = ir.read_one_infrared(1)
            cen = ir.read_one_infrared(2)
            der = ir.read_one_infrared(3)

            # Evaluamos si alguno de los sensores detecta la línea (asumimos que 1 = línea detectada)
            # NOTA: Si al ponerlo en la línea verde los sensores marcan 0, cambia los '== 1' por '== 0'
            linea_detectada = (izq == 1) or (cen == 1) or (der == 1)
            
            if linea_detectada:
                estado = "🟢 ¡Línea verde detectada!"
            else:
                estado = "⚪ Suelo libre..."

            # Imprimir en una misma línea reemplazándola para limpiar la terminal
            print(f"Sensores [Izq:{izq} Cen:{cen} Der:{der}] --> {estado}          ", end="\r")
            
            # Pequeña pausa para no saturar la CPU
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nPrograma interrumpido por el usuario.")

    finally:
        # Siempre liberar los pines GPIO al final
        ir.close()
        print("Sensores liberados de forma segura.")

if __name__ == '__main__':
    main()