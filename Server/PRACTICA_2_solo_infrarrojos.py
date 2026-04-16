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

            # Generar algo muy visual para la terminal. 
            # NOTA: Dependiendo de tu hardware, 1 puede ser Blanco y 0 Negro, o viceversa.
            # Representaremos el 1 como 🟩 y el 0 como ⬛ solo para que destaque el cambio.
            viz_izq = "🟩" if izq == 1 else "⬛"
            viz_cen = "🟩" if cen == 1 else "⬛"
            viz_der = "🟩" if der == 1 else "⬛"

            # Imprimir en una misma línea reemplazándola (usando retorno de carro \r) para limpiar la terminal
            print(f"Sensores (Izq, Cen, Der): [{izq}] [{cen}] [{der}]  -->  {viz_izq} {viz_cen} {viz_der}      ", end="\r")
            
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