import time
from ultrasonic import Ultrasonic

def main():
    print("INICIANDO PRUEBA DE SONAR (ULTRASONIDOS)...")
    print("Acerca la mano o un obstáculo al frontal del robot para ver la distancia.\n")
    
    # Inicializar el sensor de ultrasonidos (sonar)
    # Utilizamos el context manager (with) si está disponible, o inicializamos normal
    sonar = Ultrasonic()
    
    # Dar un pequeño respiro al iniciar hardware
    time.sleep(1)

    try:
        while True:
            # Obtener distancia en centímetros
            distancia = sonar.get_distance()
            
            # Formatear la salida visual
            if distancia >= 0:
                if distancia < 20.0:
                    estado = "🛑 ¡PELIGRO! Muy cerca "
                elif distancia < 40.0:
                    estado = "⚠️ Precaución, obstáculo"
                else:
                    estado = "✅ Vía libre            "
                
                # Imprimir en una misma línea reemplazándola
                print(f"Distancia medida: {distancia:5.1f} cm  -->  {estado}      ", end="\r")
            else:
                print("Error de lectura en el sensor...                       ", end="\r")
            
            # Pequeña pausa para no saturar al sensor de pulsos
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nPrograma interrumpido por el usuario.")

    finally:
        # Siempre liberar los pines GPIO al final
        sonar.close()
        print("Sensor ultrasónico liberado de forma segura.")


if __name__ == '__main__':
    main()