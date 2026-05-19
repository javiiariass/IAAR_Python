import time
from ultrasonic import Ultrasonic

def main():
    sonar = Ultrasonic()
    print("Iniciando lectura de sonar (Presiona Ctrl+C para salir)...")
    try:
        while True:
            distancia = sonar.get_distance()
            if distancia < 0:
                print(f"Distancia: Error      ", end="\r")
            else:
                print(f"Distancia: {distancia:5.1f} cm ", end="\r")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario.")
    finally:
        # Asegurarse de liberar/limpiar los pines del sonar
        sonar.close()
        print("Fin del programa.")

if __name__ == '__main__':
    main()
