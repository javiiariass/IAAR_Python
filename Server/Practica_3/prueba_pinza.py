import time
import sys
import os

# Ajustar el path para importar desde la carpeta Server que está un nivel arriba
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from servo import Servo

# Creación del objeto Servo
servo_obj = Servo()

# Configuración de los canales de la pinza basados en el comportamiento del robot
# Canal '1': Altura del brazo (Bajar / Levantar)
# Canal '0': Pinza (Abrir / Cerrar)

def levantar_brazo():
    print("Levantando el brazo...")
    # 140 grados suele ser hacia arriba (según Practica 2)
    servo_obj.setServoAngle('1', 140)
    time.sleep(1)

def bajar_brazo_despacio():
    print("Bajando el brazo despacio...")
    # Bajar de 140 (arriba) a 90 (abajo) gradualmente
    for angle in range(150, 90, -2):
        servo_obj.setServoAngle('1', angle)
        time.sleep(0.02)

def abrir_pinza():
    print("Abriendo la pinza...")
    # ~150 grados para abrir
    servo_obj.setServoAngle('0', 90)
    time.sleep(1)

def cerrar_pinza():
    print("Cerrando la pinza...")
    # ~90 grados suele estar cerrada (dependerá del montaje, a veces 0-90)
    servo_obj.setServoAngle('0', 135)
    time.sleep(1)

def main():
    print("Iniciando prueba de pinza y brazo...")
    
    # Para poder leer la tecla espacio sin pulsar Enter
    import termios
    import tty
    
    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    try:
        # Colocamos el brazo en posición inicial arriba antes de empezar
        levantar_brazo()
        # servo_obj.setServoAngle('0', 150)
        servo_obj.setServoAngle('0', 90)
        # servo_obj.setServoAngle('0', 90)
        
        while True:
            print("\nPRESIONA LA BARRA ESPACIADORA para coger la bola (o 'q' para salir)")
            
            char = getch()
            if char == '\x03' or char.lower() == 'q': # Ctrl+C o la letra q
                print("\nSaliendo...")
                break
            
            if char == ' ':
                print("\n----- SECUENCIA: COGER BOLA -----")
                
                abrir_pinza()
                bajar_brazo_despacio()
                time.sleep(0.5)
                
                cerrar_pinza()
                time.sleep(0.5)
                
                levantar_brazo()
                
                print("Secuencia completada. Listo para la siguiente.")
            
    except KeyboardInterrupt:
        print("\nDeteniendo prueba y relajando servos...")
        ervo_obj.setServoAngle('0', 90)
        servo_obj.setServoAngle('1', 90)
    finally:
        servo_obj.setServoStop()
        print("Programa terminado.")

if __name__ == '__main__':
    main()
