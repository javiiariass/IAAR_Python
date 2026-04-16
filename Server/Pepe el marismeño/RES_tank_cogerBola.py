from ultrasonic import Ultrasonic
from motor import tankMotor 
from servo import Servo
import time

def read_sonar_distance():
    
    ultrasonic=Ultrasonic()
    try:
        data=ultrasonic.get_distance()   #Get the value
        if data:
            return data
        else:
            print("No se pudo obtener la distancia del sonar.")
            return None
        
    except Exception as e:
        print(f"Error al leer el sonar: {e}")
        return None
    
def main():
    # SE PODRÍAN APLICAR UN ACERCAMIENTO PROGRESIVO PARA EVITAR CHOQUES, HACIENDO LA DERIVA DEL SONAR 
    # Y DETENIENDO EL TANQUE ANTES DE ALCANZAR EL OBJETIVO.

    PWM = tankMotor()

    distance = read_sonar_distance()
    print(f"Distancia medida por el sonar: {distance} cm")
    
    while distance < 2.7 and distance > 0:
        print("El tanque está en movimiento hacia el objetivo.")
        distance = read_sonar_distance()
        # AVANCE
        #PWM.setMotorModel(2000, 2000)
        # sleep
        time.sleep(0.1)
        distance = read_sonar_distance()

    print("El tanque ha alcanzado el objetivo o está fuera de rango.")
    # Detener el tanque
    #PWM.setMotorModel(0, 0)


    """
        UNA VEZ HAYAIS REALIZADO LAS PRUEBAS DE LA PINZAS, ACTUALIZAR EL CÓDIGO QUE SE EJECUTA
        AQUÍ. PROBAR QUE FUNCIONA BIEN Y QUE NO HAY PROBLEMAS DE CHOQUES O FALLOS EN LA PINZA.
    
    """
    time.sleep(1)  # Esperar un segundo antes de continuar
    servo = Servo()
    if distance>6.4 and distance<8.5:
        servo.setServoAngle('1', 90)
        time.sleep(1)
        servo.setServoAngle('0',150)
        time.sleep(1)
        servo.setServoAngle('1', 140)
        time.sleep(1)
        servo.setServoAngle('0', 90)
    # Bajar la pinza
    #servo.setServoAngle('0', 90)  # Ajusta el ángulo según sea necesario, no se cual será ni cual es el servo!!!
    #time.sleep(1)  # Esperar a que la pinza baje completamente
    
    # Cerrar la pinza
    #servo.setServoAngle('1', 0)  # Ajusta el ángulo según sea necesario, no se cual será ni cual es el servo!!!
    #time.sleep(1)  # Esperar a que la pinza cierre completamente

    
    # Levantar la pinza
    #servo.setServoAngle('0', 0)  # Ajusta el ángulo según sea necesario, no se cual será ni cual es el servo!!!
    print("Pinza levantada.")
    #time.sleep(1)  # Esperar a que la pinza se levante completamente

    #time.sleep(5)  # Esperar un poco antes de continuar

    print("Operación completada.")

    # Abrir la pinza
    #servo.setServoAngle('1', 90)  # Ajusta el ángulo según sea necesario, no se cual será ni cual es el servo!!!
    #time.sleep(1)  # Esperar a que la pinza abra completamente
    print("Pinza abierta.")


if __name__ == "__main__":
    main()