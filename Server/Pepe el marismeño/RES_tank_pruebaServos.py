from servo import Servo
import time
from motor import tankMotor
from Tank_Consts import ArticulacionInfo, sonarInfo, VelocidadInfo
import os

def main():
    
    opc = 1
    print("Control de Pinza con Servo")
    while opc != 0:
        os.system('clear')  # Limpiar la pantalla
        print("Iniciando operación de pinza...")
        print("1. Corre fermin")
        print("2. Para el carro")
        print("3. Giro")
        print("4. Coger y subir")
        print("0. Salir")
        opc = int(input("Selecciona una opción (1-4): "))
        usarServo(opc)

def girar_grados(PWM, grados):
    """
    Gira el robot una cantidad de grados.
    grados > 0 → derecha
    grados < 0 → izquierda
    """

    # Constantes de calibración (ajústalas con pruebas)
    TIEMPO_POR_GRADO_DERECHA = 0.01  # tiempo en segundos por grado hacia la derecha
    TIEMPO_POR_GRADO_IZQUIERDA = 0.01  # diferente si hacia atrás responde distinto

    if grados > 0:
        tiempo = grados * TIEMPO_POR_GRADO_DERECHA
        PWM.setMotorModel(VelocidadInfo.DelanteIzq.value, VelocidadInfo.AtrasDer.value)  # Giro en el lugar hacia la derecha
    else:
        tiempo = abs(grados) * TIEMPO_POR_GRADO_IZQUIERDA
        PWM.setMotorModel(VelocidadInfo.AtrasIzq.value, VelocidadInfo.DelanteDer.value)  # Giro hacia la izquierda

    time.sleep(tiempo)
    PWM.setMotorModel(0, 0)  # Detener motores

def usarServo(opc):
    PWM = tankMotor()
    servo = Servo()
    if opc == 1:
        PWM.setMotorModel(-2000,-1380)
        time.sleep(2)  
    elif opc == 2:
        PWM.setMotorModel(0, 0)
        time.sleep(1)
    elif opc == 3:
        girar_grados(PWM, 90)
    elif opc == 4:  
        servo.setServoAngle('1', 90)
        time.sleep(1)
        servo.setServoAngle('0',150)
        time.sleep(1)
        servo.setServoAngle('1', 140)
    elif opc == 0:
        print("Saliendo del programa.")
    else:  
        print("Opción no válida. Por favor, selecciona una opción entre 1 y 4.")
        input("Presiona Enter para continuar...")
        

if __name__ == "__main__":
    main()