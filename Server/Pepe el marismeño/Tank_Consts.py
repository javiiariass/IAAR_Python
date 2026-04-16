from enum import Enum
"""
  os.system('clear')  # Limpiar la pantalla
        print("Iniciando operación de pinza...")
        print("1. Corre fermin")
        print("2. Para el carro")
        print("3. Levantar pinza")
        print("4. Coger y subir")
        print("0. Salir")
        opc = int(input("Selecciona una opción (1-4): "))
        usarServo(opc)
    
    if opc == 1:
    PWM.setMotorModel(-2000,-1380)
        time.sleep(2)  
    elif opc == 2:
        PWM.setMotorModel(0, 0)
        time.sleep(1)
    elif opc == 3:
        servo.setServoAngle('0', 150)
        time.sleep(1)
    elif opc == 4:  
        servo.setServoAngle('1', 90)
        time.sleep(1)
        servo.setServoAngle('0',150)
        time.sleep(1)
        servo.setServoAngle('1', 140)
    elif opc == 0:
        print("Saliendo del programa.")

"""


class VelocidadInfo(Enum):
    # Recto hacia alante
    DelanteIzq = 2000 
    DelanteDer = 2000
    # Recto hacia atras
    AtrasIzq = -2000
    AtrasDer = -2000

class ArticulacionInfo(Enum):
    Pinza = 0
    PinzaMinAngle = 90
    PinzaMaxAngle = 140 # Originalmente 150
    Brazo = 1
    BrazoMinAngle = 90 # Creo q es 90, estaba puesto 0
    BrazoMaxAngle = 140

class sonarInfo(Enum):
    pelotaMin = 6.4
    pelotaMax = 8.5