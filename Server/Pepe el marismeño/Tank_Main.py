#from servo import Servo
import time
#from motor import tankMotor
import os
#from ultrasonic import Ultrasonic

from Tank_Consts import ArticulacionInfo, sonarInfo, VelocidadInfo
from multiprocessing import Manager, Process
from Estado_compartido import EstadoCompartido
from PruebaParaEstado import bucle_vision
from tank_pruebaServos import ejecutar_menu

# --------------------------------------------------------------------------------------------------------------------------------------
# Variables
# --------------------------------------------------------------------------------------------------------------------------------------
intentandoCapturarPelota = False # Se activa si intento capturar pelota para saber si se pierde y no evadir mal



# --------------------------------------------------------------------------------------------------------------------------------------
# Funciones Principales
# --------------------------------------------------------------------------------------------------------------------------------------
def iniciar():
    PWM = tankMotor()
    servo = Servo()
    return PWM, servo

def read_sonar_distance():
    print( end='')
    
    return 60
"""    
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
"""
           


def peligroPorSonar():
    distance = read_sonar_distance()
    return False
"""""
    if distance < (sonarInfo.pelotaMin.value-2):
        return True
    else:
        return False
"""   
def cogerPelota(servo):
    distance = read_sonar_distance()
    if distance>sonarInfo.pelotaMin.value and distance<sonarInfo.pelotaMax.value:
        print("Cogiendo pelota")
        servo.setServoAngle(ArticulacionInfo.Brazo.value, ArticulacionInfo.BrazoMinAngle.value)
        time.sleep(1)
        servo.setServoAngle(ArticulacionInfo.Pinza.value, ArticulacionInfo.PinzaMaxAngle.value)
        time.sleep(1)
        servo.setServoAngle(ArticulacionInfo.Brazo.value, ArticulacionInfo.BrazoMaxAngle.value)
        time.sleep(1)
        servo.setServoAngle(ArticulacionInfo.Pinza.value, ArticulacionInfo.PinzaMinAngle.value)
        return True
    else: 
        return False

def avanzar(PWM):
    PWM.setMotorModel(VelocidadInfo.DelanteDer.value,VelocidadInfo.DelanteIzq.value)
    time.sleep(0.2) 

def retroceder(PWM):
    PWM.setMotorModel(VelocidadInfo.AtrasDer.value,VelocidadInfo.AtrasIzq.value)
    time.sleep(0.2) 

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

def peligroPorImagen(estado):
    datos = estado.obtener_clase(2)
   
   
    for obj in datos:
        if obj.get("cy", 1000) <= 470:
            print("📦 Línea verde detectada:", datos)
            return True
    
    return False

    
def pelotaVisible(estado):
    datos = estado.obtener_clase(3)
    if datos:
        print("📦 Pelota Roja detectada:", datos)
        print("Hay al menos un objeto")
        return True
    else:
        return False
   
    
    
    

def irAPelotaVisible(estado): # No puede ser bucle
    print( end='')
    datos = estado.obtener_clase(3)
    desviado=True
    while desviado:
        if datos[0].get("cx",1000) <= 300:
            print( end='')
        #girar_grados(PWM, -10)
        if datos[0].get("cx",1000) >= 340:
            print( end='')
        #girar_grados(PWM, 10)
        if datos[0].get("cx",1000) <= 300 or datos[0].get("cx",1000) >= 340:
            desviado=False
    #if(cogerPelota(servo)):
     #   print( end='')
    #else:
     #   print( end='')     
        #avanzar(PWM)



        
        






def hayPeligro(estado):
    return False
    #return (peligroPorSonar() or peligroPorImagen(estado))

def evasion(estado):
    #girar_grados(PWM, 90)
    if hayPeligro(estado):
        print( end='')
        #girar_grados(PWM, -180)
        if hayPeligro(estado): 
            print( end='')  
          #girar_grados(PWM, -90)  
          #avanzar(PWM)
          #girar_grados(PWM, 90)
            if hayPeligro(estado):
                print( end='')
                #girar_grados(PWM, -180)
                if hayPeligro(estado):
                    print( end='')
                    #girar_grados(PWM, 90)
                    if hayPeligro(estado):
                        print( end='')
                        
                    else:
                        print( end='')
                        #avanzar(PWM)
                else:
                    print( end='')
                    #avanzar(PWM)
                    #girar_grados(PWM, -90)

            else:
                print( end='')
                #avanzar(PWM)
                #girar_grados(PWM, 90)


        else:
            print( end='')
            #avanzar(PWM)
            #girar_grados(PWM, 90)


    else:
        print( end='')
        #avanzar(PWM)
        #girar_grados(PWM, -90)



def hexapodoVisible(estado):
    datos = estado.obtener_clase(1)
    if datos:
        print("📦 Hexapodo detectada:", datos)
        print("Hay al menos un objeto")
        return True
    else:
        return False
def irACoordenadasHexapodo(mensaje): # No puede ser bucle
    print( end='')



def moverAHexapodo(estado): # No puede ser bucle
    print( end='')
    datos = estado.obtener_clase(1)
    desviado=True
    while desviado:
        if datos[0].get("cx",1000) <= 300:
            print( end='')
        #girar_grados(PWM, -10)
        if datos[0].get("cx",1000) >= 340:
            print( end='')
        #girar_grados(PWM, 10)
        if datos[0].get("cx",1000) <= 300 or datos[0].get("cx",1000) >= 340:
            desviado=False    
    #avanzar(PWM)

def recogerMensaje():
    return 1
# --------------------------------------------------------------------------------------------------------------------------------------
# Main y selecciones
# --------------------------------------------------------------------------------------------------------------------------------------

def main():
    #PWM, servo = iniciar()
    # Ponemos el brazo hacia arriba y pinza abierta
    #servo.setServoAngle(ArticulacionInfo.Brazo.value, ArticulacionInfo.BrazoMaxAngle.value)
    #servo.setServoAngle(ArticulacionInfo.Pinza.value, ArticulacionInfo.PinzaMaxAngle.value)
     with Manager() as manager:
        datos = manager.dict({i: manager.list() for i in range(5)})
        lock = manager.Lock()
        estado = EstadoCompartido(datos, lock)

        p1 = Process(target=bucle_vision, args=(estado,))
        p1.start()

        time.sleep(1)  # Deja que arranque la simulación

        pelotaCogida = False
        pelotavista =False
        while not pelotaCogida:
            if hayPeligro(estado):#hayPeligro(estado,PWM):
                evasion(estado)
            else:
                mensaje = recogerMensaje()

                if mensaje is not None:
                    if pelotaVisible(estado):
                        pelotavista=True
                        pelotaCogida = irAPelotaVisible(estado)
                    elif pelotavista:
                        print( end='')
                        #retroceder(PWM)
                        pelotavista=False
                    elif hexapodoVisible(estado):
                        moverAHexapodo(estado)
                    else:
                        irACoordenadasHexapodo(mensaje)
            estado.limpiar()

        p1.terminate()
        p1.join()
                    

if __name__ == "__main__":
    main()
    