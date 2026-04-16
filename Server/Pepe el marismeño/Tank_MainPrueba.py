from servo import Servo
import time
from motor import tankMotor
import os
from ultrasonic import Ultrasonic
import random
from Tank_Consts import ArticulacionInfo, sonarInfo, VelocidadInfo
from multiprocessing import Manager, Process
from Estado_compartido import EstadoCompartido
from PruebaParaEstado import bucle_vision
from tank_pruebaServos import ejecutar_menu
from Tank_CamaraConIntermedio import camara
from RobotClient import RobotClient 

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


def peligroPorSonar():
    distance = read_sonar_distance()
    print("dist: ", distance)
    if distance < (sonarInfo.pelotaMin.value-2):
        return True
        print("true sonar")
    else:
        return False
        print("false sonar")
  
def cogerPelota(servo,PWM):
    distance = read_sonar_distance()
    if distance>sonarInfo.pelotaMin.value and distance<sonarInfo.pelotaMax.value:
        print("Cogiendo pelota")
        servo.setServoAngle(ArticulacionInfo.Brazo.value, ArticulacionInfo.BrazoMinAngle.value)
        time.sleep(1)
        servo.setServoAngle(ArticulacionInfo.Pinza.value, ArticulacionInfo.PinzaMaxAngle.value)
        time.sleep(1)
        servo.setServoAngle(ArticulacionInfo.Brazo.value, ArticulacionInfo.BrazoMaxAngle.value)
        time.sleep(1)
        avanzar(PWM)
        servo.setServoAngle(ArticulacionInfo.Pinza.value, ArticulacionInfo.PinzaMinAngle.value)
        return True
    elif distance<sonarInfo.pelotaMin.value:
        retrocedertmp(PWM, 0.2)
        print("esta muy cerca")
    else:
        if distance>14:
            avanzar(PWM)
            print("esta lejos")
        else:
            avanzartmp(PWM, 0.1)
            
        return False
    
def parar(PWM):
    PWM.setMotorModel(0,0)

def avanzar(PWM):
    PWM.setMotorModel(VelocidadInfo.DelanteDer.value,VelocidadInfo.DelanteIzq.value)
    time.sleep(0.5)
    parar(PWM)

def avanzartmp(PWM, tiempo):
    PWM.setMotorModel(VelocidadInfo.DelanteDer.value,VelocidadInfo.DelanteIzq.value)
    time.sleep(tiempo)
    parar(PWM)

def avanzardist(PWM, distancia):
    tiempo_necesario = distancia / 100
    PWM.setMotorModel(VelocidadInfo.DelanteDer.value, VelocidadInfo.DelanteIzq.value)
    time.sleep(tiempo_necesario)
    parar(PWM)

def retroceder(PWM):
    PWM.setMotorModel(VelocidadInfo.AtrasDer.value,VelocidadInfo.AtrasIzq.value)
    time.sleep(0.2)
    parar(PWM)

def retrocedertmp(PWM, tiempo):
    PWM.setMotorModel(VelocidadInfo.AtrasDer.value,VelocidadInfo.AtrasIzq.value)
    time.sleep(tiempo)
    parar(PWM)

def retrocederdist(PWM, distancia):
    tiempo_necesario = distancia / 15
    PWM.setMotorModel(VelocidadInfo.AtrasDer.value, VelocidadInfo.AtrasIzq.value)
    time.sleep(tiempo_necesario)
    parar(PWM)

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
    if datos:
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
   




def irAPelotaVisible(estado, PWM, servo):
    datos = estado.obtener_clase(3)
    if not datos:
        print("❌ No se detecta pelota.")
        return

    pelota = datos[0]
    cx = pelota.get("cx", 320)  # Centro horizontal de la pelota
    cy = pelota.get("cy", 320)  # Centro vertical de la pelota
    
    print(cx)
    print(cy)
    
    # 1. Corregir orientación (centrado horizontal)
    error_horizontal = cx - 320  # Imagen es de 600x600
    print("err", error_horizontal)
    if abs(error_horizontal) > 25:
        grados = error_horizontal * 0.1  # Ajusta esta constante según tu robot
        print(f"↪ Girando {grados:.1f}° para centrar pelota")
        girar_grados(PWM, grados)
    else:
        print("✅ Pelota centrada horizontalmente.")
        #irAPelotaVisible(estado, PWM, servo)

        if cogerPelota(servo, PWM):
            print("🟢 Pelota capturada.")
            return True
        else:
            return False
    


def hayPeligro(estado):
    return (peligroPorSonar() or peligroPorImagen(estado))

def evasion2(estado, PWM):
    giros = [90, -180, -90, 90, -180, 90]
    for angulo in giros:
        girar_grados(PWM, angulo)  
        time.sleep(1.5)
        estado.limpiar()
        while estado.obtener_todo()== []:
            time.sleep(0.5)
        if not hayPeligro(estado):
            print(f"✅ Peligro evitado con giro de {angulo}°")
            avanzardist(PWM, 5)
            return
    print("⚠️ No se pudo evitar el peligro tras varios intentos")


def evasion1(PWM):
    girar_grados(PWM, 90)

def hexapodoVisible(estado):
    datos = estado.obtener_clase(1)
    if datos:
        print("📦 Hexapodo detectada:", datos)
        print("Hay al menos un objeto")
        return True
    else:
        return False
def irACoordenadasHexapodo(cliente): # No puede ser bucle

    return False


def deambular(PWM):
    numero=random.random()
    if numero<=0.8:
        avanzar(PWM)
    elif  numero>0.9:
        girar_grados(PWM, 90)
    else:
        girar_grados(PWM, -90)



def moverAHexapodo(estado,PWM,cliente): # No puede ser bucle
   
    datos = estado.obtener_clase(1)
    
    if not datos:
        print("❌ No se detecta Hexapodo Nuestro.")
        return

    pelota = datos[0]
    cx = pelota.get("cx", 320)  # Centro horizontal de la pelota
    cy = pelota.get("cy", 320)  # Centro vertical de la pelota

    # 1. Corregir orientación (centrado horizontal)
    error_horizontal = cx - 320  # Imagen es de 600x600
    if abs(error_horizontal) > 25:
        grados = error_horizontal * 0.1  # Ajusta esta constante según tu robot
        print(f"↪ Girando {grados:.1f}° para centrar Hexapodo")
        girar_grados(PWM, grados)
    else:
        print("✅ Hexapodo centrada horizontalmente.")
        avanzar(PWM)
        if(peligroPorSonar()):
            cliente.enviar_texto("He encontrado hexapodo")




def recogerMensaje(cliente):
    return cliente.recibir_datos()

def relajar(servo):
    servo.setServoAngle(ArticulacionInfo.Brazo.value, ArticulacionInfo.BrazoMinAngle.value)
    servo.setServoAngle(ArticulacionInfo.Pinza.value, ArticulacionInfo.PinzaMinAngle.value)
# --------------------------------------------------------------------------------------------------------------------------------------
# Main y selecciones
# --------------------------------------------------------------------------------------------------------------------------------------

def main():
    PWM, servo = iniciar()
   # cliente = RobotClient(host='localhost', port=5000)
   # cliente.connectar()

    servo.setServoAngle(ArticulacionInfo.Brazo.value, ArticulacionInfo.BrazoMaxAngle.value)
    servo.setServoAngle(ArticulacionInfo.Pinza.value, ArticulacionInfo.PinzaMinAngle.value)


    with Manager() as manager:
        datos = manager.dict({i: manager.list() for i in range(5)})
        lock = manager.Lock()
        estado = EstadoCompartido(datos, lock)

        p1 = Process(target=camara, args=(estado,))
        p1.start()
        time.sleep(1)  # Deja tiempo a la cámara

        while True:
            print("\n--- MENÚ DE PRUEBA ---")
            print("1. Primer Hito Comunicación")
            print("2. Segundo Hito Detección de lineas verdes y comportamiento reactivo")
            print("3. Segundo y Tercer Hito localizar compañero")
            print("4. Cuarto Hito Detectar bola roja y pillarla")
            print("5. Avanzar")
            print("6. Retroceder")
            print("7. Girar 90º")
            print("8. Girar -90º")
            print("9. Deambular")
            print("10. Sonar")
            print("0. Salir")
            pelotavista=False
            opcion = input("Selecciona una opción: ")
            estado.limpiar()
            time.sleep(1.5)
            #estado.limpiar()
            #while not datos:
             #   time.sleep(0.5)
                
            if opcion == "1":
                cliente = RobotClient(host='192.168.50.139', port=5000)
                cliente.connectar()
                cliente.enviar_texto("Pelota Cogida")
                mensaje = recibir_datos(cliente)
                print("El mensaje recibido es :",mensaje)
                
            elif opcion == "2":
                if hayPeligro(estado):
                    evasion2(estado,PWM)
                
            elif opcion == "3":
                if hexapodoVisible(estado):
                    moverAHexapodo(estado,PWM,cliente)
                else:
                    deambular(PWM)
            elif opcion == "4":
                irAPelotaVisible(estado, PWM, servo)
                    
            elif opcion == "5":
                avanzar(PWM)
            elif opcion == "6":
               retroceder(PWM)
            elif opcion == "7":
                girar_grados(PWM,90)
            elif opcion == "8":
                girar_grados(PWM,-90)
            elif opcion == "9":
                deambular(PWM)
            elif opcion == "10":
                distance = read_sonar_distance()
                print("distancia", distance)
            elif opcion == "0":
                relajar(servo)
                print("Finalizando prueba.")
                break
            else:
                print("Opción no válida.")
        
        p1.terminate()
        p1.join()

if __name__ == "__main__":
    main()
    