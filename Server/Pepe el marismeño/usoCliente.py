from RobotClient import RobotClient 

def main():
    """
        Función principal para ejecutar el cliente del robot.
        Permite conectarse a un servidor y enviar mensajes o archivos.
    """
    cliente = RobotClient(host='localhost', port=5000)
    cliente.connect()

    while True:
        entrada = input("Comando (/exit, /file ruta, mensaje): ")
        if entrada.strip() == "/exit":
            cliente.cerrar()
            break
        elif entrada.startswith("/file "):
            ruta = entrada[6:]
            cliente.enviar_archivo(ruta)
        else:
            cliente.enviar_texto(entrada)

if __name__ == "__main__":
   main()