import socket
import threading
import struct
import os

class RobotClient:
    """
    Clase para el cliente del robot que se conecta a un servidor.
    Permite enviar y recibir mensajes de texto y archivos.
    """

    def __init__(self, host='localhost', port=5000):
        """
            Inicializa el cliente con la dirección y el puerto del servidor.
            :param host: Dirección del servidor (por defecto 'localhost').
            :param port: Puerto del servidor (por defecto 5000).
        """
        self.host = host
        self.port = port
        self.sock = socket.socket()

    def connectar(self):
        """
            Conecta el cliente al servidor.
        """
        self.sock.connect((self.host, self.port))
        print(f"[Cliente] Conectado a {self.host}:{self.port}")
        threading.Thread(target=self.recibir_datos, daemon=True).start()

    def recibir_datos(self):
        """
            Escucha y recibe datos del servidor.
            Maneja mensajes de texto y archivos.
        """

        while True:
            try:
                tipo = self.sock.recv(4).decode()
                if not tipo:
                    return False
                    break

                if tipo == 'EXIT':
                    print("[Cliente] El servidor cerró la conexión.")
                    self.sock.close()
                    return False
                    break

                elif tipo == 'TEXT':
                    mensaje = self.sock.recv(1024).decode()
                    print(f"[Servidor]: {mensaje}")
                    return True

                elif tipo == 'FILE':
                    nombre_len = struct.unpack('I', self.sock.recv(4))[0]
                    nombre_archivo = self.sock.recv(nombre_len).decode()
                    tamano = struct.unpack('I', self.sock.recv(4))[0]
                    contenido = b''
                    while len(contenido) < tamano:
                        datos = self.sock.recv(1024)
                        contenido += datos
                    with open(nombre_archivo, 'wb') as f:
                        f.write(contenido)
                    print(f"[Archivo recibido] {nombre_archivo} ({tamano} bytes)")
                    return True

            except Exception as e:
                print("[Cliente] Error al recibir datos:", e)
                self.sock.close()
                break

    def enviar_texto(self, mensaje):
        """
            Envía un mensaje de texto al servidor.
            :param mensaje: Mensaje a enviar.
        """
        try:
            self.sock.send("TEXT".encode())
            self.sock.send(mensaje.encode())
        except Exception as e:
            print("[Cliente] Error al enviar texto:", e)

    def enviar_archivo(self, ruta):
        """
            Envía un archivo al servidor.
            :param ruta: Ruta del archivo a enviar.
        """
        if not os.path.isfile(ruta):
            print("[Cliente] Archivo no encontrado.")
            return
        try:
            with open(ruta, 'rb') as f:
                contenido = f.read()
            nombre = os.path.basename(ruta)
            self.sock.send("FILE".encode())
            self.sock.send(struct.pack('I', len(nombre)))
            self.sock.send(nombre.encode())
            self.sock.send(struct.pack('I', len(contenido)))
            self.sock.sendall(contenido)
            print(f"[Archivo enviado] {nombre}")
        except Exception as e:
            print("[Cliente] Error al enviar archivo:", e)

    def cerrar(self):
        """
            Cierra la conexión del cliente.
        """
        
        try:
            self.sock.send("EXIT".encode())
            
        except:
            pass
        print("[Cliente] Cerrando conexión...")
        self.sock.close()
