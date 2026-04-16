from multiprocessing import Manager, Process
import time
from Estado_compartido import EstadoCompartido
from PruebaParaEstado import bucle_vision
from tank_pruebaServos import ejecutar_menu

if __name__ == '__main__':
    with Manager() as manager:
        datos = manager.dict({i: manager.list() for i in range(5)})
        lock = manager.Lock()
        estado = EstadoCompartido(datos, lock)
        # Solo la simulación va en proceso separado
        p1 = Process(target=bucle_vision, args=(estado,))
        p1.start()
        time.sleep(1)  # Deja que arranque la simulación
        # El menú se ejecuta en el proceso principal
        ejecutar_menu(estado)
        p1.terminate()
        p1.join()
