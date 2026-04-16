from multiprocessing import Process
import time
import Tank_Estado as estado
import RES_tank_pruebaServos
import Tank_CamaraConIntermedio


if __name__ == '__main__':
    estado = EstadoCompartido()

    p1 = Process(target=Tank_CamaraConIntermedio.main, args=(estado,))
    p2 = Process(target=Res_tank_pruebaServos.main, args=(estado,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()