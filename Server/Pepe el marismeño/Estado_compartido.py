class EstadoCompartido:
    

    def __init__(self, datos, lock,):
        self.datos = datos
        self.lock = lock
        self.actualizado=False
    def agregar(self, cx, cy, X1, X2,Y1,Y2, clase, score):
        if clase not in self.datos:
            return
        with self.lock:
            self.datos[clase].append({
                "X1": X1,
                "X2": X2,
                "Y1": Y1,
                "Y2": Y2,
                "cx": cx,
                "cy": cy,
                "score": score
            })
            self.actualizado=True    

    def obtener_clase(self, clase):
        with self.lock:
            return list(self.datos.get(clase, []))
    def estado_actualizado(self):
        with self.lock:
            return self.actualizado

    def obtener_todo(self):
        with self.lock:
            return {k: list(v) for k, v in self.datos.items()}

    def limpiar(self):
        with self.lock:
            for k in self.datos:
                self.datos[k][:] = []
            self.actualizado = False
