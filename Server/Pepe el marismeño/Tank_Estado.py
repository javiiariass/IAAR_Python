from threading import Lock, Event

datos = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []
}

_lock = Lock()
actualizado_event = Event()

def agregar(cx, cy, clase, score):
    if clase not in datos:
        return
    with _lock:
        
        datos[clase].append({
            "cx": cx,
            "cy": cy,
            "score": score
        })
    actualizado_event.set()

def obtener_todo():
    
    actualizado_event.wait()
    with _lock:
        return {k: v.copy() for k, v in datos.items()}

def obtener_clase(clase):
    actualizado_event.wait()
    with _lock:
        return datos.get(clase, []).copy()

def limpiar():
    with _lock:
        for k in datos:
            datos[k].clear()
    actualizado_event.clear()
