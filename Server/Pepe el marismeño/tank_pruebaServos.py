import time

def ejecutar_menu(ec):
    opc = -1
    print("🛠️ Control de Pinza con Servo")
    while opc != 0:
        print("\n--- Menú ---")
        print("1. Mostrar datos de la clase 0")
        print("2. Limpiar todos los datos")
        print("3. Mostrar datos de la clase 3")
        print("4. Mostrar datos de la clase 4")
        print("0. Salir")

        try:
            opc = int(input("Selecciona una opción: "))
        except ValueError:
            print("❌ Entrada inválida.")
            continue

        usarServo(opc, ec)

def usarServo(opc, ec):
    if opc == 1:
        datos = ec.obtener_clase(0)
        print("📦 Clase 0:", datos)
       
    elif opc == 2:
        ec.limpiar()
        print("🧹 Datos limpiados.")
    elif opc == 3:
        datos = ec.obtener_clase(3)
        print("📦 Clase 3:", datos)
    elif opc == 4:
        datos = ec.obtener_clase(4)
        print("📦 Clase 4:", datos)
    elif opc == 0:
        print("👋 Saliendo del programa.")
    else:
        print("❌ Opción no válida.")
    time.sleep(1)
