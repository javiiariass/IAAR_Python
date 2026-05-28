#!/usr/bin/env python3
"""
fix_class_ids.py — Corrige las inconsistencias de IDs de clase en el dataset.

Problema: labelImg etiquetó con predefined_classes.txt que tenía 15 clases basura
antes de las nuestras, resultando en:
  - clase 16 = bola_roja (debería ser 0)
  - clase 17 = linea_verde (debería ser 1)

Mientras que los scripts automáticos sí usaron 0 y 1 correctamente.

Este script unifica todo a: 0 = bola_roja, 1 = linea_verde
"""
import os
import glob

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset_clasificacion")
MAPPING = {"16": "0", "17": "1"}

corregidos = 0
total = 0

for txt_path in sorted(glob.glob(os.path.join(DATASET_DIR, "*.txt"))):
    basename = os.path.basename(txt_path)
    if basename == "classes.txt":
        continue

    total += 1
    lines = open(txt_path).readlines()
    new_lines = []
    changed = False

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        # Ignorar líneas con texto como "0=bola_roja" o "1=linea_verde"
        if "=" in parts[0]:
            changed = True
            continue
        # Mapear IDs incorrectos
        if parts[0] in MAPPING:
            parts[0] = MAPPING[parts[0]]
            changed = True
        # Verificar que el ID resultante es 0 o 1
        if parts[0] not in ("0", "1"):
            print(f"  ADVERTENCIA: {basename} tiene clase desconocida '{parts[0]}' — se conserva")
        new_lines.append(" ".join(parts) + "\n")

    if changed:
        with open(txt_path, "w") as f:
            f.writelines(new_lines)
        corregidos += 1
        print(f"  Corregido: {basename}")

# Reescribir classes.txt limpio
classes_path = os.path.join(DATASET_DIR, "classes.txt")
with open(classes_path, "w") as f:
    f.write("bola_roja\nlinea_verde\n")

print(f"\n=== RESULTADO ===")
print(f"Archivos procesados: {total}")
print(f"Archivos corregidos: {corregidos}")
print(f"classes.txt reescrito: {classes_path}")

# Verificación final
print(f"\n=== VERIFICACIÓN ===")
from collections import Counter
counter = Counter()
for txt_path in glob.glob(os.path.join(DATASET_DIR, "*.txt")):
    if os.path.basename(txt_path) == "classes.txt":
        continue
    for line in open(txt_path):
        parts = line.strip().split()
        if parts:
            counter[parts[0]] += 1

for class_id, count in sorted(counter.items()):
    label = "bola_roja" if class_id == "0" else ("linea_verde" if class_id == "1" else "DESCONOCIDA")
    print(f"  Clase {class_id} ({label}): {count} anotaciones")
