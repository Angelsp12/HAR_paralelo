"""
===========================================================
ENTRENAMIENTO HAR – SECUENCIAL Y PARALELO
Módulo reutilizable para benchmarking
===========================================================
"""

import time
from joblib import Parallel, delayed


# =========================================================
# FUNCIONES DE ENTRENAMIENTO (SIMULADAS)
# =========================================================
# NOTA:
# Estas funciones simulan carga computacional.
# Más adelante puedes sustituir time.sleep()
# por tu entrenamiento real del notebook.
# =========================================================

def entrenar_secuencial():
    print("   → Entrenando en modo SECUENCIAL...")
    time.sleep(3)  # Simula entrenamiento pesado
    print("   ✓ Entrenamiento secuencial finalizado")


def entrenar_paralelo(n_jobs):
    print(f"   → Entrenando en modo PARALELO con {n_jobs} núcleos...")
    Parallel(n_jobs=n_jobs)(
        delayed(time.sleep)(3) for _ in range(n_jobs)
    )
    print("   ✓ Entrenamiento paralelo finalizado")


# =========================================================
# FUNCIÓN PRINCIPAL (IMPORTABLE)
# =========================================================
def entrenar_modelo(modo="secuencial", n_jobs=1):

    if modo == "secuencial":
        entrenar_secuencial()

    elif modo == "paralelo":
        entrenar_paralelo(n_jobs)

    else:
        raise ValueError("Modo no válido. Usa 'secuencial' o 'paralelo'")
