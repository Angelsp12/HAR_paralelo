"""
===========================================================
BENCHMARK DE CÓMPUTO PARALELO – HAR
Mide tiempo, speedup y eficiencia
===========================================================
"""

import time
import csv
import os
from entrenamiento import entrenar_modelo


# =========================================================
# CONFIGURACIÓN
# =========================================================
NUCLEOS = [1, 2, 4, 8, 12, 16, 20]
CARPETA_RESULTADOS = "Resultados"
ARCHIVO_CSV = os.path.join(CARPETA_RESULTADOS, "benchmark_tiempos.csv")


# =========================================================
# FUNCIÓN DE MEDICIÓN
# =========================================================
def medir_tiempo(modo, n_jobs):
    inicio = time.perf_counter()
    entrenar_modelo(modo=modo, n_jobs=n_jobs)
    fin = time.perf_counter()
    return fin - inicio


# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================
def main():

    print("==============================================")
    print("  INICIANDO BENCHMARK DE CÓMPUTO PARALELO HAR  ")
    print("==============================================\n")

    # Crear carpeta de resultados si no existe
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

    resultados = []

    # 1️⃣ Ejecución secuencial (baseline)
    print("Ejecutando entrenamiento SECUENCIAL (1 núcleo)\n")
    tiempo_base = medir_tiempo("secuencial", 1)

    resultados.append({
        "modo": "secuencial",
        "nucleos": 1,
        "tiempo": round(tiempo_base, 4),
        "speedup": 1.0,
        "eficiencia": 100.0
    })

    # 2️⃣ Ejecuciones paralelas
    for n in NUCLEOS[1:]:
        print(f"\nEjecutando entrenamiento PARALELO con {n} núcleos\n")
        tiempo = medir_tiempo("paralelo", n)

        speedup = tiempo_base / tiempo
        eficiencia = (speedup / n) * 100

        resultados.append({
            "modo": "paralelo",
            "nucleos": n,
            "tiempo": round(tiempo, 4),
            "speedup": round(speedup, 2),
            "eficiencia": round(eficiencia, 2)
        })

    # 3️⃣ Guardar resultados en CSV
    with open(ARCHIVO_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["modo", "nucleos", "tiempo", "speedup", "eficiencia"]
        )
        writer.writeheader()
        writer.writerows(resultados)

    print("\n==============================================")
    print(" BENCHMARK FINALIZADO CORRECTAMENTE ")
    print(f" Resultados guardados en: {ARCHIVO_CSV}")
    print("==============================================\n")


if __name__ == "__main__":
    main()
