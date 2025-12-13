"""
===========================================================
GENERACIÓN AUTOMÁTICA DE GRÁFICAS – BENCHMARK HAR
Lee el CSV del benchmark y genera gráficas de rendimiento
===========================================================
"""

import csv
import os
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURACIÓN
# =========================================================
CARPETA_RESULTADOS = "Resultados"
ARCHIVO_CSV = os.path.join(CARPETA_RESULTADOS, "benchmark_tiempos.csv")
CARPETA_GRAFICAS = os.path.join(CARPETA_RESULTADOS, "graficas")


# =========================================================
# CARGA DE DATOS
# =========================================================
def cargar_datos(csv_path):
    nucleos = []
    tiempos = []
    speedups = []
    eficiencias = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nucleos.append(int(row["nucleos"]))
            tiempos.append(float(row["tiempo"]))
            speedups.append(float(row["speedup"]))
            eficiencias.append(float(row["eficiencia"]))

    return nucleos, tiempos, speedups, eficiencias


# =========================================================
# GRÁFICAS
# =========================================================
def grafica_tiempo(nucleos, tiempos):
    plt.figure()
    plt.plot(nucleos, tiempos, marker="o")
    plt.xlabel("Número de núcleos")
    plt.ylabel("Tiempo total (s)")
    plt.title("Tiempo de ejecución vs Núcleos")
    plt.grid(True)
    plt.savefig(os.path.join(CARPETA_GRAFICAS, "tiempo_vs_nucleos.png"))
    plt.close()


def grafica_speedup(nucleos, speedups):
    plt.figure()
    plt.plot(nucleos, speedups, marker="o")
    plt.xlabel("Número de núcleos")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Núcleos")
    plt.grid(True)
    plt.savefig(os.path.join(CARPETA_GRAFICAS, "speedup_vs_nucleos.png"))
    plt.close()


def grafica_eficiencia(nucleos, eficiencias):
    plt.figure()
    plt.plot(nucleos, eficiencias, marker="o")
    plt.xlabel("Número de núcleos")
    plt.ylabel("Eficiencia (%)")
    plt.title("Eficiencia vs Núcleos")
    plt.grid(True)
    plt.savefig(os.path.join(CARPETA_GRAFICAS, "eficiencia_vs_nucleos.png"))
    plt.close()


# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================
def main():

    if not os.path.exists(ARCHIVO_CSV):
        raise FileNotFoundError(
            f"No se encontró el archivo {ARCHIVO_CSV}. Ejecuta primero el benchmark."
        )

    os.makedirs(CARPETA_GRAFICAS, exist_ok=True)

    nucleos, tiempos, speedups, eficiencias = cargar_datos(ARCHIVO_CSV)

    grafica_tiempo(nucleos, tiempos)
    grafica_speedup(nucleos, speedups)
    grafica_eficiencia(nucleos, eficiencias)

    print("==============================================")
    print(" Gráficas generadas correctamente ")
    print(f" Ubicación: {CARPETA_GRAFICAS}")
    print("==============================================")


if __name__ == "__main__":
    main()
