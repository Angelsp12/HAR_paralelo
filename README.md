![Python](https://img.shields.io/badge/python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)


# HAR_paralelo

## Reconocimiento de Actividades Humanas con Cómputo Paralelo

Este proyecto implementa un sistema de **Reconocimiento de Actividades Humanas (HAR)** utilizando datos del **acelerómetro y giroscopio** de teléfonos y relojes inteligentes.  
El objetivo principal es **comparar el rendimiento entre una ejecución secuencial y una paralela** durante el procesamiento y entrenamiento de modelos de Machine Learning.

## Objetivos
- Analizar datos sensoriales del acelerómetro y giroscopio (teléfono y reloj).  
- Implementar técnicas de **extracción de características** mediante ventanas de tiempo.  
- Entrenar clasificadores para identificar diferentes actividades humanas.  
- Evaluar el impacto del **cómputo paralelo** en el tiempo de ejecución y desempeño del modelo.  

---

## Tecnologías Utilizadas
- **Python 3.x**
- **NumPy, Pandas** – manejo y procesamiento de datos.  
- **Scikit-learn** – entrenamiento y evaluación de modelos.  
- **Joblib / Multiprocessing** – paralelización del procesamiento y entrenamiento.  
- **Matplotlib** – visualización de resultados.

---

##  Estructura del Repositorio

HAR_paralelo/
│
├── analisis_accelerometer.ipynb
├── analisis_gyroscope.ipynb
├── analisis_watch_accelerometer.ipynb
├── analisis_watch_gyroscope.ipynb
├── entrenamiento_Phones.ipynb
├── Entrenamiento_Phones_Gyroscope.ipynb
├── Entrenamiento_phones_gyroscopes_secuencial.ipynb
│
├── entrenamiento.py # Ejemplo de entrenamiento paralelo con Joblib
├── requirements.txt # Librerías necesarias
│
├──  resultados/ 
│ ├── tiempos_entrenamiento.png
│ ├── comparacion_precision.png
│ └── matriz_confusion.png
│
└── README.md

## Ejecución
###  Configurar entorno
Instala las dependencias ejecutando:
```bash
pip install -r requirements.txt

## Modo Secuencial

Ejecuta el entrenamiento de forma tradicional:

python entrenamiento.py --modo secuencial

## Modo Paralelo

Ejecuta el entrenamiento distribuyendo las tareas entre varios núcleos:

python entrenamiento.py --modo paralelo --n_jobs 4

O ejecuta los notebooks directamente en Jupyter:

Entrenamiento_phones_gyroscopes_secuencial.ipynb

Entrenamiento_Phones_Gyroscope.ipynb


## 📊 Evaluación del Desempeño Computacional

### Tabla 1. Resultados del módulo de Accelerometer

| Núcleos | Tiempo Total (s) | Speedup | Eficiencia (%) |
|:--------:|:----------------:|:--------:|:----------------:|
| 1  | 132.32 | 1.00× | 100.00 |
| 2  | 74.56  | 1.77× | 88.73 |
| 4  | 44.14  | 3.00× | 74.94 |
| 8  | 41.07  | 3.22× | 40.27 |
| 12 | 41.50  | 3.19× | 26.57 |
| 16 | 43.10  | 3.07× | 19.19 |
| 20 | 42.97  | 3.08× | 15.40 |

---

### Tabla 2. Resultados del módulo de Gyroscope

| Núcleos | Tiempo Total (s) | Speedup | Eficiencia (%) |
|:--------:|:----------------:|:--------:|:----------------:|
| 1  | 145.46 | 1.00× | 100.00 |
| 2  | 80.99  | 1.80× | 89.80 |
| 4  | 53.78  | 2.70× | 67.62 |
| 8  | 47.94  | 3.03× | 37.93 |
| 12 | 46.40  | 3.14× | 26.13 |
| 16 | 43.95  | 3.31× | 20.69 |
| 20 | 43.72  | 3.33× | 16.63 |

---

### Comparación General

| Sensor | Speedup Máximo | Eficiencia Promedio (1–8 núcleos) | Reducción de Tiempo |
|:--------|:----------------:|:---------------------------------:|:--------------------:|
| **Accelerometer** | 3.22× (con 8 núcleos) | ~68 % | ↓ 69 % (132.3 s → 41.0 s) |
| **Gyroscope**      | 3.33× (con 20 núcleos) | ~70 % | ↓ 70 % (145.5 s → 43.7 s) |

📈 **Conclusión:**  
El rendimiento mejora notablemente al aplicar paralelismo, alcanzando aceleraciones de hasta **3.3×** con 8–20 núcleos.  
La eficiencia comienza a disminuir más allá de los 8 núcleos, lo que evidencia el impacto del *overhead* de coordinación entre procesos.  
En general, el tiempo total de procesamiento se redujo alrededor del **70 %** sin afectar el desempeño del modelo.


# Limitaciones

Aunque el uso de cómputo paralelo permitió reducir significativamente los tiempos de ejecución, el sistema presenta algunas limitaciones inherentes al enfoque empleado. En primer lugar, no todo el pipeline de procesamiento es paralelizable; existen etapas que deben ejecutarse de forma secuencial, lo cual limita la aceleración total alcanzable. Este comportamiento es consistente con la **Ley de Amdahl**, que establece que la mejora en el rendimiento de un sistema paralelo está acotada por la fracción secuencial del proceso.

Asimismo, al incrementar el número de núcleos, la eficiencia disminuye debido al *overhead* asociado a la creación, sincronización y comunicación entre procesos. Este efecto se vuelve más evidente a partir de cierto número de núcleos, donde el costo de coordinación supera los beneficios del paralelismo.

Finalmente, la implementación actual se basa exclusivamente en paralelismo a nivel de CPU, sin aprovechar aceleradores de hardware como GPUs, lo cual podría limitar el desempeño en escenarios de mayor complejidad.

#Trabajo Futuro

Como trabajo futuro, se propone integrar el entrenamiento real de los modelos directamente en el módulo de benchmark, sustituyendo las simulaciones actuales por la ejecución completa del pipeline de Machine Learning. Además, sería relevante evaluar el uso de **aceleración por GPU** para comparar su desempeño frente al cómputo paralelo en CPU.

Otra línea de mejora consiste en explorar frameworks de computación distribuida como **Apache Spark**, **Dask** o **Ray**, que permitirían escalar el procesamiento a múltiples nodos y analizar el impacto del paralelismo a nivel de clúster. Asimismo, se plantea comparar diferentes estrategias de paralelización y modelos de aprendizaje más complejos, como redes neuronales profundas, para evaluar su escalabilidad y eficiencia computacional.


![alt text](image.png) #descripciones de cada Notebook
##Conclusiones

El procesamiento paralelo mejora significativamente la eficiencia del entrenamiento, reduciendo los tiempos de cómputo sin comprometer el rendimiento del modelo.

La carga de trabajo se distribuye de manera efectiva entre núcleos del procesador aprovechando los recursos disponibles.

El enfoque paralelo permite escalar el procesamiento a conjuntos de datos más grandes y modelos más complejos.

Este proyecto demuestra el potencial del cómputo paralelo aplicado al Machine Learning y la Ciencia de Datos.

Autores
Ángel Miguel Sánchez Pérez, Samuel Soriano Chavez, Sergio de Jesus Castillo Molano
Instituto Politécnico Nacional (IPN)
Unidad Profesional Interdisciplinaria de Ingeniería campus Tlaxcala (UPIIT)