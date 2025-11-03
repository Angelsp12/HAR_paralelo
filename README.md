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