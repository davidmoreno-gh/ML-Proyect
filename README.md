# Predicción de Diabetes Usando Modelos de Machine Learning

## Descripción del Proyecto
Este proyecto utiliza técnicas de machine learning para predecir la presencia de diabetes en pacientes. A través de un análisis de un conjunto de datos de registros médicos, se implementaron varios modelos de clasificación como K-Nearest Neighbors (KNN), Regresión Logística y Random Forest para determinar qué pacientes podrían tener diabetes.

El proyecto incluye la limpieza de datos, análisis exploratorio, creación de modelos de machine learning, ajuste de hiperparámetros y evaluación de modelos utilizando métricas estándar como la matriz de confusión, precisión, recall y F1-score. También se manejó el desbalance en las clases usando técnicas como SMOTE.

## Contenido del Proyecto
Este repositorio contiene los siguientes archivos y carpetas:

- `diabetes_prediction.ipynb`: El cuaderno Jupyter que contiene todo el análisis, desde la carga de datos hasta la evaluación de los modelos.
- `dia.csv`: El conjunto de datos utilizado en el proyecto.
- `requirements.txt`: Lista de librerías necesarias para ejecutar el proyecto.
- `README.md`: Este archivo con la descripción del proyecto.
  
## Conjunto de Datos
El dataset utilizado proviene de [fuente del dataset, si es aplicable] y contiene información médica de pacientes, tales como:

- **Edad**: Edad del paciente.
- **Género**: Sexo del paciente (masculino o femenino).
- **Hipertensión**: Indicador de si el paciente tiene antecedentes de hipertensión.
- **Nivel de glucosa en sangre**: Niveles de glucosa en ayunas.
- **Historial de fumar**: Categoría sobre si el paciente ha fumado.
- **IMC**: Índice de Masa Corporal (BMI).
- **Diabetes**: Variable objetivo que indica si el paciente tiene diabetes (0: No, 1: Sí).

El dataset original contiene 96,128 registros con 9 variables.

## Instalación y Requisitos
Para ejecutar este proyecto, necesitarás las siguientes librerías de Python:

```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
imblearn
```

Puedes instalar todas las dependencias usando el archivo requirements.txt:

