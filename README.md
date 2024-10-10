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
```pip install -r requirements.txt```

# Estructura del Proyecto

## Limpieza de Datos:

- Eliminación de duplicados y valores irrelevantes.
- Filtrado de categorías innecesarias.
- Verificación de valores nulos.

## Análisis Exploratorio de Datos (EDA):

- Estadísticas descriptivas de las variables.
- Mapas de calor para visualizar correlaciones entre las características.
- Análisis de la distribución de la variable objetivo (diabetes).
  
## Modelado:

- K-Nearest Neighbors (KNN): Implementación del modelo KNN con ajuste de hiperparámetros usando GridSearchCV.
- Regresión Logística: Modelo básico de clasificación.
- Random Forest: Utilización de este modelo junto con técnicas de balanceo de clases para mejorar la predicción.
  
## Evaluación del Modelo:

- Matriz de confusión.
- Métricas de rendimiento como precisión, recall, y F1-score.

## Técnicas de Balanceo de Clases:

- Implementación de SMOTE para mejorar la predicción de la clase minoritaria.


## Resultados

- El modelo Random Forest con ajuste de hiperparámetros resultó ser el mejor modelo en términos generales, alcanzando una precisión del 95% en el conjunto de prueba.
- Las variables más importantes para predecir la diabetes fueron los niveles de glucosa en sangre y el índice de masa corporal (BMI).

## Cómo Ejecutar el Proyecto

Clona este repositorio:
```git clone https://github.com/tu-usuario/diabetes-prediction.git```

Instala las dependencias:
```pip install -r requirements.txt```

Ejecuta el archivo diabetes_prediction.ipynb en un entorno de Jupyter Notebook para realizar el análisis completo.

## Próximos Pasos

- Mejorar el manejo del desbalance de clases: Probar otras técnicas de balanceo, como Undersampling o combinaciones de técnicas de oversampling.
- Explorar más modelos: Implementar otros modelos como XGBoost o SVM para comparar su rendimiento.
- Optimización adicional de hiperparámetros: Realizar una búsqueda más exhaustiva para optimizar aún más los modelos.

## Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir, por favor abre un issue o crea un pull request explicando las mejoras o problemas que has encontrado.

