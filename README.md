# Reto Parcial 1 - Clasificación de Movimientos de Rehabilitación (REHAB)

**Equipo:** Angel Lugo, Jose Pablo, Juan Pablo  
**Curso:** IA Avanzada para Ciencia de Datos  
**Rama activa:** `ramaJP`  

---

## 📌 Contexto del Reto

El reto utiliza el dataset **REHAB**: señales cinemáticas de sensores portátiles (2 IMUs en antebrazo y brazo + guante de flexión con 5 sensores digitales y pitch de muñeca) recolectadas de **120 pacientes post-ACV** durante un programa de rehabilitación de dos semanas.
Trabajamos únicamente con la sección de **Rehab_exercise** (16 movimientos estandarizados de entrenamiento de rehabilitación).

- **Paper original:** Lv et al. (2026), *A wearable sensor-based kinematic dataset collected under standardized rehabilitation tasks from 120 post-stroke patients*, Scientific Data 13:1136. https://doi.org/10.1038/s41597-026-07802-2
- **Dataset de origen:** Science Data Bank, https://doi.org/10.57760/sciencedb.37018

**Problemática a resolver:** Dada la información cinemática multicanal (12 canales continuos), clasificar cuál de los ejercicios de rehabilitación se está ejecutando (**Clasificación Supervisada Multiclase** sobre 15 clases activas, $N = 4,257$ muestras tabulares).

> ⚠️ **Tratamiento de dato corrupto:** El archivo `014_1.npy` está corrupto en el propio repositorio original de Science Data Bank, por lo que se excluyó el movimiento 14 para evitar introducciones espurias mediante imputación artificial.

---

## 🏆 Resumen de Resultados de Selección y Entrenamiento de Modelos

Se entrenaron y compararon 6 familias de algoritmos clásicos de Machine Learning sobre 60 descriptores tabulares (media, std, min, max, RMS por canal) con división estratificada 80/20 y estandarización `StandardScaler` (sin *Data Leakage*):

| Modelo | Exactitud (Test) | F1 Macro (Test) | F1 Weighted (Test) | Tiempo Train (s) | Tiempo Inferencia |
|---|:---:|:---:|:---:|:---:|:---:|
| 🥇 **Random Forest (300 árboles)** | **0.9695** | **0.9690** | **0.9697** | **0.63 s** | **35.3 ms** |
| 🥈 **SVM (Kernel RBF, C=10)** | 0.9531 | 0.9509 | 0.9530 | 0.12 s | 119.0 ms |
| 🥉 **Árbol de Decisión (prof=12)** | 0.8756 | 0.8733 | 0.8757 | 0.12 s | 0.3 ms |
| **KNN (k=5, distance)** | 0.8721 | 0.8709 | 0.8721 | 0.001 s | 28.9 ms |
| **Regresión Logística** | 0.8263 | 0.8242 | 0.8268 | 0.08 s | 0.3 ms |
| **Gaussian Naive Bayes** | 0.7171 | 0.7076 | 0.7101 | 0.002 s | 1.5 ms |

**Modelo Seleccionado:** **Random Forest** ($n=300$). Destaca por lograr el mayor F1 Macro (0.9690), mínima varianza en validación cruzada 5-fold ($\pm 0.008$) y ser 3.3 veces más rápido en inferencia que SVM RBF.

---

## 🗂️ Dónde Encontrar los Cambios e Implementación

Toda la etapa de **Selección, Configuración, Entrenamiento y Evaluación del Modelo** se encuentra estructurada y modularizada en los siguientes componentes:

### 1. 📓 Notebook Principal de Selección y Entrenamiento
- **[`notebooks/02_model_selection_and_training.ipynb`](file:///Users/josepablo13/Documents/José%20Pablo/TEC/Septimo_Semestre/Rehab_Equipo_Reto/notebooks/02_model_selection_and_training.ipynb):** Notebook completo y auto-contenido en Jupyter.
  - Carga y preprocesamiento de características.
  - Validación cruzada estratificada de 5 pliegues (*5-Fold Stratified CV*).
  - Experimentos de variación de hiperparámetros (KNN, Árboles, SVM, Random Forest).
  - Reporte de clasificación detallado por clase, Matriz de Confusión Normalizada e Importancia de Variables (*Feature Importances MDI* por sensor y por estadístico).

### 2. 📄 Documentación Técnica de Decisiones
- **[`docs/seleccion_modelo.md`](file:///Users/josepablo13/Documents/José%20Pablo/TEC/Septimo_Semestre/Rehab_Equipo_Reto/docs/seleccion_modelo.md):** Documento técnico exhaustivo que responde detalladamente a los requerimientos de selección de modelo, tipo de datos, métrica rectora (Macro F1), comparación de hiperparámetros y justificación biomecánica sin ambigüedades.

### 3. 🧩 Módulos de Modelos Clásicos
- **`modelos/`:** Directorio modular conteniendo las definiciones de cada algoritmo:
  - [`modelos/random_forest.py`](file:///Users/josepablo13/Documents/José%20Pablo/TEC/Septimo_Semestre/Rehab_Equipo_Reto/modelos/random_forest.py)
  - [`modelos/svm.py`](file:///Users/josepablo13/Documents/José%20Pablo/TEC/Septimo_Semestre/Rehab_Equipo_Reto/modelos/svm.py)
  - [`modelos/decision_tree.py`](file:///Users/josepablo13/Documents/José%20Pablo/TEC/Septimo_Semestre/Rehab_Equipo_Reto/modelos/decision_tree.py)
  - [`modelos/knn.py`](file:///Users/josepablo13/Documents/José%20Pablo/TEC/Septimo_Semestre/Rehab_Equipo_Reto/modelos/knn.py)
  - [`modelos/logistic_regression.py`](file:///Users/josepablo13/Documents/José%20Pablo/TEC/Septimo_Semestre/Rehab_Equipo_Reto/modelos/logistic_regression.py)

### 4. ⚙️ Script Reproducible de Entrenamiento
- **[`src/train_models.py`](file:///Users/josepablo13/Documents/José%20Pablo/TEC/Septimo_Semestre/Rehab_Equipo_Reto/src/train_models.py):** Script en Python para automatizar el entrenamiento, evaluación y exportación de resultados.
  ```bash
  python src/train_models.py --csv data/rehab_features_dataset.csv
  ```

### 5. 📦 Artefactos y Resultados Exportados
- **[`results/comparacion_modelos.csv`](file:///Users/josepablo13/Documents/José%20Pablo/TEC/Septimo_Semestre/Rehab_Equipo_Reto/results/comparacion_modelos.csv):** Tabla comparativa de métricas exportada en CSV.
- **[`results/mejor_modelo.joblib`](file:///Users/josepablo13/Documents/José%20Pablo/TEC/Septimo_Semestre/Rehab_Equipo_Reto/results/mejor_modelo.joblib):** Objeto serializado con el mejor modelo (`Random Forest`), el `StandardScaler` ajustado y las etiquetas clínicas para inferencia directa.

---

## 📂 Estructura General del Proyecto

```text
Rehab_Equipo_Reto/
├── data/
│   ├── raw/                         # Archivos .npy originales
│   └── rehab_features_dataset.csv   # Dataset redefinido (60 features x 4257 muestras)
├── docs/
│   ├── reportes/                    # Reporte de EDA previo (PDF)
│   └── seleccion_modelo.md          # Documentación detallada de decisiones y modelos
├── modelos/
│   ├── __init__.py
│   ├── decision_tree.py             # Árbol de Decisión
│   ├── knn.py                       # K-Nearest Neighbors
│   ├── logistic_regression.py       # Regresión Logística
│   ├── random_forest.py             # Random Forest
│   └── svm.py                       # Support Vector Machine (Kernel RBF)
├── notebooks/
│   ├── 01_data_loading_eda.ipynb    # Carga de datos y EDA sobre señales crudas
│   └── 02_model_selection_and_training.ipynb  # Notebook principal de selección y modelado
├── results/
│   ├── comparacion_modelos.csv      # Tabla de resultados comparativos
│   └── mejor_modelo.joblib          # Modelo final y scaler serializados
├── src/
│   ├── data_loader.py               # Carga y consolidación de datos .npy
│   ├── features.py                  # Extracción de características tabulares
│   └── train_models.py              # Script principal de entrenamiento
├── README.md                        # Descripción y guía del proyecto
└── requirements.txt                 # Dependencias del proyecto
```
