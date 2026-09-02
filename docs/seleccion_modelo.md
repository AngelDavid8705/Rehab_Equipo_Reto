# Seleccion, configuracion y entrenamiento del modelo

Script: `src/train_models.py`. Resultados: `results/comparacion_modelos.csv`,
`results/mejor_modelo.joblib`.

## 1. Tipo de modelo requerido
El problema es clasificacion supervisada multiclase: cada muestra tiene una etiqueta
conocida y el objetivo es predecir esa etiqueta a partir de las senales del sensor.

## 2. Tipo de datos y modelos compatibles
Se usa el dataset redefinido (`data/rehab_features_dataset.csv`) 4,257 muestras, cada
una con 60 caracteristicas numericas continuas (media, desviacion estandar, minimo,
maximo y RMS de los 12 canales de sensor) mas la etiqueta de clase. Es un dataset
tabular, numerico, sin valores faltantes, con clases moderadamente desbalanceadas
(212 a 385 muestras por clase).

Con datos de ese tipo son compatibles la mayoria de los algoritmos clasicos de
clasificacion supervisada: regresion logistica, K-Nearest Neighbors, arboles de
decision, ensambles de arboles (Random Forest) y maquinas de soporte vectorial (SVM).

## 3. Modelos investigados
Se entrenaron y compararon 5 modelos con configuracion base (ver `src/train_models.py`):

| Modelo | Exactitud (test) | F1 macro (test) |
|---|---|---|
| Random Forest (300 arboles) | 0.970 | 0.969 |
| SVM (kernel RBF, C=10) | 0.953 | 0.951 |
| Arbol de Decision (profundidad 12) | 0.876 | 0.873 |
| KNN (k=5) | 0.872 | 0.871 |
| Regresion Logistica | 0.826 | 0.824 |

Se uso F1 macro ademas de exactitud porque las clases estan desbalanceadas y F1 macro
no le da mas peso a las clases con mas muestras.

## 4. Modelo elegido
**Random Forest** (`n_estimators=300`, resto de hiperparametros por defecto de
scikit-learn). Se eligio porque obtuvo el mejor resultado en ambas metricas, es
relativamente rapido de entrenar (0.7s con este dataset) y, al ser un ensamble de
arboles, es menos propenso a sobreajustarse que un solo arbol de decision, ademas de que
no requiere que las features tengan una relacion lineal con la clase.

## 5. Configuracion y entrenamiento

- Split estratificado 80% train / 20% test (`train_test_split`, `stratify=y`,
  `random_state=42`), para mantener la proporcion de clases en ambos conjuntos.
- Estandarizacion de features (`StandardScaler`) ajustada solo con train, aplicada a
  train y test.
- El modelo final se entrena sobre las 3,405 muestras de train y se evalua una sola
  vez sobre las 852 de test.
- El modelo entrenado y el scaler se guardan juntos en
  `results/mejor_modelo.joblib` para poder reusarlos sin reentrenar.
