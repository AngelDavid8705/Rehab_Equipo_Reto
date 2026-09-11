# Seleccion, configuracion y entrenamiento del modelo

Script: `src/train_models.py`. Resultados: `results/comparacion_modelos.csv`,
`results/mejor_modelo.joblib`.

## 1. Tipo de modelo requerido
El problema es clasificacion supervisada multiclase: cada muestra tiene una etiqueta
conocida y el objetivo es predecir esa etiqueta a partir de las señales del sensor.

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

| Modelo | Exactitud (validation) | F1 macro (validation) |
|---|---|---|
| Random Forest (300 arboles) | 0.9593 | 0.9578 |
| SVM (kernel RBF, C=10) | 0.9577 | 0.9560 |
| Arbol de Decision (profundidad 12) | 0.8920 | 0.8888 |
| KNN (k=5) | 0.8748 | 0.8737 |
| Regresion Logistica | 0.8341 | 0.8296 |

Se uso F1 macro ademas de exactitud porque las clases estan desbalanceadas y F1 macro
no le da mas peso a las clases con mas muestras.

El arbol de decision usa entropia (la misma medida de Shannon) como criterio para elegir en que variable dividir cada
nodo, en cada paso busca la particion que mas reduce la entropia de las clases dentro
de esa rama.

## 4. Modelo elegido

**Random Forest** (`n_estimators=300`, resto de hiperparametros por defecto de
scikit-learn). Random Forest y SVM quedaron practicamente empatados (95.93% vs 95.77%
de exactitud), asi que la diferencia entre ambos no es grande. Se eligio Random Forest
por tener el mejor resultado en ambas metricas, y porque al ser un ensamble de arboles
es menos propenso a sobreajuste que un solo arbol de decision y no requiere que las
features tengan una relacion lineal con la clase. SVM entreno mas rapido (0.24s vs
4.97s), pero el tiempo de entrenamiento no fue el criterio principal de seleccion.

## 5. Configuracion y entrenamiento

- Split estratificado en 3 partes, 70% train / 15% validation / 15% test
  (`train_test_split` en dos pasos, `stratify=y`, `random_state=42`), para mantener la
  proporcion de clases en los tres conjuntos.
- Estandarizacion de features (`StandardScaler`) ajustada solo con train, aplicada a
  train, validation y test.
- Los 5 modelos se entrenan con train y se comparan entre si usando validation,
esa comparacion es la que decide cual modelo es el mejor.
- El modelo elegido (Random Forest) se evalua una sola vez, al final, sobre test con 0.966 de exactitud y 0.966 de F1 macro.
  Ese numero es la estimacion final del desempeno del modelo.
- El modelo entrenado y el scaler se guardan juntos en `results/mejor_modelo.joblib`
  para poder reusarlos sin reentrenar.

  
