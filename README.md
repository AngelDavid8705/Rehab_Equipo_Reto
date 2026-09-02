# Reto Parcial 1 - Clasificacion de movimientos de rehabilitacion (REHAB)

**Equipo:** Angel Lugo, Jose Pablo, Juan Pablo
**Curso:** IA Avanzada para Ciencia de Datos

## Contexto del reto

El reto usa el dataset **REHAB**: señales de sensores portatiles (IMUs + guante de flexion)
recolectadas de 120 pacientes post-ACV durante un programa de rehabilitacion de dos semanas.
Trabajamos unicamente con la parte de **Rehab_exercise** (16 movimientos de entrenamiento de rehabilitacion) y no con la parte de evaluacion clinica FMA de 27 items.

- Paper: Lv et al. (2026), *A wearable sensor-based kinematic dataset collected under
  standardized rehabilitation tasks from 120 post-stroke patients*, Scientific Data 13:1136.
  https://doi.org/10.1038/s41597-026-07802-2
- Dataset: Science Data Bank, https://doi.org/10.57760/sciencedb.37018

**Problematica a resolver:** dado un tramo de señal de sensores (880 puntos de tiempo x 12
canales), predecir cual de los 16 movimientos de rehabilitacion se esta ejecutando
(clasificacion multiclase).

## Dato corrupto

El archivo `014_1.npy`esta **corrupto en el propio repositorio de Science Data Bank**.

## Estructura

- `docs/reportes/` - reporte de EDA.
- `docs/seleccion_modelo.md` - decisiones sobre que modelo se eligio y por que.
- `notebooks/eda.ipynb` - EDA sobre el dataset completo (12 canales), retomando los
  hallazgos del reporte.
- `data/rehab_features_dataset.csv` - dataset redefinido: 60 features tabulares por
  muestra (media/std/min/max/rms de cada canal) mas la etiqueta de movimiento.
- `data/raw/` - aqui estan los `.npy` originales.
- `src/data_loader.py` - carga y consolida las señales crudas.
- `src/features.py` - extraccion de features tabulares.
- `src/train_models.py` - entrena y compara varios modelos de clasificacion.
- `results/` - tabla comparativa de modelos y el mejor modelo entrenado.
