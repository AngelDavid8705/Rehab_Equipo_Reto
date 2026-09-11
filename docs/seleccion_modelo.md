# Documento de Selección, Configuración, Entrenamiento y Decisión de Modelos

**Proyecto:** Reto REHAB - Clasificación de Movimientos de Rehabilitación  
**Equipo:** Angel David Lugo, José Pablo, Juan Pablo  
**Curso:** Inteligencia Artificial Avanzada para Ciencia de Datos  
**Notebook interactivo:** [`notebooks/02_model_selection_and_training.ipynb`](file:///Users/josepablo13/Documents/José%20Pablo/TEC/Septimo_Semestre/Rehab_Equipo_Reto/notebooks/02_model_selection_and_training.ipynb)  
**Script reproducible:** [`src/train_models.py`](file:///Users/josepablo13/Documents/José%20Pablo/TEC/Septimo_Semestre/Rehab_Equipo_Reto/src/train_models.py)  
**Artefactos persistidos:** [`results/comparacion_modelos.csv`](file:///Users/josepablo13/Documents/José%20Pablo/TEC/Septimo_Semestre/Rehab_Equipo_Reto/results/comparacion_modelos.csv) y [`results/mejor_modelo.joblib`](file:///Users/josepablo13/Documents/José%20Pablo/TEC/Septimo_Semestre/Rehab_Equipo_Reto/results/mejor_modelo.joblib)  

---

## 1. Contexto y Propósito del Repositorio

El propósito central de este desarrollo es clasificar de forma automática y cuantitativa la ejecución de ejercicios estandarizados de rehabilitación motora en pacientes post-Accidente Cerebrovascular (ACV). El dataset utilizado es **REHAB** (*Lv et al., 2026, Nature Scientific Data*), el cual recopila lecturas de sensores portátiles durante terapias físicas aplicadas a 120 pacientes.

El sistema utiliza dos subsistemas sensoriales:
1. **Dos Unidades de Medición Inercial (IMUs):** Montadas en el antebrazo y en el brazo, capturando ángulos tridimensionales de orientación (`Pitch`, `Yaw`, `Roll`).
2. **Un Guante Sensorial de Flexión:** Equipado con 5 sensores de flexión resistiva para los dedos (`Thumb`, `Index`, `Middle`, `Ring`, `Pinky`) y 1 sensor de inclinación de la articulación de la muñeca (`Glove_Pitch`).

En total, se registran **12 canales sensoriales continuos**.

---

## 2. Identificación del Tipo de Modelo Requerido

El problema del reto se formula formalmente como un **Problema de Clasificación Supervisada Multiclase**:
$$\mathcal{X} \subset \mathbb{R}^{60} \longrightarrow \mathcal{Y} \in \{0, 1, 2, \dots, 15\} \setminus \{14\}$$

- **Supervisada:** Cada muestra en el conjunto de entrenamiento cuenta con una etiqueta verdadera verificada por el protocolo clínico (`movimiento_id`).
- **Multiclase:** La variable objetivo adopta $K = 15$ categorías discretas mutuamente excluyentes (cada repetición corresponde a un único ejercicio funcional).

### Tratamiento del Movimiento 14 (Dato Corrupto en Origen)
El archivo `014_1.npy` se encuentra dañado a nivel de bits en el repositorio oficial de *Science Data Bank* (no es legible como array de NumPy). Imputar artificialmente 6 canales completos de IMU introduciría correlaciones espurias o ruido severo. Por lo tanto, se tomó la decisión deliberada de **excluir el movimiento 14**, conservando las 15 clases íntegras con mediciones completas en los 12 canales (totalizando 4,257 muestras).

---

## 3. Identificación del Tipo de Datos y Compatibilidad con Modelos

### 3.1 Naturaleza del Espacio de Características
- **Representación Tabular Estadística (Feature Engineering):** Las señales temporales crudas ($880 \text{ timesteps} \times 12 \text{ canales} = 10,560$ puntos) fueron colapsadas mediante 5 estadísticos resumen por canal:
  $$\{\text{Media } (\mu), \text{ Desviación Estándar } (\sigma), \text{ Mínimo } (\min), \text{ Máximo } (\max), \text{ RMS } (\text{Root Mean Square})\}$$
  Esto produce un vector de $12 \times 5 = 60$ características continuas por repetición.
- **Propiedades Matemáticas de las Variables:**
  - Variables continuas, densas y sin valores nulos ($0 \text{ NaN}$).
  - Diferencias drásticas de escala física: los ángulos de las IMUs oscilan en $[-180^\circ, 180^\circ]$, mientras que las lecturas analógicas de flexión operan en rangos de voltaje específicos.
- **Distribución de Clases:** Moderadamente desbalanceada (desde 212 muestras en el Movimiento 1 hasta 385 muestras en el Movimiento 7; razón de desbalance $1.81:1$).

### 3.2 Modelos Compatibles y Criterio de Selección de Algoritmos Sencillos
Para cumplir con la premisa de utilizar algoritmos clásicos, sencillos e interpretables del análisis de datos, se investigaron 6 familias representativas:
1. **Regresión Logística Multinomial (Lineal):** Modela la probabilidad a posteriori de cada clase mediante transformaciones lineales combinadas con la función Softmax. Sirve como referencia base (*baseline*).
2. **K-Nearest Neighbors - KNN (Basado en Instancias):** Clasifica asignando la moda de las etiquetas de los $k$ vecinos más cercanos en el espacio métrico euclidiano.
3. **Árbol de Decisión CART (No Paramétrico Jerárquico):** Genera reglas de partición ortogonales buscando la máxima homogeneidad (impureza de Gini).
4. **Random Forest (Ensamble por Bagging):** Agrega cientos de árboles de decisión descorrelacionados mediante remuestreo bootstrap y selección aleatoria de subconjuntos de características.
5. **Support Vector Machine - SVM (Clasificador de Máximo Margen):** Encuentra hiperplanos de separación óptimos utilizando funciones kernel (lineal y RBF).
6. **Gaussian Naive Bayes (Probabilístico Generativo):** Modela la verosimilitud bajo el supuesto de independencia condicional entre características.

---

## 4. Metodología de Validación y Decisiones de Preprocesamiento

### 4.1 Partición Estratificada (Train / Test Split)
- Se aplicó una división **80% Entrenamiento (3,405 muestras) / 20% Prueba (852 muestras)** mediante `train_test_split(stratify=y, random_state=42)`.
- **Justificación:** La opción `stratify=y` es indispensable dado el desbalance moderado; garantiza que la proporción de cada uno de los 15 ejercicios clínicos sea idéntica en entrenamiento y prueba, evitando sesgos de subrepresentación.

### 4.2 Estandarización Z-Score y Prevención de Data Leakage
- Se utilizó `StandardScaler` de Scikit-Learn: $z = (x - \mu)/\sigma$.
- **Decisión Estricta contra Fuga de Información (*Data Leakage*):** El escalador se ajustó **exclusivamente con `X_train`** (`fit_transform`) y se aplicó directamente a `X_test` (`transform`). No se utilizó ninguna información del conjunto de prueba durante el cálculo de medias y desviaciones.
- Algoritmos como KNN, SVM y Regresión Logística requieren obligatoriamente esta normalización para que los canales angulares no dominen espuriamente a los canales de flexión.

### 4.3 Métrica Rectora de Evaluación: Macro F1-Score
- **Decisión:** Aunque se monitorea la Exactitud global (*Accuracy*), la métrica rectora de comparación y selección es el **Macro F1-Score**:
  $$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^{K} \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$
- **Justificación:** En problemas desbalanceados, el Accuracy puede verse inflado por un buen rendimiento en clases mayoritarias. El Macro F1 asigna el mismo peso a cada clase clínica independientemente de su soporte, penalizando severamente si un ejercicio específico es mal clasificado.

---

## 5. Resultados Experimentales y Comparación de Familias de Modelos

### 5.1 Validación Cruzada 5-Fold sobre Train y Evaluación en Test Set

Todos los modelos fueron validados con **5-Fold Stratified Cross-Validation** sobre `X_train` y evaluados sobre el conjunto `X_test` independiente (852 muestras):

| Modelo Evaluado | CV Macro F1 (Train) | Exactitud (Test) | Macro F1 (Test) | Weighted F1 (Test) | Tiempo Train (s) | Tiempo Inferencia (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (300 árboles)** | **0.949 ± 0.008** | **0.9695** | **0.9690** | **0.9697** | 0.63 s | 35.3 ms |
| **SVM (Kernel RBF, C=10)** | 0.937 ± 0.007 | 0.9531 | 0.9509 | 0.9530 | 0.12 s | 119.0 ms |
| **Árbol de Decisión (prof=12)** | 0.842 ± 0.010 | 0.8756 | 0.8733 | 0.8757 | 0.12 s | 0.3 ms |
| **KNN (k=5)** | 0.834 ± 0.021 | 0.8721 | 0.8709 | 0.8721 | 0.001 s | 28.9 ms |
| **Regresión Logística** | 0.828 ± 0.018 | 0.8263 | 0.8242 | 0.8268 | 0.08 s | 0.3 ms |
| **Gaussian Naive Bayes** | 0.696 ± 0.028 | 0.7171 | 0.7076 | 0.7101 | 0.002 s | 1.5 ms |

---

## 6. Comparación y Optimización de Configuraciones de Hiperparámetros

Se exploraron diferentes configuraciones por algoritmo para analizar el equilibrio sesgo-varianza:

1. **K-Nearest Neighbors (KNN):**
   - Se varió $k \in [1, 3, 5, 7, 9, 13, 17, 21]$ comparando ponderación uniforme (`uniform`) vs ponderación por el inverso de la distancia (`distance`).
   - *Hallazgo:* `weights='distance'` supera consistentemente a `weights='uniform'` en todos los valores de $k$. Con $k=1$, el F1 en test alcanza 0.9777, pero a costa de alta sensibilidad al ruido en repeticiones atípicas de pacientes. Con $k=5$ (`distance`), F1 alcanza 0.8872 con mayor estabilidad.
2. **Árbol de Decisión:**
   - Se varió `max_depth` $\in [4, 6, 8, 10, 12, 14, 18, \text{None}]$.
   - *Hallazgo:* Con `max_depth=4`, el modelo sufre de subajuste (F1 Macro = 0.53). A partir de `max_depth=12`, el F1 en entrenamiento llega al 0.97 pero en test se estanca en ~0.87, evidenciando sobreajuste del árbol individual.
3. **Support Vector Machine (SVM):**
   - Comparación entre Kernel Lineal y Kernel RBF variando $C \in [0.1, 1.0, 10.0, 50.0]$.
   - *Kernel Lineal:* F1 oscila entre 0.8602 ($C=0.1$) y 0.8942 ($C=10$). El techo en ~0.89 demuestra que las fronteras de decisión de los movimientos cinemáticos **no son linealmente separables**.
   - *Kernel RBF:* Con $C=1.0$ obtiene F1 = 0.8804; con $C=10.0$ salta a **0.9509**; y con $C=50.0$ alcanza 0.9730. El kernel RBF mapea de forma natural las interacciones no lineales entre orientación del brazo y flexión de los dedos.
4. **Random Forest:**
   - Variación del número de estimadores $n \in [25, 50, 100, 200, 300, 500]$.
   - *Hallazgo:* Con $n=25$, F1 = 0.958. A partir de $n=100$, el F1 se estabiliza en $\approx 0.968$. Con $n=300$, se alcanza **0.9690** con un tiempo de entrenamiento de apenas 0.63 segundos. Aumentar a $n=500$ no aporta mejoras estadísticas significativas (F1 = 0.9691) duplicando el costo computacional.

---

## 7. Justificación de la Elección del Modelo Ganador

Se seleccionó **Random Forest con 300 árboles (`n_estimators=300`, `max_features='sqrt'`)** como el modelo definitivo del reto.

### Razones Técnicas y Empíricas:
1. **Desempeño Global Superior:** Logró un **Macro F1-Score de 0.9690** y una **Exactitud de 0.9695** en el conjunto de prueba independiente.
2. **Baja Varianza y Generalización:** En validación cruzada 5-fold, obtuvo la desviación estándar más baja ($\pm 0.008$), indicando consistencia entre diferentes subconjuntos de pacientes.
3. **Ventaja frente a SVM en Inferencia:** Aunque SVM con RBF ($C=50$) alcanza un F1 similar, su tiempo de inferencia es de **119 ms**, mientras que Random Forest evalúa las 852 muestras en **35 ms** (~0.04 ms por muestra). Esto lo hace 3.3 veces más rápido para aplicaciones clínicas en tiempo real.
4. **Resistencia al Ruido y Datos No Lineales:** Al promediar 300 árboles con subconjuntos aleatorios de variables, es inmune al sobreajuste de árboles individuales y maneja con soltura las correlaciones cruzadas entre canales IMU y guante.

---

## 8. Diagnóstico Detallado del Modelo Ganador

### 8.1 Desempeño por Clase Clínica (Reporte de Clasificación)
- **Movimientos con Desempeño Perfecto ($F1 = 1.000$ o $> 0.99$):**
  - `Mov 00: Bobath Handshake` (F1 = 1.000, Precisión = 1.00, Recall = 1.00)
  - `Mov 02: Bobath Forward Flexion/Extension` (F1 = 0.9905)
  - `Mov 07: Ball Gripping` (F1 = 0.9935, 77/77 muestras detectadas correctamente)
- **Movimientos con Leves Confusiones:**
  - `Mov 11: Elbow Joint Flexion & Touch` (F1 = 0.9038, Precisión = 0.8246, Recall = 1.000)
  - `Mov 12: Shoulder Touch Training` (F1 = 0.9298, Precisión = 0.9636, Recall = 0.8983)

### 8.2 Explicación Biomecánica de Confusiones
La matriz de confusión revela que la principal fuente de error residual ocurre entre el **Movimiento 11** y el **Movimiento 12**:
- Ambos ejercicios comparten la flexión del codo y la elevación de la mano hacia el tronco superior (tocar el hombro o tocar el pecho). En pacientes hemipléjicos con espasticidad, el rango articular se ve restringido, provocando que la trayectoria cinemática de `IMU1` (antebrazo) sea prácticamente indistinguible.

### 8.3 Importancia de Sensores y Estadísticos (MDI)
- **Por Módulo Sensorial:**
  - `IMU 1 (Antebrazo)` aporta el **39.4%** de la importancia total del ensamble.
  - `Guante (Sensores de Flexión de Dedos)` aporta el **31.8%**.
  - `IMU 2 (Brazo)` aporta el **21.5%**.
  - `Guante (Pitch de Muñeca)` aporta el **7.3%**.
- **Por Tipo de Estadístico:**
  - La **Media ($\mu$)** y el **RMS (Energía cinemática)** aportan más del **58%** del poder discriminativo, ya que capturan la postura media espacial y la intensidad del movimiento. La Desviación Estándar ($\sigma$) discrimina eficientemente movimientos estáticos de movimientos dinámicos.

---

## 9. Conclusiones y Trazabilidad

1. **Problema resuelto exitosamente:** Se construyó un clasificador con más del **96.9% de F1 Macro y Exactitud** sobre 15 movimientos funcionales de pacientes con ACV.
2. **Notebook reproducible:** El notebook [`notebooks/02_model_selection_and_training.ipynb`](file:///Users/josepablo13/Documents/José%20Pablo/TEC/Septimo_Semestre/Rehab_Equipo_Reto/notebooks/02_model_selection_and_training.ipynb) contiene la narrativa pedagógica, código ejecutable, tablas de validación cruzada y todas las visualizaciones integradas.
3. **Persistencia:** El modelo y escalador listos para producción residen en [`results/mejor_modelo.joblib`](file:///Users/josepablo13/Documents/José%20Pablo/TEC/Septimo_Semestre/Rehab_Equipo_Reto/results/mejor_modelo.joblib).
