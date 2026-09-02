"""
Seleccion, configuracion y entrenamiento del modelo - Reto REHAB (equipo)

Entrena y compara varios modelos clasicos de clasificacion sobre el dataset
redefinido (data/rehab_features_dataset.csv: 60 features tabulares por
muestra, 16 clases de movimiento). Guarda una tabla comparativa en
results/comparacion_modelos.csv y el mejor modelo entrenado en
results/mejor_modelo.joblib.

Uso:
    python src/train_models.py --csv data/rehab_features_dataset.csv
"""

import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from modelos import decision_tree, knn, logistic_regression, random_forest, svm

RANDOM_STATE = 42

MODELOS = {
    "Regresion Logistica": logistic_regression.get_model(),
    "KNN (k=5)": knn.get_model(),
    "Arbol de Decision": decision_tree.get_model(),
    "Random Forest": random_forest.get_model(),
    "SVM (RBF)": svm.get_model(),
}


def cargar_dataset(csv_path):
    df = pd.read_csv(csv_path)
    y = df["movimiento_id"].values
    X = df.drop(columns=["movimiento_id", "movimiento_nombre"]).values
    return X, y, df


def entrenar_y_comparar(X_train, y_train, X_test, y_test):
    resultados = []
    modelos_entrenados = {}

    for nombre, modelo in MODELOS.items():
        inicio = time.time()
        modelo.fit(X_train, y_train)
        duracion = time.time() - inicio

        pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1_macro = f1_score(y_test, pred, average="macro")

        resultados.append({
            "modelo": nombre,
            "exactitud_test": round(acc, 4),
            "f1_macro_test": round(f1_macro, 4),
            "tiempo_entrenamiento_s": round(duracion, 2),
        })
        modelos_entrenados[nombre] = modelo

        print(f"{nombre:22s}  exactitud={acc:.3f}  f1_macro={f1_macro:.3f}  ({duracion:.1f}s)")

    return pd.DataFrame(resultados).sort_values("f1_macro_test", ascending=False), modelos_entrenados


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/rehab_features_dataset.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    print("cargando dataset...")
    X, y, df = cargar_dataset(args.csv)
    print(f"X: {X.shape}   y: {y.shape}   clases: {len(np.unique(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\ntrain: {X_train.shape[0]} muestras   test: {X_test.shape[0]} muestras")
    print("\nentrenando y comparando modelos...\n")

    tabla, modelos = entrenar_y_comparar(X_train, y_train, X_test, y_test)

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    tabla_path = os.path.join(args.out_dir, "comparacion_modelos.csv")
    tabla.to_csv(tabla_path, index=False)
    print(f"\ncomparacion guardada en {tabla_path}")

    mejor_nombre = tabla.iloc[0]["modelo"]
    mejor_modelo = modelos[mejor_nombre]
    modelo_path = os.path.join(args.out_dir, "mejor_modelo.joblib")
    joblib.dump({"modelo": mejor_modelo, "scaler": scaler, "nombre": mejor_nombre}, modelo_path)
    print(f"mejor modelo ({mejor_nombre}) guardado en {modelo_path}")

    print("\ntabla final:")
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
