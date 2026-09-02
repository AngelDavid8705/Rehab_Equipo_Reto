"""Arbol de decision para clasificar los movimientos de REHAB. Autor: Angel Lugo."""

from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def get_model():
    return DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE)
