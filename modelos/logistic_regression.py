"""Regresion logistica para clasificar los movimientos de REHAB."""

from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42


def get_model():
    return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
