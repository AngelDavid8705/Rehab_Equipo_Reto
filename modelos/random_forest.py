"""Random Forest para clasificar los movimientos de REHAB. Autor: Angel Lugo."""

from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42


def get_model():
    return RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    )
