"""Support Vector Machine (SVM) para clasificar los movimientos de REHAB."""

from sklearn.svm import SVC

RANDOM_STATE = 42


def get_model(kernel="rbf", C=10.0, gamma="scale"):
    """Retorna un clasificador SVM configurado con kernel RBF y C=10.0 por defecto."""
    return SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        random_state=RANDOM_STATE,
    )
