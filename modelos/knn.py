"""KNN para clasificar los movimientos de REHAB."""

from sklearn.neighbors import KNeighborsClassifier


def get_model():
    return KNeighborsClassifier(n_neighbors=5)
