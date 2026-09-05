"""SVM para clasificar los movimientos de REHAB."""

from sklearn.svm import SVC

RANDOM_STATE = 42


def get_model():
    return SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE)
