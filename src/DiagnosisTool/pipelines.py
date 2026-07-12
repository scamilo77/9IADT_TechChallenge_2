from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def build_diabetes_pipeline(l1_ratio=0.0, C=1.0, solver="saga", max_iter=100):
    """Pipeline para Diabetes usando Logistic Regression"""
    model = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=l1_ratio,
        C=C,
        solver=solver,
        max_iter=max_iter
    )
    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),  # ajuste para melhor convergência
        ("clf", model)
    ])
    return pipeline

def build_breast_cancer_pipeline(kernel="linear", C=1.0, max_iter=1000):
    """Pipeline para Câncer de Mama usando SVM"""
    model = SVC(
        kernel=kernel,
        C=C,
        max_iter=max_iter
    )
    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),  # ajuste para melhor convergência
        ("clf", model)
    ])
    return pipeline
