from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from data import load_diabetes_data, load_breast_cancer_data
from features import clean_diabetes_data, split_diabetes_data, clean_breast_cancer_data, split_breast_cancer_data
from pipelines import build_diabetes_pipeline, build_breast_cancer_pipeline, build_diabetes_pipeline_rf, build_breast_cancer_pipeline_rf

def train_diabetes_model():
    
    diabetes_data = load_diabetes_data()
    diabetes_data = clean_diabetes_data(diabetes_data)
    X, y = split_diabetes_data(diabetes_data, target_column="Outcome")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": build_diabetes_pipeline(),
        "Random Forest": build_diabetes_pipeline_rf()
    }
    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(f"{model_name} Diabetes Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"{model_name} Diabetes Classification Report:\n", classification_report(y_test, y_pred))
    
    
def train_breast_cancer_model():
    breast_cancer_data = load_breast_cancer_data()
    breast_cancer_data = clean_breast_cancer_data(breast_cancer_data)
    X, y = split_breast_cancer_data(breast_cancer_data, target_column="diagnosis")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": build_breast_cancer_pipeline(),
        "SVM": build_breast_cancer_pipeline_rf()
    }

    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(f"{model_name} Breast Cancer Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))                  
    
    
if __name__ == "__main__":
    train_diabetes_model()    
    train_breast_cancer_model() 