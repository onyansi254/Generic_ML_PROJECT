import os
import sys
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}

        for model_name, model in models.items():
            model.fit(x_train, y_train)  # Train Model

            y_train_pred = model.predict(x_train)  # Predictions on training data
            y_test_pred = model.predict(x_test)  # Predictions on test data

            train_model_score = r2_score(y_train, y_train_pred)  # R^2 score for training data
            test_model_score = r2_score(y_test, y_test_pred)  # R^2 score for test data

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
