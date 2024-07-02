import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Split training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regressor": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30]
                },
                "Decision Tree": {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                "Linear Regressor": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                },
                "XGBRegressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                "CatBoosting Regressor": {
                    'iterations': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [3, 5, 7]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
            }

            best_models = {}
            for model_name in models.keys():
                model = models[model_name]
                param = params[model_name]
                logging.info(f"Tuning hyperparameters for {model_name}")
                grid_search = GridSearchCV(estimator=model, param_grid=param, cv=5, scoring='r2', n_jobs=-1)
                grid_search.fit(x_train, y_train)
                best_models[model_name] = grid_search.best_estimator_

            model_report: dict = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=best_models
            )

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = best_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
