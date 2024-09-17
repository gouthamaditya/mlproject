import os
import sys
import pickle
import dill
import pandas as pd
import numpy as np
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    This function performs hyperparameter tuning for a list of models using GridSearchCV, 
    trains the best model on training data, and evaluates it on test data.

    Args:
        X_train (ndarray): Features for training.
        y_train (ndarray): Target for training.
        X_test (ndarray): Features for testing.
        y_test (ndarray): Target for testing.
        models (dict): A dictionary of models to be trained and evaluated.
        param (dict): A dictionary of hyperparameters for each model.

    Returns:
        dict: A report of R² scores for each model after testing.
    """
    try:
        report = {}

        for model_name in models.keys():
            model = models[model_name]
            parameters = param[model_name]

            logging.info(f"Performing Grid Search for {model_name}")

            # GridSearchCV for hyperparameter tuning
            gs = GridSearchCV(model, parameters, cv=3, n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)

            # Best model after hyperparameter tuning
            best_model = gs.best_estimator_

            logging.info(f"Best parameters for {model_name}: {gs.best_params_}")

            # Train the best model on the training data
            best_model.fit(X_train, y_train)

            # Predict on the training set
            y_train_pred = best_model.predict(X_train)
            # Predict on the test set
            y_test_pred = best_model.predict(X_test)

            # Calculate R² score for training and testing sets
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            logging.info(f"{model_name} - Train R2 Score: {train_model_score}")
            logging.info(f"{model_name} - Test R2 Score: {test_model_score}")

            # Save the test score in the report
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load a Python object from a file using pickle.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
