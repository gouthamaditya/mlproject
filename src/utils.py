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

        for model_name, model in models.items():
            para = param.get(model_name, {})
            
            # Use GridSearchCV for models with hyperparameters
            if para:
                gs = GridSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_  # Get the best estimator from GridSearchCV
            else:
                # If no parameters, just fit the model directly
                model.fit(X_train, y_train)
                best_model = model  # If no GridSearchCV, use the directly trained model

            # Make predictions on the train and test set
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate R² scores for train and test sets
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test R² score in the report dictionary
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
