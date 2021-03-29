# General imports
import logging
import os
import numpy as np
import pandas as pd
from pampy import match
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings

# Sklearn imports
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV


# Get logger
logger = logging.getLogger("__main__")


def model_factory(model_type: str):
    """
    This function instantiates the correct learner given the specified model type
    """
    return match(model_type,
        'RandomForest',           RandomForestRegressor(criterion='mse'),
        'ExtraTrees',             ExtraTreesRegressor(),
        'XGBoost',                XGBRegressor(),
        'LightGBM',               LGBMRegressor(verbose=-1),
        'CatBoost',               CatBoostRegressor()
        )


class MLmodel(object):

    def __init__(self, settings: dict):

        self.settings = settings
        self.model = model_factory(model_type=settings['model_type'])
        return


    def optimize(self, X, y, params= {}):
        """
        Method that launches the hyperparameter tuning routine.
        I have included in the code the gridsearch method (scan the full hyperparameter space)
        and the random search (sample just some points of the hyperparameter space)
        """

        logger.info('* * * Optimizing hyperparameters')
        tuning_function = self.settings['tuning_function']
        hyperparameter_space = self.settings['hyperparameter_space']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if tuning_function == 'gridsearch':
                logger.info('* * * Launching the Grid Search')
                grid_search_estimator = GridSearchCV(self.model, param_grid=hyperparameter_space, **params)
                grid_search_estimator.fit(X, y)
                self.model = grid_search_estimator.best_estimator_
                logger.info('* * * * Optimal Hyperparameters: {0}'.format(grid_search_estimator.best_params_))
            elif tuning_function  == 'randomsearch':
                logger.info('* * * Launching the Random Search')
                random_search_estimator = RandomizedSearchCV(self.model,param_distributions=hyperparameter_space,**params)
                random_search_estimator.fit(X, y)

                self.model = random_search_estimator.best_estimator_
                logger.info('* * * * Optimal Hyperparameters: {0}'.format(self.model))

            else:
                logger.error("This hyperparameter tuning method doesn't exists!")
                raise NotImplementedError

        logger.info('* * * Hyperparameters optimized!')
        return

    def fit(self, X, y, fit_params={}):
        """
        Method that fit the model
        """
        self.model.fit(X, y, **fit_params)
        return

    def predict(self, X):
        """
        Method that return the prediction of the model
        """
        return self.model.predict(X)


    def plotFeaturesImportances(self, train_features):
        """
        A function to plot and print the feature importances of the model.

        """

        tree_model = self.model
        importances = pd.DataFrame({'Importance': tree_model.feature_importances_,
                                    'Feature_name': train_features})

        indices = np.argsort(tree_model.feature_importances_)[::-1]

        importances = importances.sort_values('Importance', ascending=False)
        N_feat = len(importances)

        plt.figure(figsize=(20, 15))
        plt.xticks(range(0, N_feat), importances['Feature_name'], rotation=90)
        plt.xlim([-1, N_feat])
        plt.tight_layout()
        plt.bar(range(0, N_feat),
                        importances['Importance'])
        plt.xlabel('Feature name')
        plt.ylabel('Feature importance')
        plt.grid()
        plt.tight_layout()
        plt.show(block=False)

        logger.info("\n--------- Feature Importance ---------")
        for f in range(0, len(train_features)):
            logger.info("%2d) %-*s %f" % (f + 1, 30, train_features[indices[f]], tree_model.feature_importances_[indices[f]]))
        return