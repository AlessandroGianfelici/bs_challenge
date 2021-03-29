# General imports
import logging
import os
import numpy as np
import pandas as pd
from pampy import match
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import curve_fit
# Get logger
logger = logging.getLogger("__main__")

class subscriptionModel(object):

    def __init__(self):
        
        return          

    def fit(self, X, y):
        """
        Method that fit the model
        """
        empiricalDist = partial(self.subscriptionDistribution, T=max(X))
        best_fit = curve_fit(f=empiricalDist,
          xdata=X,
          ydata=y,
          p0 = 0.5)
        return best_fit[0].item()

    @staticmethod
    def subscriptionDistribution(n, p, T):
        return np.where(n==T, (1-p)**n, p*(1-p)**(n-1))

    def simulate(self, p, T, N_customers):
        """
        Method that return the prediction of the model
        """
        data = pd.DataFrame()
        data['renewals'] = list(range(T+1))
        partialSim = partial(self.subscriptionDistribution, p=p, T=T)
        data['N_simulated'] = (N_customers*data['renewals'].apply(partialSim)).astype(int)
        data['T'] = T
        return data


    def plot(self, data, columns=['N']):
        """
        A function to plot and print the feature importances of the model.

        """
        data = data.sort_values('T', ascending=False)
        plt.figure(figsize=(20, 15))
        plt.xticks(data['renewals'], data['renewals'], rotation=90)
        plt.xlim([-1, int(max(data['T']))+1])
        plt.tight_layout()
        for col in columns:
            plt.bar(data['renewals'].astype(int), data[col].astype(int), alpha=0.3)
        plt.xlabel('renewals')
        plt.ylabel(columns)
        plt.grid()
        plt.tight_layout()
        return