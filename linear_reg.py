import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.animation import FuncAnimation

class LinearRegression:

    def __init__(self, dependentVariables, outputs, aplha=1, beta=2, learingRate=0.01) -> None:
        self.alpha = aplha
        self.beta = beta
        self.dependentVars = dependentVariables
        self.outputs = outputs
        self.lr = learingRate
        self.samples = len(outputs)

    def forwardPropagation(self):
        return self.alpha + np.multiply(self.beta, self.dependentVars)

    # MEAN SQUARED ERROR
    def getMSELoss(self, predictions): 
        return np.mean((self.outputs - predictions) ** 2)
    
    # MEAN ABSOLUTE ERROR 
    def getMAELoss(self, predictions):
        return np.abs((self.outputs - predictions) ** 2)
    
    # ROOT MEAN SQUARED ERROR
    def getRMSELoss(self, predictions):
        return np.sqrt(self.getMSELoss(predictions))
    
    # BATCH GRADIENT DECENT
    def backwardPropagationBGD(self):
        alphaGradient = 2 / self.samples * np.multiply()
        pass