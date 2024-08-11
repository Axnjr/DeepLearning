import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.animation import FuncAnimation

class LinearRegression:

    def __init__(self, dependentVariables, outputs, slope=1, intercept=1, learningRate=0.0001) -> None:
        self.slope = slope
        self.intercept = intercept
        self.dependentVars = dependentVariables
        self.outputs = outputs
        self.lr = learningRate
        self.samples = len(outputs)

        print(slope, intercept, self.lr, self.samples)

    def forwardPropagation(self):
        return self.slope + np.multiply(self.intercept, self.dependentVars)

    # MEAN SQUARED ERROR
    def getMSELoss(self, predictions): 
        return np.mean((self.outputs - predictions) ** 2)
    
    # MEAN ABSOLUTE ERROR 
    def getMAELoss(self, predictions):
        return np.mean(np.abs(self.outputs - predictions))
    
    # ROOT MEAN SQUARED ERROR
    def getRMSELoss(self, predictions):
        return np.sqrt(self.getMSELoss(predictions))
    
    # BATCH GRADIENT DECENT
    def backwardPropagationBGD(self, predictions):
        diff = predictions - self.outputs
        # Actual formula = 2 / n * summation of ( diff * inputs )
        gradient_slope = 2 * (np.mean(np.multiply(self.dependentVars, diff)))
        # Actual formula = 2 / n * summation of (ypred - yreal) i.e diff
        gradient_intercept = 2 * (np.mean(predictions - self.outputs))

        self.slope -= self.lr * gradient_slope
        self.intercept -= self.lr * gradient_intercept

    def train(self, epochs=10):
        for epoch in range(epochs):
            predictions = self.forwardPropagation()
            self.backwardPropagationBGD(predictions)
            
            # Calculate and print the loss at each epoch
            mse = self.getMSELoss(predictions)
            mae = self.getMAELoss(predictions)
            rmse = self.getRMSELoss(predictions)

            if mse > 1e10:  # You can adjust this threshold
                print("Loss is too high. Stopping training.")
                break
            
        print(f"\n Epochs: {epoch+1}/{epochs} | MSE={mse} | MAE={mae} | RMSE={rmse} \n")

    def predict(self, new_data):
        return self.slope + np.multiply(self.intercept, new_data)

    def plot(self, newData, newDataOutput):
        plt.figure(figsize=(10, 6))
        plt.scatter(newData, newDataOutput, color='blue', label='Actual Data')

        x_values = np.linspace(min(self.dependentVars), max(self.dependentVars), 100)
        y_values = self.slope + self.intercept * x_values
        plt.plot(x_values, y_values, color='red', label='Regression Line')

        predicted_outputs = self.predict(newData)
        plt.scatter(newData, predicted_outputs, color='green', label='Predicted Data', marker='x')
        
        plt.xlabel('Dependent Variables')
        plt.ylabel('Outputs')
        plt.title('Linear Regression Fit with Predictions')
        plt.legend()
        plt.show()


url = 'https://media.geeksforgeeks.org/wp-content/uploads/20240320114716/data_for_lr.csv'
data = pd.read_csv(url)

# Drop the missing values
data = data.dropna()

# print(data)

# training dataset and labels
train_input = np.array(data.x[0:5]).reshape(5, 1)
train_output = np.array(data.y[0:5]).reshape(5, 1)
test_input = np.array(data.x[199:204]).reshape(5, 1)
test_output = np.array(data.y[199:204]).reshape(5, 1)

lr = LinearRegression(train_input, train_output)
lr.train(500)
res = lr.predict(test_input)
lr.plot(test_input, test_output)

for i in range(len(res)):
    print("X = ", test_input[i], " | Predicted = ", res[i], " | Real =", test_output[i]);