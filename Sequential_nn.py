from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()

iris = pd.DataFrame(iris.data, columns=iris.feature_names)

X, y = datasets.load_iris(return_X_y = True)

ionospher = datasets.load_ionosphere()

ionospher = pd.DataFrame(ionospher.data, columns=ionospher.feature_names)

X1, y1 = datasets.load_ionosphere(return_X_y = True)

from  sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.30)

X_train, X_test
X_train1, X_test1

# Define the network model and its arguments
# Set the number of neurons/nodes for each layer
model = Sequential()
model.add(Dense(32, input_shape=(4,))) # input layer shape = 4
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('sigmoid'))

# Compile the moedl and calculate its accuracy
#sgd = SGD(lr=0.001, decay=1e.6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

from keras.utils import to_categorical

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

model.fit(X_train, y_train_encoded, epochs=50, batch_size=32)

score = model.evaluate(X_test, y_test_encoded)
print(score)

"""**64 NEURON HIDDEN LAYER - NeuralNetwork**"""

model64 = Sequential()
model64.add(Dense(64, input_shape=(4,))) # input layer shape = 4
model64.add(Activation('sigmoid'))
model64.add(Dense(3))
model64.add(Activation('sigmoid'))

model64.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# ENCODIGN LABELS TO BINARY / CATEGORICAL FORM: 0 / 1
y_train_encoded_64 = to_categorical(y_train)
y_test_encoded_64 = to_categorical(y_test)

model64.fit(X_train, y_train_encoded_64, epochs=150, batch_size=32)

score_64 = model.evaluate(X_test, y_test_encoded)
print("64 LAYER MODEL SCORE:  ",score_64)

"""**EXP - 4: With SGD Optimizer**"""

from keras.optimizers import SGD

sgd = SGD(lr=0.0001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train_encoded, epochs=55, batch_size=32, validation_data=(X_test, y_test_encoded))
scores = model.evaluate(X_test, y_test_encoded)
print("SCORE AFTER SGD, & VALIDATION-SET:", scores)

"""**Nestorev Gradient Descent 64 hidden layers With 345 Epochs ~ 94% Accuracy**"""

sgd_nes = SGD(lr=0.00001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train_encoded, epochs=345, batch_size=32, validation_data=(X_test, y_test_encoded))
scores = model.evaluate(X_test, y_test_encoded)
print("SCORE AFTER SGD, & VALIDATION-SET:", scores)

model128 = Sequential()
model128.add(Dense(128, input_shape=(4,))) # input layer shape = 4
model128.add(Activation('sigmoid'))
model128.add(Dense(3))
model128.add(Activation('sigmoid'))

model128.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

sgd_nes = SGD(lr=0.00001, momentum=0.9, nesterov=True)
model128.fit(X_train, y_train_encoded, epochs=345, batch_size=32, validation_data=(X_test, y_test_encoded))
scores = model.evaluate(X_test, y_test_encoded)
print("SCORE AFTER SGD, & VALIDATION-SET:", scores)