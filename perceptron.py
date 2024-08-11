#Implement Multilayer Perceptron algorithm to simulate XOR gate.​​
import numpy as np
# define Unit Step Function​
def unitStep(v):
    if v >= 0:
      return 1
    else:
      return 0
# design Perceptron Model​

def perceptronModel(x, w, b):
  v = np.dot(w, x) + b
  y = unitStep(v)
  return y

def NOT_logicFunction(x):
    wNOT = -1
    bNOT = 0.5
    return perceptronModel(x, wNOT, bNOT)

def AND_logicFunction(x):
    wAND = np.array([1, 1])
    bAND = -1.5
    return perceptronModel(x, wAND, bAND)

def OR_logicFunction(x):
    wOR = np.array([1, 1])
    bOR = -0.5
    return perceptronModel(x, wOR, bOR)

def NAND_logicFunction(x):
    wNAND = np.array([-1, -1])
    bNAND = 1.5
    return perceptronModel(x, wNAND, bNAND)

def XOR_logicFunction(x):
    y1 = NAND_logicFunction(x)
    y2 = OR_logicFunction(x)
    y = AND_logicFunction([y1, y2])
    return y

def XNOR_logicFunction(x):
    y1 = NAND_logicFunction(x)
    y2 = OR_logicFunction(x)
    y = NOT_logicFunction([y1, y2])
    return y

# print(NOT_logicFunction([1,0]))
# print(AND_logicFunction([1,0]))
print(XNOR_logicFunction([1,1]))