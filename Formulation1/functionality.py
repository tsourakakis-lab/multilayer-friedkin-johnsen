import numpy as np
import networkx as nx
import torch


#Prediction
def pred(lambdas, b, Ws, x, n, L, t = 0):
    prediction = lambdas[0:n]*b
    for l in range(L): 
        if len(Ws)==1:
            prediction = prediction+lambdas[n+l]*(Ws[0][l]*((1-lambdas[0:n])@torch.ones((1,n)).double()))@x[t]
    return prediction  

#Loss Function
def lossL1(T, lambdas, b, Ws, x, y, criterion = torch.nn.MSELoss()):
    lossret = []
    #print(criterion)
    for t in range(T):
        if len(Ws)==1:
            lossret.append(criterion(pred(lambdas, b, Ws, x, len(x[0]), len(lambdas) - len(x[0]), t=t), y[t]).numpy())
        else:
            lossret.append(criterion(pred(lambdas, b, Ws, x), y[t]).numpy())
    return lossret


