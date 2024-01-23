import numpy as np
import networkx as nx
import torch


#predictions
def pred(lambdas, b, Ws, x):
    n = len(x)
    L = len(lambdas) - n
    y = lambdas[0:n]*b
    for u in range(n): 
        for l in range(L):
            y[u] += float((1-lambdas[u])*lambdas[n+l]*(Ws[l][u,:].reshape(1,n)@x))
    return y


#Loss Function
def lossL1(T, lambdas, b, Ws, x, y, criterion = torch.nn.MSELoss()):
    lossret = []
    for t in range(T):
        if len(Ws)==1:
            lossret.append((1/T)*criterion(pred(lambdas, b, Ws[0], x[t]), y[t]).numpy())
        else:
            lossret.append((1/T)*criterion(pred(lambdas, b, Ws[t], x[t]), y[t]).numpy())
    return lossret

