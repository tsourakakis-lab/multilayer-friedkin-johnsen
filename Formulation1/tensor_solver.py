import numpy as np
from numpy import linalg as LA
import torch
import torch.nn as nn
import cvxpy as cp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TensorSolver:
    #Init Function
    def __init__(self, b, Ws, x, y):
        #number of nodes
        self.n = len(Ws[0][0])
        #number of layers
        self.L =  len(Ws[0])
        #number of timestamps
        self.T = len(x)
        #unknown lambdas: first n are biased-lambdas and second L are layer-lambdas
        self.lambdas = torch.from_numpy(np.ones(self.n+self.L).reshape(self.n+self.L, 1)).to(device)
        self.lambdas.requires_grad_(True)
        #Loss function
        self.criterion = torch.nn.MSELoss()
        #biases
        self.b = b
        #stochastic matrices
        self.Ws = Ws
        #input
        self.x = x
        #output
        self.y = y


    #Predict Function 
    def pred(self, t = 0):
        prediction = self.lambdas[0:self.n]*self.b
        for l in range(self.L): 
            if len(self.Ws)==1:
                prediction = prediction+self.lambdas[self.n+l]*(self.Ws[0][l]*((1-self.lambdas[0:self.n])@torch.ones((1,self.n)).double()))@self.x[t]
            else:
                prediction = prediction+self.lambdas[self.n+l]*(self.Ws[t][l]*((1-self.lambdas[0:self.n])@torch.ones((1,self.n)).double()))@self.x[t]
        return prediction    


    #Loss Function
    def loss(self, T=None):
        if T is not None: toprint = True
        if T is None: T = self.T
        squared_all = 0
        for t in range(T):
            squared_all = squared_all + (1/self.T)*self.criterion(self.pred(t), self.y[t])
        return squared_all
    
    
    #L1
    def lossL1(self, T=None):
        if T is not None: toprint = True
        if T is None: T = self.T
        squared_all = 0
        for t in range(T):
            squared_all = squared_all + (1/self.T)*torch.nn.L1Loss()(self.pred(t), self.y[t])
        return squared_all
    
     #L1
    def lossL12(self, T):
        squared_all = 0
        for t in T:
            squared_all = squared_all + (1/self.T)*torch.nn.L1Loss()(self.pred(t), self.y[t])
        return squared_all
    
    #Gradient Descent
    def gradient(self, lr = 1):
        if self.lambdas.requires_grad:
            self.loss().backward()
            self.lambdas.data  -= lr*self.lambdas.grad.data
            self.projection()


    #Projection to feasible solution
    def projection(self):
        self.lambdas.grad.zero_()
        Lambdas = self.lambdas.data
        Lnew = cp.Variable(Lambdas.shape)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(Lambdas - Lnew)),
                          [Lnew >= 0,
                           Lnew <= 1,
                           cp.sum(Lnew[self.n:])==1])
        prob.solve()
        self.lambdas = torch.tensor(Lnew.value)
        self.lambdas.requires_grad_()
    
    
    #Training Phase
    def training(self, num_epochs = 500, lrdiv = 5, printall = True, printL1 = True): 
        if printall: 
            if printL1: print('Initial: ',self.lossL1().item())
            else: print('Initial: ',self.loss().item())
        training_loss = []
        lr = self.n/lrdiv
        for epoch in range(num_epochs):
            self.gradient(lr=lr)        
            lossval = 0
            if printL1: lossval = self.lossL1().item()
            else: lossval = self.loss().item()
            training_loss.append(lossval)
            if printall:
                if epoch % 1 == 0:
                    print(f"Epoch: {epoch}, loss {lossval:.8}")
                    print(f"Epoch: {epoch}, loss {self.loss().item():.8}")
                    print()
        return training_loss

    
    #Testing Phase
    def test(self, Ws, x, y):
        #number of nodes
        self.n = len(Ws[0][0])
        #number of layers
        self.L =  len(Ws[0])
        #number of timestamps
        self.T = len(Ws)
        #Loss function
        self.criterion = torch.nn.MSELoss()
        #input
        self.x = x
        #output
        self.y = y
        return self.loss(T=len(y)) 

