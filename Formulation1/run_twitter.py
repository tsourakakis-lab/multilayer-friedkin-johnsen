import sys
import torch
import numpy as np
from tensor_solver import TensorSolver
from functionality import lossL1
import os
import copy
import networkx as nx
import pandas as pd

def weighted(G):
    A = nx.adjacency_matrix(G).todense()
    con = 0
    for i in range(len(G.nodes())): 
        if np.sum(A[i,:])==0: 
            A[i,i] = 1
            con+=1
#     print(con, A.shape)
    return torch.from_numpy((A.T*1/A.sum(axis = 1).T).T)

def read_edge_list(filename, n = 3409):
    G = nx.Graph()
    for i in range(n): 
        G.add_node(i)
    with open(filename) as file:
        for line in file:
            u = int(line.split(' ')[0])
            v = int(line.split(' ')[1][0:-1])
            G.add_edge(u,v)
    return G

#Twitter Dataset experiments
def make_experiments_twitter(T_train = 10, T_test = 10, 
                             optype = 'vax'):
    import networkx as nx
    import pandas as pd
    Wsall = []
    G = read_edge_list("./Twitter/G.edgelist", n = 3409)
    n = len(G.nodes())
    G = weighted(G)
    G_2 = read_edge_list(f'./Twitter/G_{optype}.edgelist', n = 3409)
    G_2 = weighted(G_2)
    for _ in range(T_train+T_test): Wsall.append([G,G_2])
    
    opinions = np.array(pd.read_csv(f'./Twitter/{optype}_opinions.txt', sep=' ', header=None))/10
    print(opinions)
    x_train = [torch.from_numpy(opinions[:,i]).reshape(n, 1) for i in range(T_train)]
    y_train = [torch.from_numpy(opinions[:,i]).reshape(n, 1) for i in range(1,T_train+1)]
    x_test = [torch.from_numpy(opinions[:,i]).reshape(n, 1) for i in range(T_train, T_train+T_test)]
    y_test = [torch.from_numpy(opinions[:,i]).reshape(n, 1) for i in range(T_train+1, T_train+T_test+1)]
    b = torch.from_numpy((0.5*np.ones(n)).reshape(n,1))
    solver  = TensorSolver(b, Wsall[0:T_train], x_train, y_train)
    train_loss = solver.training()
    print(train_loss[-1])
    test_loss =  loss(T_test, solver.lambdas.detach(), b,
                      Wsall[T_train:T_train+T_test], x_test, y_test)
    print(test_loss)


def load_real_data(n, optype):
    Ws = []
    G1 = read_edge_list(f'./Twitter/G_{optype[0]}_scc.edgelist', n)
    G2 = read_edge_list(f'./Twitter/G_{optype}_scc.edgelist', n)
    Ws.append(weighted(G1))
    Ws.append(weighted(G2))
    A1 = nx.adjacency_matrix(G1).todense()
    A2 = nx.adjacency_matrix(G2).todense()
    for i in range(n): 
        if np.sum(A1[i,:])==0: A1[i,i] = 1
    for i in range(n): 
        if np.sum(A2[i,:])==0: A2[i,i] = 1
    Ws.append((torch.from_numpy(((A1+A2).T*1/(A1+A2).sum(axis = 1).T).T)))
    return Ws


#Splitting to train and test data
def split_train_and_test_data(n, ops_all, T_train, T_test):
    x_train = [ops_all[i].reshape(n, 1) for i in range(T_train)]
    y_train = [ops_all[i].reshape(n, 1) for i in range(1,T_train+1)]
    x_test = [ops_all[i].reshape(n, 1) for i in range(T_train, T_train+T_test)]
    y_test = [ops_all[i].reshape(n, 1) for i in range(T_train+1, T_train+T_test+1)]
    return x_train, y_train, x_test, y_test 



#Twitter Dataset experiments
def make_experiments_twitter_new(n = 430,
                                 T_train = 45, 
                                 T_test = 5, 
                                 optype = 'vax',
                                 algname = 'multi',
                                start_pos=0, num_epochs=100, printall=False):   
    Ws = load_real_data(n, optype)
    ops_all = np.array(pd.read_csv(f'./Twitter/{optype}_ops.txt', sep=' ', header=None))/10
    ops = torch.from_numpy(ops_all[:,0].reshape(n,1))
    x_train, y_train, x_test, y_test = split_train_and_test_data(n, torch.from_numpy( ops_all[:,start_pos:start_pos+T_train+T_test+1].T), T_train, T_test)
    #find best learning rate
    best_lrdiv = 10 #fix best_lrdiv
    #----------------------------------------------------------------------
    if best_lrdiv is None:
        max_train_loss = 100000000000
        for lrdiv in [2, 5, 10, 20]:
            solver  = TensorSolver(ops, [[Ws[1], Ws[0]]], x_train, y_train)
            train_loss = solver.training(lrdiv = lrdiv, num_epochs = 100, printall = False)
            print(f'Final train loss for {lrdiv} is {train_loss[-1]}...')
            if train_loss[-1] < max_train_loss: 
                max_train_loss = train_loss[-1]
                best_lrdiv = lrdiv
    #----------------------------------------------------------------------
    Ws_ = copy.deepcopy(Ws)
    print(f'Algorithm {algname}')
    if algname == 'multi': Ws_ = [[Ws[0], Ws[1]]]
    elif algname == 'l1': Ws_ = [[Ws[0], Ws[0]]]
    elif algname == 'l2': Ws_ = [[Ws[1], Ws[1]]]
    elif algname == 'lboth': Ws_ = [[Ws[2], Ws[2]]]
    else: return
    solver  = TensorSolver(ops, Ws_, x_train, y_train)
    train_loss = solver.training(num_epochs = num_epochs, lrdiv = best_lrdiv, printall=printall)
    test_loss =  lossL1(T_test, solver.lambdas.detach(), ops, Ws_, x_test, y_test, criterion = torch.nn.L1Loss())
    print(test_loss)
    return (train_loss, test_loss, solver.lambdas.detach())



if __name__ == "__main__": 
    make_experiments_twitter_new(n = 430, optype = 'vax', algname = 'lboth')
    make_experiments_twitter_new(n = 343, optype = 'war', algname = 'l1')
    pass