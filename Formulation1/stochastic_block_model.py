import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
import os
import sys


#probability q
_q = 0
#We make two stochastic block models G1(p, _q) and G1(_q, p)
def make_block_model(n = 1000, p = 1, seed = 1):
    Ws = []
    true_list = [i for i in range(n)]
    probs = [[p, _q], [_q, p]]
    G1 = nx.stochastic_block_model(sizes=[int(n/2), n-int(n/2)], p=probs, 
                                  seed = seed, nodelist=true_list, directed = True)
    A1 = nx.adjacency_matrix(G1)
    for i in range(n): 
        if np.sum(A1[i,:])==0: A1[i,i] = 1
    Ws.append(torch.from_numpy((A1.T*1/A1.sum(axis = 1).T).T))
    
    probs = [[_q, p], [p, _q]]
    G2 = nx.stochastic_block_model(sizes=[int(n/2), n-int(n/2)], p=probs, 
                                  seed = seed, nodelist=true_list, directed = True)
    
    A2 = nx.adjacency_matrix(G2)
    for i in range(n): 
        if np.sum(A2[i,:])==0: A2[i,i] = 1
    Ws.append(torch.from_numpy((A2.T*1/A2.sum(axis = 1).T).T))
    
    Ws.append((torch.from_numpy(((A1+A2).T*1/(A1+A2).sum(axis = 1).T).T)))
    
    return Ws


#probability q
_q = 0
#We make two stochastic block models G1(p, _q) and G1(_q, p)
def make_directed_block_model(n = 1000, p = 1, seed = 1):
    Ws = []
    true_list = [i for i in range(n)]
    probs = [[p, _q], [p, _q]]
    G1 = nx.stochastic_block_model(sizes=[int(n/2), n-int(n/2)], p=probs, 
                                  seed = seed, nodelist=true_list, directed = True)
    A1 = nx.adjacency_matrix(G1)
    for i in range(n): 
        if np.sum(A1[i,:])==0: A1[i,i] = 1
    Ws.append(torch.from_numpy((A1.T*1/A1.sum(axis = 1).T).T))
    
    probs = [[_q, p], [_q, p]]
    G2 = nx.stochastic_block_model(sizes=[int(n/2), n-int(n/2)], p=probs, 
                                  seed = seed, nodelist=true_list, directed = True)
    
    A2 = nx.adjacency_matrix(G2)
    for i in range(n): 
        if np.sum(A2[i,:])==0: A2[i,i] = 1
    Ws.append(torch.from_numpy((A2.T*1/A2.sum(axis = 1).T).T))
    
    Ws.append((torch.from_numpy(((A1+A2).T*1/(A1+A2).sum(axis = 1).T).T)))
    
    return Ws

#mean 1, mean 2 and std 1, std 2
_m1 = 1
_s1 = 0
_m2 = 0
_s2 = 0
#We set opinions using a distribution N(_m1, _s1), N(_m2, _s2)
def make_initial_opinions(n1 = 500, n2 = 500, seed = 1):
    np.random.seed(seed)
    ops1 = np.random.normal(_m1, _s1, n1)
    for i in range(len(ops1)):
        if ops1[i]>1: ops1[i] = 1
        if ops1[i]<0: ops1[i] = 0
    ops2 = np.random.normal(_m2, _s2, n2)
    for i in range(len(ops2)):
        if ops2[i]>1: ops2[i] = 1
        if ops2[i]<0: ops2[i] = 0
    return torch.from_numpy(np.concatenate([ops1, ops2]))


#We make resistance parameters based on two distributions
#Uniform [0,1] and Normal(x, 0.1).
#Default Uniform.
def make_resistance_parameters(n = 1000, distr = 'uniform', x = 0.5, seed = 1):
    np.random.seed(seed)
    if distr not in ['uniform', 'normal']: distr = 'uniform'
    if distr == 'uniform':
        respar = np.random.uniform(low=0, high=1, size=(n,))
    else:
        respar = np.random.normal(x, 0.1, n)
        for i in range(len(respar)):
            if respar[i]>1: respar[i] = 1
            if respar[i]<0: respar[i] = 0
    return torch.from_numpy(respar)


#We make the layer weights l1 and l2, with l1 + l2 = 1 
#Default l1 = 1/4
def make_layer_parameters(l1 = 0.5):
    l2 = 1 - l1
    return torch.from_numpy(np.array([l1, l2]))


#Given initial opinions (initops), the resistance parameters (respar),
#the layer weights (layers), the weighted adjacency matrices (Ws) and 
#the opinions at time t (ops), function returns opinions at t+1.
def predict(initops, respar, layers, Ws, ops):
    n = len(initops)
    y = respar*initops
    for u in range(n): 
        for l in range(len(layers)):
            y[u] += float((1-respar[u])*layers[l]*(Ws[l][u,:].reshape(1,n)@ops))
    return y


#Adding noise to generated data from Normal N(0, s)
#i) Noise at end (nae = True) 
#ii) Noise at each timestamp (nae = False)
#Default nae = False and s = 0.1
def make_train_and_test_data(n = 1000,
                             p = 0.5, 
                             distr = 'normal', x = 0.1,
                             l1 = 1/4,
                             nae = True, s = 0.1,
                             T = 11,
                             seed = 1):
    Ws = make_block_model(n, p, seed)
    #Ws = make_directed_block_model(n, p, seed)
    ops = make_initial_opinions(int(n/2), n - int(n/2), seed)
    respar = make_resistance_parameters(n, distr, x, seed)
    layers = make_layer_parameters(l1)
    ops_all = []
    ops_all.append(ops)
    if nae == False: 
        ops_all[-1] += torch.from_numpy(np.random.normal(0, s**2, n))
        for j in range(n): 
            if ops_all[-1][j]>1: ops_all[-1][j] = 1
            if ops_all[-1][j]<0: ops_all[-1][j] = 0
    for _ in range(T):
        ops_all.append(predict(ops_all[0], respar, layers, Ws[0:2], ops_all[-1]))
    if nae: 
        for i in range(len(ops_all)):
            ops_all[i] += torch.from_numpy(np.random.normal(0, s**2, n))
            for j in range(n): 
                if ops_all[i][j]>1: ops_all[i][j] = 1
                if ops_all[i][j]<0: ops_all[i][j] = 0
    return Ws, ops, respar, layers, ops_all


#Saving data
def save_data(folder_name, ending, Ws, ops, respar, layers, ops_all):
    torch.save(Ws[0], f'{folder_name}/W1_{ending}.pt')
    torch.save(Ws[1], f'{folder_name}/W2_{ending}.pt')
    torch.save(Ws[2], f'{folder_name}/W3_{ending}.pt')
    torch.save(ops, f'{folder_name}/initops_{ending}.pt')
    torch.save(respar, f'{folder_name}/respar_{ending}.pt')
    torch.save(layers, f'{folder_name}/layers_{ending}.pt')   
    torch.save(torch.stack(ops_all, dim=0), f'{folder_name}/opsall_{ending}.pt')


#Generate and save data
def make_X_data(n = 100, iters = 10, typeofexp = 'p', list_of_el = [0.9]):
    print(f'Making {typeofexp}_data...')
    if not os.path.exists('./Graphs'): os.makedirs('./Graphs')
    folder_name_all = f'./Graphs/{typeofexp}_data'
    if not os.path.exists(folder_name_all): os.makedirs(folder_name_all)
    for iter in range(iters):
        folder_name = f'{folder_name_all}/{iter}'
        if not os.path.exists(folder_name): os.makedirs(folder_name)
        for el in list_of_el:
            if typeofexp == 'p': Ws, ops, respar, layers, ops_all = make_train_and_test_data(n = n, p = el, seed = 100*iter+int(el*10))
            elif typeofexp == 'l1': Ws, ops, respar, layers, ops_all = make_train_and_test_data(n = n, l1 = el, seed = 100*iter+int(el*10))
            elif typeofexp == 'uniform': Ws, ops, respar, layers, ops_all = make_train_and_test_data(n = n, distr = 'uniform', seed = 100*iter+int(el*10))
            elif typeofexp == 'normal': Ws, ops, respar, layers, ops_all = make_train_and_test_data(n = n, distr = 'normal', x = el, seed = 100*iter+int(el*10))
            elif typeofexp == 'naetrue': Ws, ops, respar, layers, ops_all = make_train_and_test_data(n = n, nae=True, s = el, seed = 100*iter+int(el*10))
            elif typeofexp == 'naefalse': Ws, ops, respar, layers, ops_all = make_train_and_test_data(n = n, nae=True,  s = el, seed = 100*iter+int(el*10))
            else: return
            save_data(folder_name, f'{typeofexp}_{el}', Ws, ops, respar, layers, ops_all)


if __name__ == "__main__": 
    n = 100
    make_X_data(n, iters = 10, typeofexp = 'p', list_of_el = [1])