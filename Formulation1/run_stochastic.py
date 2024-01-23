import sys
import torch
import numpy as np
from tensor_solver import TensorSolver
from functionality import lossL1
import os
import copy


#Loading data
def load_data(folder_name, ending):
    Ws = []
    Ws.append(torch.load(f'{folder_name}/W1_{ending}.pt'))
    Ws.append(torch.load(f'{folder_name}/W2_{ending}.pt'))
    Ws.append(torch.load(f'{folder_name}/W3_{ending}.pt'))
    respar = torch.load(f'{folder_name}/respar_{ending}.pt')
    ops = torch.load(f'{folder_name}/initops_{ending}.pt').reshape(len(respar),1)
    layers = torch.load(f'{folder_name}/layers_{ending}.pt')   
    ops_all = torch.load(f'{folder_name}/opsall_{ending}.pt')
    return Ws, ops, respar, layers, ops_all


#Splitting to train and test data
def split_train_and_test_data(ops_all, T_train, T_test):
    x_train = [ops_all[i].reshape(n, 1) for i in range(T_train)]
    y_train = [ops_all[i].reshape(n, 1) for i in range(1,T_train+1)]
    x_test = [ops_all[i].reshape(n, 1) for i in range(T_train, T_train+T_test)]
    y_test = [ops_all[i].reshape(n, 1) for i in range(T_train+1, T_train+T_test+1)]
    return x_train, y_train, x_test, y_test 


#Saving results
def save_results(folder_name_all, iter, extras, train_loss, test_loss, resloss, algname):
    np.savetxt(f'{folder_name_all}/{iter}_{extras}_train_loss_{algname}.txt', train_loss)
    np.savetxt(f'{folder_name_all}/{iter}_{extras}_test_loss_{algname}.txt', test_loss)
    np.savetxt(f'{folder_name_all}/{iter}_{extras}_resloss_{algname}.txt', resloss)


#Run the experiments 
def run_the_experiments(T_train, T_test, iter, folder_name_all, 
                        best_lrdiv = None, typeofexp = 'p', 
                        el = 0.1, algname = 'multi'):
    extras = f'{typeofexp}_{el}'
    Ws, ops, respar, layers, ops_all = load_data(f'./Graphs/{typeofexp}_data/{iter}', f'{extras}')
    x_train, y_train, x_test, y_test = split_train_and_test_data(ops_all, T_train, T_test)
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
    train_loss = solver.training(num_epochs = 100, lrdiv = best_lrdiv)
    test_loss =  lossL1(T_test, solver.lambdas.detach(), ops, Ws_, x_test, y_test, criterion = torch.nn.L1Loss())
    resloss = np.abs(np.concatenate([respar.numpy(), layers.numpy()])-solver.lambdas.detach().numpy().reshape(-1))
    save_results(folder_name_all, iter, f'{extras}', train_loss, test_loss, resloss, algname)
    return best_lrdiv


#Make the experiments 
def make_X_experiments(iters = 10, T_train = 7, T_test = 3, typeofexp = 'p', 
                       list_of_el = [0.1, 0.3, 0.5, 0.7, 0.9]):
    if not os.path.exists('./ExpResults'): os.makedirs('./ExpResults')
    folder_name_all = f'./ExpResults/{typeofexp}_data'
    if not os.path.exists(folder_name_all): os.makedirs(folder_name_all)
    for algname in ['multi', 'l1', 'l2', 'lboth']:
        best_lrdiv = 2
        for el in list_of_el:
            for iter in range(iters):
                if iter == 0: best_lrdiv = None
                best_lrdiv = run_the_experiments(T_train, T_test, iter, folder_name_all, 
                                                best_lrdiv, typeofexp, el, algname)

                        
if __name__ == "__main__": 
    n = 100
    dictexp = {'p': [1]}
    for typeofexp in dictexp:
        list_of_el = dictexp[typeofexp]
        make_X_experiments(iters = 10, typeofexp = typeofexp, 
                           list_of_el = list_of_el)