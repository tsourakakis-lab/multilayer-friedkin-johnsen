from stochastic_block_model import *
import numpy as np
import matplotlib.pyplot as plt
import sys


def load_results(folder_name_all, iter, extras, ending = ''):
    train_loss = np.loadtxt(f'{folder_name_all}/{iter}_{extras}_train_loss{ending}.txt')
    test_loss = np.loadtxt(f'{folder_name_all}/{iter}_{extras}_test_loss{ending}.txt')
    resloss = np.loadtxt(f'{folder_name_all}/{iter}_{extras}_resloss{ending}.txt')
    return train_loss, test_loss, resloss


def plot_weight_data(ys, xlabel, ylabel, plotboxplot = False):
    fig, axes = plt.subplots()
    datas = []
    for algname in ys:
        labels, data = list(ys[algname].keys()), list(ys[algname].values())
        datas.append(data)
    names_plot = ['Multi-Layer', 'Singe-Layer1', 'Singe-Layer2', 'Singe-Layers']
    d = ['lightcoral', 'cornflowerblue', 'lightgreen', 'pink']
    c = ['indianred', 'royalblue', 'greenyellow', 'hotpink']
    m = ['darkred', 'blue', 'darkolivegreen', 'darkmagenta']
    meandata = []
    for i in range(len(names_plot)):
        meandata.append([np.mean(data) for data in datas[i]])
        if plotboxplot:
            axes.boxplot(datas[i], True, 'x', showmeans=True, 
                        patch_artist=True,
                        boxprops=dict(facecolor=d[i], color=c[i], linewidth=1),
                        capprops=dict(color=c[i], linewidth=1),
                        whiskerprops=dict(color=d[i], linewidth=1),
                        flierprops=dict(color=d[i], markeredgecolor=c[i], linewidth=1),
                        medianprops=dict(color=c[i]),
                        meanprops = dict(markerfacecolor = m[i], markeredgecolor=m[i], linewidth=1))
            line, = axes.plot(range(1, len(labels) + 1), meandata[-1], color = m[i])
            line.set_label(names_plot[i])
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.xlabel(xlabel)
    if plotboxplot == False:
        toprint = {}
        for j in range(len(names_plot)):
            toprint[names_plot[j]] = {meandata[j][0]}
        x = np.arange(1)  # the label locations
        width = 0.15  # the width of the bars
        multiplier = 0
        for attribute in toprint:
            offset = width * multiplier
            axes.bar(x + offset, toprint[attribute], width, 
                     label=attribute, color = m[multiplier])
            multiplier += 1
        plt.xlabel(f'{xlabel}={labels[0]}')
    plt.legend()
    plt.ylabel(ylabel)
    plt.savefig(f'./Figures/{typeofexp}_{ylabel}.png', bbox_inches='tight', dpi=400)
    plt.show()


def plot_layer_data(ys, xlabel, ylabel, plot_line = False):
    fig, axes = plt.subplots()
    datas = []
    for algname in ys:
        labels, data = list(ys[algname].keys()), list(ys[algname].values())
        datas.append(data)
    m = ['darkred', 'blue', 'darkolivegreen', 'darkmagenta']
    names_plot = ['Multi-Layer', 'Singe-Layer1', 'Singe-Layer2', 'Singe-Layers']
    meandata = []
    for i in range(len(datas)):
        meandata.append([np.mean(d) for d in datas[i]])
        if plot_line: 
            axes.plot(labels, meandata[-1], color = m[i], label = names_plot[i])
            plt.xlabel(xlabel)
    if plot_line == False:
        toprint = {}
        for j in range(len(names_plot)):
            toprint[names_plot[j]] = {meandata[j][0]}
        x = np.arange(1)  # the label locations
        width = 0.15  # the width of the bars
        multiplier = 0
        for attribute in toprint:
            offset = width * multiplier
            axes.bar(x + offset, toprint[attribute], width, 
                     label=attribute, color = m[multiplier])
            multiplier += 1
        plt.xlabel(f'{xlabel}={labels[0]}') 
    plt.legend()
    plt.ylabel(ylabel)
    plt.savefig(f'./Figures/{typeofexp}_{ylabel}.png', bbox_inches='tight', dpi=400)
    plt.show()


def plot_train_error_data(ys, typeofexp, list_of_el):
    datas = []
    for algname in ys:
        labels, data = list(ys[algname].keys()), list(ys[algname].values())
        datas.append(data)
    fig, axes = plt.subplots(len(datas[0]))
    m = ['darkred', 'blue', 'darkolivegreen', 'darkmagenta']
    names_plot = ['Multi-Layer', 'Singe-Layer1', 'Singe-Layer2', 'Singe-Layers']
    ax1 = axes
    for i in range(len(datas[0])):
        for j in range(len(names_plot)):
            if len(datas[0]) > 1: ax1 = axes[i]
            ax1.plot(range(1, len(datas[j][i]) + 1), datas[j][i], 
                    color = m[j], label = names_plot[j],linewidth=4.0)
            ax1.text(225, (min(datas[j][i])+max(datas[j][i]))/2,
                    f'{typeofexp}={list_of_el[i]}')
        if i<=(len(datas[j])-2): ax1.get_xaxis().set_visible(False)
        if i==(int(len(datas[j])/2)): ax1.set_ylabel('train_error')
        if i==0 and len(datas[0]) > 1: ax1.set_title(f'Training error vs {typeofexp}')
    plt.legend()
    plt.xlabel('Epochs')
    plt.savefig(f'./Figures/{typeofexp}_train_error.png', bbox_inches='tight', dpi=400)
    plt.show()
   
    
def plot_results(iters = 10, typeofexp = 'p', 
                 list_of_el = [0.1, 0.3, 0.5, 0.7, 0.9]):
    folder_name_all = f'./ExpResults/{typeofexp}_data'
    algnames = ['multi', 'l1', 'l2', 'lboth']
    fjm = {'train_loss':{}, 'test_loss':{}, 'resloss':{}, 'layerloss':{}}
    for algname in algnames: 
        for losses in fjm: fjm[losses][algname] = {}
    for el in list_of_el:
        for algname in algnames: 
            for losses in fjm: fjm[losses][algname][el] = []
        extras = f'{typeofexp}_{el}'
        for iter in range(iters):
            for algname in algnames: 
                train_loss, test_loss, resloss = load_results(folder_name_all, iter, extras, ending = f'_{algname}')
                fjm['train_loss'][algname][el].append(train_loss)
                fjm['test_loss'][algname][el].append(test_loss)
                fjm['resloss'][algname][el].append(resloss[0:-2])
                fjm['layerloss'][algname][el].append(resloss[-2:])
        for losstype in ['train_loss', 'test_loss', 'resloss', 'layerloss']:
            for algname in algnames: 
                fjm[losstype][algname][el] = np.mean(fjm[losstype][algname][el], axis = 0)
    plot_weight_data(fjm['resloss'], typeofexp, 'resloss')
    plot_weight_data(fjm['test_loss'], typeofexp, 'test_loss')
    plot_layer_data(fjm['layerloss'], typeofexp, 'layerloss')
    plot_train_error_data(fjm['train_loss'], typeofexp, list_of_el)


if __name__ == "__main__": 
    dictexp = {'p': [1]#[0.1, 0.3, 0.5, 0.7, 0.9],
               #'uniform': [0.1, 0.2, 0.3, 0.4, 0.5],
               #'normal': [0.1, 0.2, 0.3, 0.4, 0.5],
               #'l1': [0.3, 0.4, 0.5, 0.6, 0.7],
               #'naefalse': [0, 0.025, 0.05, 0.1, 0.2],
               #'naetrue': [0, 0.025, 0.05, 0.1, 0.2]
               }
    for typeofexp in dictexp:
        list_of_el = dictexp[typeofexp]
        plot_results(iters = 10, typeofexp = typeofexp, 
                     list_of_el = list_of_el)