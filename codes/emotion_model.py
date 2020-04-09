#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 20:23:26 2019

@author: goivagoi
"""

import torch
import torch.nn as nn
import pickle


def get_data(path):
    data = torch.load(path)
    dim = data[0][0][0].size(0)
    return data, dim


class customRNN(nn.Module):
    
    def __init__(self, inp_dim, hid_dim, out_dim):
        super(customRNN, self).__init__()
        
        self.hidden_size = hid_dim
        
        self.i2h = nn.Linear(inp_dim, hid_dim)
        self.h2h = nn.Linear(2*hid_dim, hid_dim)
        self.h2o = nn.Linear(hid_dim, out_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        
    def forward(self, uterance):
        hidden_prev = self.initHidden(); preds= torch.tensor([0]*7).float()
        i= 0
        for vector in uterance:
            hidden_cur = self.tanh(self.i2h(vector))
            combined = torch.cat((hidden_prev, hidden_cur))
            hidden_prev = self.tanh(self.h2h(combined))
            pred = self.softmax(self.h2o(hidden_prev)).view(7)
            preds += pred
            i += 1
        return preds/i
    
    def initHidden(self):
        return torch.zeros(self.hidden_size)


def main2(n, m, r):
    lpc_train = '../dataset/features/LPC/train/lpc_train.pt'
    lpc_p_train = '../dataset/features/LPC_plus/train/lpc_p_train.pt'
    mfcc_train = '../dataset/features/MFCC/train/mfcc_train.pt'
    mfcc_p_train = '../dataset/features/MFCC_plus/train/mfcc_p_train.pt'
    mfe_train = '../dataset/features/MFE/train/mfe_train.pt'
    mfe_p_train = '../dataset/features/MFE_plus/train/mfe_p_train.pt'
    paths = [lpc_train, lpc_p_train, mfcc_train, mfcc_p_train, mfe_train, mfe_p_train]
    path = paths[n]
    train_data, dim = get_data(path)
    model = customRNN(dim, 50, 7)
    criterion = torch.nn.modules.loss.BCELoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=r, rho=0.9, eps=1e-06)
    results = []
    for j in range(m):
        for i, dat in enumerate(train_data):
            model.zero_grad()
            data = dat[0]
            answer = dat[1]
            y_pred = model(data)
            loss = criterion(y_pred, answer)
            if loss != loss:
                print('report nan gradient')
                torch.save(model.state_dict('model_state.dict.{}_{}.pt'.format(j,i)))
            loss.backward()
            optimizer.step()
        results.append(test(model, n))
    
    return results


def test(model, n):
    lpc_test = '../dataset/features/LPC/test/lpc_test.pt'
    lpc_p_test = '../dataset/features/LPC_plus/test/lpc_p_test.pt'
    mfcc_test = '../dataset/features/MFCC/test/mfcc_test.pt'
    mfcc_p_test = '../dataset/features/MFCC_plus/test/mfcc_p_test.pt'
    mfe_test = '../dataset/features/MFE/test/mfe_test.pt'
    mfe_p_test = '../dataset/features/MFE_plus/test/mfe_p_test.pt'
    paths = [lpc_test, lpc_p_test, mfcc_test, mfcc_p_test, mfe_test, mfe_p_test]
    path = paths[n]
    test_data, dim = get_data[path]
    total = 0
    for dat in test_data:
        data = dat[0]; answer= dat[1]
        y = model(data)
        total += int(torch.argmax(y) == torch.argmax(answer) )
    
    return total/50


if __name__ == '__main__':
    names = ['lpc', 'lpc_p', 'mfcc', 'mfcc_p', 'mfe', 'mfe_p']
    for i in range(6):
        for r in [0.001, 0.01, 0.1, 1]:
            results = main2(i, 50, r)
            file = 'result.{}.{}.pkl'.format(names[i],r)
            with open(file, 'wb') as fh:
                pickle.dump(results, fh)




