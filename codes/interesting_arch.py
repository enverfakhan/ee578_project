#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:25:50 2019

@author: goivagoi
"""

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
    dim = data[0][0][0].shape[-1]
    return data, dim


class customRNN(nn.Module):
    
    def __init__(self, inp_dim,  out_dim):
        super(customRNN, self).__init__()
        
        self.hidden_size = inp_dim
        
        self.i2h = nn.Linear(2*inp_dim, inp_dim)
        self.h2h = nn.Linear(2*inp_dim, inp_dim)
        self.h2o = nn.Linear(5*inp_dim, out_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        
    def forward(self, uterance):
        hidden_prev1 = self.initHidden(); hidden_prev2= self.initHidden()
        preds = torch.tensor([0]*7).float()
        i = 0
        for tensors in uterance:
            tensor = self.resize(torch.cat((hidden_prev1, tensors), dim=1))
            hidden = self.tanh(self.i2h(tensor))
            hidden_prev1 = hidden[:,4:5, :]
            tensor = self.resize(torch.cat((hidden_prev2, tensors), dim=1))
            hidden = self.tanh(self.h2h(tensor))
            hidden_prev2 = hidden[:,4:5, :]
            pred = self.softmax(self.h2o(self.resize(hidden, out=True)))
            preds += pred
            i += 1
        return preds/i
    
    def initHidden(self):
        return torch.zeros([1, 1, self.hidden_size]).float()
    
    @staticmethod
    def resize(tensors, out=False):
        if out:
            return tensors.view(tensors.shape[1] * tensors.shape[2])
        else:
            return torch.cat(tuple(torch.cat(
                    (tensors[:, i:i+1, :], tensors[:, i+1:i+2, :]), 
                    dim=2)
                    for i in range(5)), 
                    dim=1)
    
    
def main2(m, r):

    path = '../dataset/features/MFCC/train/mfcc_arch_train.pt'

    train_data, dim = get_data(path)
    model = customRNN(dim, 7)
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
        results.append(test(model))
    
    return results


def test(model):

    path= '../dataset/features/MFCC/test/mfcc_arch_test.pt'
    test_data, dim = get_data(path)
    total = 0
    for dat in test_data:
        data = dat[0]
        answer = dat[1]
        y = model(data)
        total += int(torch.argmax(y) == torch.argmax(answer))
    
    return total/50


if __name__ == '__main__':
   
    for r in [0.001, 0.01, 0.1, 1]:
        results = main2(50, r)
        file = 'result.{}.{}.pkl'.format('mfcc_arch', r)
        with open(file, 'wb') as fh:
            pickle.dump(results, fh)




