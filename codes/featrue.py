#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:19:08 2019

@author: goivagoi
"""
import torch
import speechpy as sp
import librosa as ls
import random
import numpy as np
import os


def get_files(path):
    b = []
    for f in os.walk(path):
        b.append(f)    
    main = b[0][0]
    files = b[0][2]
    paths = [os.path.join(main, c) for c in files ]
    return paths


def mfcc(y):
    mfcc_raw = sp.feature.mfcc(y, 16000, frame_length=0.1, frame_stride=0.05)
    mfcc_with_delta = sp.feature.extract_derivative_feature(mfcc_raw)
    return mfcc_with_delta.reshape([len(mfcc_raw), 39], order='F')


def tensorfy(features):
    outs = []
    l = len(features) % 5
    features = np.concatenate((features, np.zeros([l, 39])))
    n = int(len(features) / 5)
    for i in range(n):
        outs.append(torch.from_numpy(features[5*i: 5*(i+1), :]).view([1, 5, 39]).float())
    
    return outs


def main(paths):

    mfcc_train = '../dataset/features/MFCC/train/mfcc_arch_train.pt'
    mfcc_test = '../dataset/features/MFCC/test/mfcc_arch_test.pt'
    classes = {'A': 0, 'E': 1, 'F': 2, 'L': 3, 'N': 4, 'W': 5, 'T': 6 }
    base = torch.tensor([0]*7).float()
    mfcc_ser = []
    random.shuffle(paths)
    for path in paths:
        answer = base.clone()
        answer[classes[path[-6]]] = 1
        y, sr = ls.load(path, sr=None)
        mfcc_feat = mfcc(y)
        mfcc_ser.append((tensorfy(mfcc_feat), answer))    
    
    torch.save(mfcc_ser[:-50, :], mfcc_train)
    torch.save(mfcc_ser[-50:], mfcc_test)
    
    return mfcc_ser 
