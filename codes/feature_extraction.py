#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:33:11 2019

@author: goivagoi
"""
import audiolazy as al
import speechpy as sp
import librosa as ls
import numpy as np
import os
import torch
import random


def get_files(path):
    b = []
    for f in os.walk(path):
        b.append(f)    
    main = b[0][0]
    files = b[0][2]
    paths = [os.path.join(main, c) for c in files]
    return paths


def mfe(y, zc):
    mfe_, energy = sp.feature.mfe(y, 16000)
    energy = energy.reshape(len(energy), 1)
    l = min(len(energy), len(zc))
    zz = zc[:l, :]
    mfe_plus = np.concatenate((mfe_[:l,:], energy[:l,:], zz), axis= 1)
    return mfe_, mfe_plus, energy[:l,:], zz


def mfcc(y):
    mfcc_raw = sp.feature.mfcc(y, 16000)
    mfcc_with_delta = sp.feature.extract_derivative_feature(mfcc_raw)
    return mfcc_with_delta.reshape([len(mfcc_raw), 39], order='F')


def mfcc_plus(mfcc, energy, zc):
    return np.concatenate((mfcc, energy, zc), axis=1)


def zero_cross(y):
    zc = ls.feature.zero_crossing_rate(y, 320, 160, center=False)
    return zc.T


def lpc(stacks):
    lpc_freq = []
    for frame in stacks:
        lpc10 = al.lpc(frame, order= 10)
        lpc30 = al.lpc(frame, order= 30)
        freq10 = np.abs(lpc10.freq_response(freq=np.linspace(0, np.pi, 20)))[::-1]
        freq30 = np.abs(lpc30.freq_response(freq=np.linspace(0, np.pi, 20)))[::-1]
        lpc_freq.append(np.concatenate((freq10, freq30)))
    
    return np.array(lpc_freq)


def lpc_plus(lpc_feat, energy, zc):
    l = min(len(zc), len(energy))
    return np.concatenate((lpc_feat[:l, :], energy[:l, :], zc[:l, :]), axis=1)


def get_stacks(y):
    return sp.processing.stack_frames(y, 16000, frame_stride=0.01)


def serialize(y):
    
    m,n = y.shape
    p = m % 20
    zeros = np.zeros((p, n))
    y = np.concatenate((y, zeros), axis=0)
    series = []
    for i in range(int(len(y)/20)):
        yy = y[20*i: 20*(i+1), :]
        series.append(yy.reshape((n*20,1), order='C'))
    
    return series


def tensorfy(series):
    l = len(series[0])
    return [torch.from_numpy(n).float().view(l) for n in series]


def main(paths):
    lpc_train = '../dataset/features/LPC/train/lpc_train.pt'
    lpc_p_train = '../dataset/features/LPC_plus/train/lpc_p_train.pt'
    mfcc_train = '../dataset/features/MFCC/train/mfcc_train.pt'
    mfcc_p_train = '../dataset/features/MFCC_plus/train/mfcc_p_train.pt'
    mfe_train = '../dataset/features/MFE/train/mfe_train.pt'
    mfe_p_train = '../dataset/features/MFE_plus/train/mfe_p_train.pt'
    
    lpc_test = '../dataset/features/LPC/test/lpc_test.pt'
    lpc_p_test = '../dataset/features/LPC_plus/test/lpc_p_test.pt'
    mfcc_test = '../dataset/features/MFCC/test/mfcc_test.pt'
    mfcc_p_test = '../dataset/features/MFCC_plus/test/mfcc_p_test.pt'
    mfe_test = '../dataset/features/MFE/test/mfe_test.pt'
    mfe_p_test = '../dataset/features/MFE_plus/test/mfe_p_test.pt'
    
    classes = {'A': 0, 'E': 1, 'F': 2, 'L': 3, 'N': 4, 'W': 5, 'T': 6 }
    base = torch.tensor([0]*7).float()
    mfcc_ser, mfcc_p_ser, mfe_ser, mfe_p_ser, lpc_ser, lpc_p_ser = [], [], [], [], [], []
    random.shuffle(paths)
    for path in paths:
        answer = base.clone()
        answer[classes[path[-6]]] = 1
        y, sr = ls.load(path, sr=None)
        stacks = get_stacks(y)
        zc = zero_cross(y)
        mfe_feat, mfe_p_feat, energy, zc = mfe(y, zc)
        mfcc_feat = mfcc(y)
        mfcc_p_feat = mfcc_plus(mfcc_feat, energy, zc)
        lpc_feat = lpc(stacks)
        lpc_p_feat = lpc_plus(lpc_feat, energy, zc)
        mfe_ser.append((tensorfy(serialize(mfe_feat)), answer))
        mfe_p_ser.append((tensorfy(serialize(mfe_p_feat)), answer))
        mfcc_ser.append((tensorfy(serialize(mfcc_feat)), answer))
        mfcc_p_ser.append((tensorfy(serialize(mfcc_p_feat)), answer))
        lpc_ser.append((tensorfy(serialize(lpc_feat)), answer))
        lpc_p_ser.append((tensorfy(serialize(lpc_p_feat)), answer))

    torch.save(mfe_ser[:-50], mfe_train)
    torch.save(mfe_ser[-50:], mfe_test)

    torch.save(mfe_p_ser[:-50], mfe_p_train)
    torch.save(mfe_p_ser[-50:], mfe_p_test)

    torch.save(mfcc_ser[:-50], mfcc_train)
    torch.save(mfcc_ser[-50:], mfcc_test)

    torch.save(mfcc_p_ser[:-50], mfcc_p_train)
    torch.save(mfcc_p_ser[-50:], mfcc_p_test)

    torch.save(lpc_ser[:-50], lpc_train)
    torch.save(lpc_ser[-50:], lpc_test)

    torch.save(lpc_p_ser[:-50], lpc_p_train)
    torch.save(lpc_p_ser[-50:], lpc_p_test)
    
    return mfe_ser, mfe_p_ser, mfcc_ser, mfcc_p_ser, lpc_ser, lpc_p_ser        


    
    
    
    