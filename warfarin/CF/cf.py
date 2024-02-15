#!/usr/bin/env python
# coding: utf-8

### Experiments for Causal Forest adapted from Wager, Athey (2017) and Athey, Imbens (2016)
# 
# _Reference: Wager, Athey (2017); Athey, Imbens (2016)_

### Packages ###

from econml.dml import CausalForestDML

import numpy as np
import pandas as pd
import os
import json
import time

from math import floor, ceil
import copy

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import scipy.stats as st

def run_experiment(seed1):

    oosp = {}
    gap = {}
    runtime = {}
    prescriptions = {}
        
    ### Parse Data ###
    
    dir_path = os.path.join('warfarin','warfarin')
    for seed_dir in os.listdir(os.path.join(dir_path)):
        if seed_dir.startswith('.'):
            continue
            
        seed2 = seed_dir[5:]
        
        oosp[seed2] = {}
        gap[seed2] = {}
        runtime[seed2] = {}
        prescriptions[seed2] = {}
        
        file_path = os.path.join(dir_path,seed_dir,'learned')
                
        for prob in [33,65,85]:
            
            file_name = 'warfarin_p={}.csv'.format(str(prob))

            df = pd.read_csv(os.path.join(file_path,file_name))
    
            print('\nOn Seed: '+str(seed1)+','+str(seed2))    
    
            train = df.sample(frac=.70, replace=False, weights=None, random_state=seed1, axis=None, ignore_index=False)
            test = df.iloc[df.index.difference(train.index)]    
            
            ### Define Parameters ###
    
            treatment = 'Random Treatment'
            covariates = list(df.columns[:75]) + list(df.columns[83:92])
                        
            N = len(df)             # Number of individuals in dataset
            D = len(covariates)     # Input data dimensions         

            all_treatment_effects = []

            ### Training ###

            X = train[covariates]
            Y = train['y']
            T = train[treatment]
    
            start_time = time.time()
    
            est = CausalForestDML(discrete_treatment=True, model_y = LogisticRegression(random_state=456),
                                  model_t = RandomForestClassifier(random_state=456), random_state=147)

            est.fit(Y, T, X=X)
    
            end_time = time.time()
    
            runtime[seed2][prob] = end_time - start_time
            gap[seed2][prob] = None
        
            ### Testing ###
            
            treatment_effects = {}
            for t0 in sorted(T.unique()):
                for t1 in sorted(T.unique()):

                    tau = est.effect(test[covariates], T0=t0,T1=t1)
                    treatment_effects[(t0,t1)] = tau

            all_treatment_effects.append(treatment_effects)

            mat = pd.DataFrame(treatment_effects)

            prescription_list = [int(treat[0]) for treat in mat.idxmin(axis=1)]            

            correct_pred = 0
            for i in range(len(test)):

                if prescription_list[i] == (test['Discrete Noisy Learned Dose']).iloc[i]:
                    correct_pred += 1

            percent_correct = (correct_pred/len(test))*100

            oosp[seed2][prob] = percent_correct
            prescriptions[seed2][prob] = prescription_list
     
            print('Out-of-Sample Probability for p={}, seed1={},seed2={}: {}%'.format(str(prob),str(seed1),str(seed2),str(percent_correct)))            

    with open(os.path.join('output',str(seed1),'oosp.json'), 'w') as outfile:
        json.dump(oosp, outfile)    
    with open(os.path.join('output',str(seed1),'gap.json'), 'w') as outfile:
        json.dump(gap, outfile)    
    with open(os.path.join('output',str(seed1),'runtime.json'), 'w') as outfile:
        json.dump(runtime, outfile)
    with open(os.path.join('output',str(seed1),'treatments.json'), 'w') as outfile:
        json.dump(prescriptions, outfile)        
                                
    return None

### Run Experiments ###

if not os.path.isdir(os.path.join('output')):
    os.mkdir(os.path.join('output'))
    print('\nOutput File Created')
else:
    print('Output File Exists')
    
for seed1 in [41,12,60,2,872]:
    
    if not os.path.isdir(os.path.join('output',str(seed1))):
        os.mkdir(os.path.join('output',str(seed1)))
        print('\nSeed File Created')
    else:
        print('Seed File Exists')
                                
    run_experiment(seed1)
