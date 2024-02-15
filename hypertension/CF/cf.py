#!/usr/bin/env python
# coding: utf-8

### Experiments for Causal Forest adapted from Wager, Athey (2017) and Athey, Imbens (2016)
# 
# _Reference: Wager, Athey (2017); Athey, Imbens (2016)_

# _Last Updated: 15th August 2022_

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
from sklearn.linear_model import ElasticNet

import scipy.stats as st


def run_experiment(version, seed=123):
        
    ### Parse Data ###
    
    file_path = os.path.join('hypertension','hypertension_learned')
    file_name = 'hypertension_'+version+'.csv'

    df = pd.read_csv(os.path.join(file_path,file_name))
    
    print('Training Version:',version)
    
    ### Define Parameters ###

    if version=='v1':

        treatment = 'bpmed.dc'        
        covariates = list(df.iloc[:,3:25].columns)

        N = len(df)             # Number of individuals in dataset
        D = len(covariates)     # Input data dimensions 

    elif version=='v2':

        treatment = 'dcDayPP'
        covariates = list(df.iloc[:,3:22].columns)

        N = len(df)             # Number of individuals in dataset
        D = len(covariates)     # Input data dimensions 

    elif version=='v3':

        treatment = 'cross_treatment'
        covariates = list(df.iloc[:,4:22].columns)

        N = len(df)             # Number of individuals in dataset
        D = len(covariates)     # Input data dimensions

    ### K-fold Cross Validation ###

    num_folds = 10
    fold_idx = [fold*floor(len(df)/num_folds) for fold in range(num_folds)]+[len(df)]

    performance = []
    all_outcomes = []
    
    gap = {}
    runtime = {}

    all_treatment_effects = []
    for fold in range(len(fold_idx)-1):

        train_set = df[0:fold_idx[fold]].append(df[fold_idx[fold+1]:]).copy(deep=True).reset_index(drop=True)
        test_set = df[fold_idx[fold]:fold_idx[fold+1]].copy(deep=True).reset_index(drop=True)

#         print('-'*75)
#         print('On Fold', fold+1)

        ### Training ###

        X = train_set[covariates]
        Y = train_set['max.sbp']
        T = train_set[treatment]
        
        start_time = time.time()
        
        est = CausalForestDML(discrete_treatment=True, model_y = ElasticNet(random_state=456),
                              model_t = LogisticRegression(random_state=456), random_state=seed)

        est.fit(Y, T, X=X)
        
        end_time = time.time()
        
        runtime[fold] = end_time - start_time
        gap[fold] = None
        
        treatment_effects = {}
        for t0 in sorted(T.unique()):
            for t1 in sorted(T.unique()):

                tau = est.effect(test_set[covariates], T0=t0,T1=t1)
                treatment_effects[(t0,t1)] = tau

        all_treatment_effects.append(treatment_effects)

        mat = pd.DataFrame(treatment_effects)

        prescriptions = [treat[0] for treat in mat.idxmax(axis=1)]

        ### Testing ###

        T = sorted(df[treatment].unique())

        outcome = []
        for i in range(len(test_set)):

            outcome.append(test_set['psi_'+treatment+'_'+str(prescriptions[i])].iloc[i])

        performance.append(sum(outcome)/len(outcome))
        all_outcomes.extend(outcome)

        print('Fold '+str(fold+1)+', Testing π(s) = ', performance[fold],'\n')
     
    var = np.mean([all_outcomes[i]**2 for i in range(len(all_outcomes))]) - (np.mean(all_outcomes)**2)
    CI = [np.mean(all_outcomes) - st.norm.ppf(0.95)*(var/len(all_outcomes))**0.5,
          np.mean(all_outcomes) + st.norm.ppf(0.95)*(var/len(all_outcomes))**0.5]
    print('\nTesting π(s) : {}| 95% Confidence Interval = {}'.format(np.mean(all_outcomes),CI))        

    with open(os.path.join('Output',version,'outcomes.json'), 'w') as outfile:
        json.dump(all_outcomes, outfile)    
    with open(os.path.join('Output',version,'gap.json'), 'w') as outfile:
        json.dump(gap, outfile)    
    with open(os.path.join('Output',version,'runtime.json'), 'w') as outfile:
        json.dump(runtime, outfile)
                                
    return(performance, all_outcomes)

### Run Experiments ###

versions = ['v1','v2','v3']

if not os.path.isdir(os.path.join('Output')):
    os.mkdir(os.path.join('Output'))
    print('\nOutput File Created')
else:
    print('Output File Exists')

for version in versions:

    if not os.path.isdir(os.path.join('Output',version)):
        os.mkdir(os.path.join('Output',version))
        print('\nVersion File Created')
    else:
        print('Version File Exists')

                            
    run_experiment(version)
