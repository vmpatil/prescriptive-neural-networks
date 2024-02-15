#!/usr/bin/env python
# coding: utf-8

### Experiments for Causal Forest adapted from Wager, Athey (2017) and Athey, Imbens (2016)
# 
# _Reference: Wager, Athey (2017); Athey, Imbens (2016)_

# _Last Updated: 15th August 2022_

### Packages ###

from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV

import numpy as np
import pandas as pd
import os
import json
import time
import sys

#### Run Experiments ####

datatypes = ['1_Easy','1_Hard','2_Easy','3_Easy']
train_lens = [100,500]
datasets = [1, 2, 3, 4, 5]
probs = [0.1, 0.25, 0.5, 0.75, 0.9]

seed = 123
        
for datatype in datatypes:        
    if not os.path.isdir(datatype):
        os.mkdir(datatype)
    else:
        print('Datatype File Exists')
        
    if datatype == '1_Easy':
        features = ['V1','V2']
        prefix = '1e'
    elif datatype == '1_Hard':
        features = ['V1','V2']
        prefix = '1h'
    elif datatype == '2_Easy':
        features = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10']
        prefix = '2e'
    elif datatype == '3_Easy':
        features = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20']
        prefix = '3e'

    for N in train_lens:
        if not os.path.isdir(os.path.join(datatype, 'Output_'+str(N))):
            os.mkdir(os.path.join(datatype, 'Output_'+str(N)))
        else:
            print('Output File Exists')
            
        oosp = {}
        gap = {}
        runtime = {}
        for prob in probs:
            oosp[prob] = {}
            gap[prob] = {}
            runtime[prob] = {}
            for dataset in datasets:
                file_path = prefix+'_athey_'+str(N)
                train_file_name = 'data_train_' + str(prob) + '_' + str(dataset) + '.csv'
                test_file_name = 'data_test_' + str(prob) + '_' + str(dataset) + '.csv'

                train_df = pd.read_csv(os.path.join(file_path, train_file_name))
                test_df = pd.read_csv(os.path.join(file_path, test_file_name))
            
                X = train_df[features]
                Y = train_df['y']
                T = train_df['t']

                X_test = test_df[features]

                print('\nStarting Training '+ datatype + '_' + str(N) + '-' + str(prob) + '_' + str(dataset) + '...')
                start_time = time.time()
                est = CausalForestDML(discrete_treatment=True, random_state=seed)

                est.fit(Y, T, X=X)
                treatment_effects = est.effect(X_test)
            
                end_time = time.time()
            
                # Confidence intervals via Bootstrap-of-Little-Bags for forests
                lb, ub = est.effect_interval(X_test, alpha=0.05)

                treatments = np.array([None for i in range(len(test_df))])
                treatments[treatment_effects <= 0] = 0
                treatments[treatment_effects > 0] = 1
        
                ### Testing ###
        
                test_df['best_t'] = [1 if test_df['y1'][n] > test_df['y0'][n] else 0 for n in range(len(test_df))]

                correct_pred = 0
                for i in range(len(treatments)):
                    if test_df['best_t'][i] == treatments[i]:
                        correct_pred += 1
        
                testing_acc = correct_pred/(len(treatments))*100
                print('Out of Sample Probability for', datatype+'_'+str(prob)+'_'+str(dataset)+'_'+str(N),'=', testing_acc, '%\n')
        
                oosp[prob][dataset] = testing_acc
                gap[prob][dataset] = None
                runtime[prob][dataset] = end_time - start_time
            
            avg_testing_acc = sum(oosp[prob][dataset] for dataset in datasets)/len(datasets)
            print('Average Testing Accuracy for ' + str(prob) + '=', avg_testing_acc, '%\n')
    
        with open(os.path.join(datatype, 'Output_'+str(N),'oosp.json'), 'w') as outfile:
            json.dump(oosp, outfile)
        with open(os.path.join(datatype, 'Output_'+str(N),'gap.json'), 'w') as outfile:
            json.dump(gap, outfile)    
        with open(os.path.join(datatype, 'Output_'+str(N),'runtime.json'), 'w') as outfile:
            json.dump(runtime, outfile)