#!/usr/bin/env python
# coding: utf-8

## Experiments for Prescriptive Neural Networks solved by MIPs 
# *Real Data from the IWPC*

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from math import floor, ceil
import pandas as pd
import os
import json
import re
import torch

import scipy.stats as st

### MIP-NN ###

def PrescriptiveNetwork_L0(data, covariates, treatment, L, K, T, λ, weight_bounds, epsilon, seed1, seed2, prob, fold):    

    x = torch.tensor(data[covariates].values.tolist())

    N = x.shape[0]                                        # Length of input data
    D = x[0].shape[0]                                     # Dimension of input data
    
    w_ub, w_lb, b_ub, b_lb = weight_bounds
    
    architecture = str(D) + ('-'+str(K))*L + '-' +str(len(T))
    print('\nNeural Network Architecture:',architecture)
    
    print('λ:',λ)    
    print('N:',N)
        
    # Create a new model
    m = gp.Model("MIP for Prescriptive Networks")
    
    m.setParam('TimeLimit', 3600) # time limit of 3600s
    m.setParam('MIPGap', 0.1/100) # gap limit of 0.1%

    # Create variables
    
    alpha = {}
    alpha_zero = {}
    beta = {}
    beta_zero = {}
    h = {}
    z = {}

    for k in range(K):
        for t in T:
            alpha[(k,t,L)] = m.addVar(lb=w_lb, ub=w_ub, vtype=GRB.CONTINUOUS, name="alpha "+str((k,t,L)))
            
        for d in range(D):
            
            alpha[(d,k,0)] = m.addVar(lb=w_lb, ub=w_ub, vtype=GRB.CONTINUOUS, name="alpha "+str((d,k,0)))
            
        for k_prime in range(K):
            for l in range(1,L):
                alpha[(k_prime,k,l)] = m.addVar(lb=w_lb, ub=w_ub, vtype=GRB.CONTINUOUS, name="alpha "+str((k_prime,k,l)))
    
        for l in range(0,L):
            beta[(k,l)] = m.addVar(lb=b_lb, ub=b_ub, vtype=GRB.CONTINUOUS, name="beta "+str((k,l))) 
    
    for t in T:
        beta[(t,L)] = m.addVar(lb=b_lb, ub=b_ub, vtype=GRB.CONTINUOUS, name="beta "+str((t,L)))
        
    for k in range(K):
        for t in T:
            alpha_zero[(k,t,L)] = m.addVar(vtype=GRB.BINARY, name="alpha_zero "+str((k,t,L)))
            
        for d in range(D):
            alpha_zero[(d,k,0)] = m.addVar(vtype=GRB.BINARY, name="alpha_zero "+str((d,k,0)))
            
        for k_prime in range(K):
            for l in range(1,L):
                alpha_zero[(k_prime,k,l)] = m.addVar(vtype=GRB.BINARY, name="alpha_zero "+str((k_prime,k,l)))
    
        for l in range(0,L):
            beta_zero[(k,l)] = m.addVar(vtype=GRB.BINARY, name="beta_zero "+str((k,l))) 
    
    for t in T:
        beta_zero[(t,L)] = m.addVar(vtype=GRB.BINARY, name="beta_zero "+str((t,L)))
    
    for n in range(N):
        for t in T:
            h[(n,t,L)] = m.addVar(lb=-np.inf, vtype=GRB.BINARY, name="h "+str((n,t,L)))
            
        for k in range(K):
            for l in range(0,L):
                h[(n,k,l)] = m.addVar(vtype=GRB.BINARY, name="h "+str((n,k,l)))
                                
            for k_prime in range(K):
                for l in range(1,L):
                    z[(n,k_prime,k,l)] = m.addVar(lb=w_lb, ub=w_ub, vtype=GRB.CONTINUOUS, name= "z "+str((n,k_prime,k,l))) 
        
            for t in T:
                z[(n,k,t,L)] = m.addVar(lb=w_lb, ub=w_ub, vtype=GRB.CONTINUOUS, name= "z "+str((n,k,t,L)))

    # Set objective
    m.setObjective(
        (1/N)*sum( sum( (h[(n,t,L)])*(data['psi_'+str(t)].iloc[n]) for t in T) 
                  for n in range(N)) 
        + λ*(sum( sum(alpha_zero[(d,k,0)] for k in range(K)) for d in range(D)) 
             + sum( sum( sum(alpha_zero[(k_prime,k,l)] for k in range(K)) for k_prime in range(K)) for l in range(1,L))
             + sum( sum(alpha_zero[(k,t,L)] for t in T) for k in range(K))
             + sum( sum(beta_zero[(k,l)] for l in range(0,L)) for k in range(K))
             + sum( beta_zero[(t,L)] for t in T)), GRB.MAXIMIZE)
    
    # Add constraints
    
    data_max = max((data[covariates].min().abs().max(),data[covariates].max().abs().max()))
    
    for n in range(N):
        
        for k in range(K):
            
            M = (D*w_ub*data_max)+b_ub # initiating M for the 0th layer
            
            m.addConstr(sum(alpha[(d,k,0)]*x[n][d] for d in range(D)) + beta[(k,0)] 
                        <= (M + epsilon)*h[(n,k,0)], name="C1 Binary Neuron "+str((n,k,0)))
            
            m.addConstr(sum(alpha[(d,k,0)]*x[n][d] for d in range(D)) + beta[(k,0)] 
                        >= epsilon + (-M - epsilon)*(1-h[(n,k,0)]), name="C2 Binary Neuron "+str((n,k,0)))  

            for l in range(1,L):
                
                M = (K*w_ub*1)+b_ub # updating M at each layer
                
                m.addConstr(sum(z[(n,k_prime,k,l)] for k_prime in range(K)) + beta[(k,l)] 
                            <= (M + epsilon)*h[(n,k,l)], name="C1 Binary Neuron "+str((n,k,l)))
            
                m.addConstr(sum(z[(n,k_prime,k,l)] for k_prime in range(K)) + beta[(k,l)] 
                            >= epsilon + (-M - epsilon)*(1-h[(n,k,l)]), name="C2 Binary Neuron "+str((n,k,l)))
                
        for k_prime in range(K):
            for k in range(K):
                for l in range(1,L):
                    m.addConstr(z[(n,k_prime,k,l)] <= alpha[(k_prime,k,l)] + (w_ub-w_lb)*(1.0-h[(n,k_prime,l-1)]), name="z-alpha UB "+str((n,k_prime,k,l))) 
                    m.addConstr(z[(n,k_prime,k,l)] >= alpha[(k_prime,k,l)] + (w_lb-w_ub)*(1.0-h[(n,k_prime,l-1)]), name="z-alpha LB "+str((n,k_prime,k,l))) 
                    m.addConstr(z[(n,k_prime,k,l)] <= (w_ub)*h[(n,k_prime,l-1)], name="z-h UB "+str((n,k,k_prime,l)))
                    m.addConstr(z[(n,k_prime,k,l)] >= (w_lb)*h[(n,k_prime,l-1)], name="z-h LB "+str((n,k,k_prime,l)))
            for t in T:
                m.addConstr(z[(n,k_prime,t,L)] <= alpha[(k_prime,t,L)] + (w_ub-w_lb)*(1.0-h[(n,k_prime,L-1)]), name="z-alpha UB "+str((n,k_prime,t,L)))
                m.addConstr(z[(n,k_prime,t,L)] >= alpha[(k_prime,t,L)] + (w_lb-w_ub)*(1.0-h[(n,k_prime,L-1)]), name="z-alpha LB "+str((n,k_prime,t,L)))
                m.addConstr(z[(n,k_prime,t,L)] <= (w_ub)*h[(n,k_prime,L-1)], name="z-h UB "+str((n,k_prime,t,L)))
                m.addConstr(z[(n,k_prime,t,L)] >= (w_lb)*h[(n,k_prime,L-1)], name="z-h LB "+str((n,k_prime,t,L)))
        
        M = (K*w_ub*1)+b_ub # updating M for the last layer
        
        for t in T:
            
            m.addConstr(sum(z[(n,k_prime,t,L)] for k_prime in range(K)) + beta[(t,L)]
                        <= (M + epsilon)*h[(n,t,L)], name='C1 Output '+str((n,t)))
            m.addConstr(sum(z[(n,k_prime,t,L)] for k_prime in range(K)) + beta[(t,L)]
                        >= epsilon + (-M - epsilon)*(1-h[(n,t,L)]), name='C2 Output '+str((n,t)))                                        

        m.addConstr(sum(h[(n,t,L)] for t in T) == 1, name='One Treatment '+str((n)))
        
    for k in range(K):
        for d in range(D):
            m.addConstr(alpha[(d,k,0)] <= (w_ub)*(1-alpha_zero[(d,k,0)]), name='C1 alpha_zero '+str((d,k,0)))
            m.addConstr(alpha[(d,k,0)] >= (w_lb)*(1-alpha_zero[(d,k,0)]), name='C2 alpha_zero '+str((d,k,0)))            
            
        for k_prime in range(K):
            for l in range(1,L):
                m.addConstr(alpha[(k_prime,k,l)] <= (w_ub)*(1-alpha_zero[(k_prime,k,l)]), name='C1 alpha_zero '+str((k_prime,k,l)))
                m.addConstr(alpha[(k_prime,k,l)] >= (w_lb)*(1-alpha_zero[(k_prime,k,l)]), name='C2 alpha_zero '+str((k_prime,k,l)))            
        
        for t in T:
            m.addConstr(alpha[(k,t,L)] <= (w_ub)*(1-alpha_zero[(k,t,L)]), name='C1 alpha_zero '+str((k,t,L)))
            m.addConstr(alpha[(k,t,L)] >= (w_lb)*(1-alpha_zero[(k,t,L)]), name='C2 alpha_zero '+str((k,t,L)))            
            
        for l in range(0,L):
            m.addConstr(beta[(k,l)] <= (b_ub)*(1-beta_zero[(k,l)]), name='C1 beta_zero '+str((k,l)))
            m.addConstr(beta[(k,l)] >= (b_lb)*(1-beta_zero[(k,l)]), name='C2 beta_zero '+str((k,l)))

    for t in T:
        m.addConstr(beta[(t,L)] <= (b_ub)*(1-beta_zero[(t,L)]), name='C1 beta_zero '+str((t,L)))
        m.addConstr(beta[(t,L)] >= (b_lb)*(1-beta_zero[(t,L)]), name='C2 beta_zero '+str((t,L)))        
                            
    # Optimize model
    
    if fold != None:
        m.params.LogFile = os.path.join('output',architecture,'grb_output_'+str(seed2)+'_'+str(prob)+'_'+str(λ)+'_'+str(fold)+'.log')    
    else:
        m.params.LogFile = os.path.join('output',architecture,'final_grb_output_'+str(seed2)+'_'+str(prob)+'.log')    
        
    m.optimize()
    
    output_dict = {}
    
    for v in m.getVars():
        output_dict[v.varName] = v.x

    print('Obj: %g' % m.objVal)
    
    model_params = []
    weights = []
    biases = []
    for k in range(K):
        weights.append([output_dict['alpha '+str((d,k,0))] for d in range(D)])
        biases.append(output_dict['beta '+str((k,0))])
    model_params.append(np.array(weights).T)
    model_params.append(np.array(biases))
    
    for l in range(1,L):
        weights = []
        biases = []
        for k in range(K):
            weights.append([output_dict['alpha '+str((k_prime,k,l))] for k_prime in range(K)])
            biases.append(output_dict['beta '+str((k,l))])
        model_params.append(np.array(weights).T)
        model_params.append(np.array(biases))
        
    weights = []
    biases = []
    for t in T:
        weights.append([output_dict['alpha '+str((k_prime,t,L))] for k_prime in range(K)])
        biases.append(output_dict['beta '+str((t,L))])
    model_params.append(np.array(weights).T)
    model_params.append(np.array(biases))
        
    if fold != None:    
        np.save(os.path.join('output',architecture,'params_'+str(seed2)+'_'+str(prob)+'_'+str(λ)+'_'+str(fold)+ '.npy'), np.array(model_params, dtype=list),allow_pickle=True)
    else:
        np.save(os.path.join('output',architecture,'final_params'+str(seed2)+'_'+str(prob)+'.npy'), np.array(model_params, dtype=list),allow_pickle=True)
        
    return(m.objVal, m.MIPGap, m.Runtime, output_dict, model_params)

### Run Experiments ###

def run_experiment(L, K, weight_bounds, epsilon, seed1):
    
    oosp = {}
    gap = {}
    runtime = {}
    prescriptions = {}    
        
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
            covariates = list(df.columns[:76])
            
            N = len(df)             # Number of individuals in dataset
            D = len(covariates)     # Input data dimensions         
    
            T = sorted(df[treatment].unique())

            architecture = str(D) + ('-'+str(K))*L + '-' +str(len(T))
            print(architecture)
    
            if not os.path.isdir(os.path.join('output',architecture)):
                os.mkdir(os.path.join('output',architecture))
                print('Architecture File Created')
            else:
                print('Architecture File Exists')

            ### K-fold Cross Validation for Hyper Parameter Tuning ###
    
            num_folds = 5
            fold_idx = [fold*floor(len(train)/num_folds) for fold in range(num_folds)]+[len(train)]
    
            lambda_set = [0,0.1,1]
            performance_by_lambda = {}
        
            for λ in lambda_set:
        
                performance_by_lambda[λ] = []
                
                for fold in range(len(fold_idx)-1):
                        
                    train_set = train[0:fold_idx[fold]].append(train[fold_idx[fold+1]:]).copy(deep=True).reset_index(drop=True)
                    test_set = train[fold_idx[fold]:fold_idx[fold+1]].copy(deep=True).reset_index(drop=True)

                    print('-'*75)
                    print('On Fold', fold+1)

                    ### Training ###

                    print('Training for λ =',λ)

                    obj, mip_gap, mip_runtime, output, params = PrescriptiveNetwork_L0(train_set, covariates, treatment, L, K, T, λ, weight_bounds, epsilon, seed1, seed2, prob, fold)
                    
                    ### Testing ###

                    T = sorted(df[treatment].unique())

                    test_X = np.array(test_set[covariates])
        
                    MIP_model_params = params

                    N = test_X.shape[0]
                    D = test_X.shape[1]

                    prescription_list = []
                    
                    correct_pred = 0
                    
                    for n in range(N):

                        h = (np.dot(MIP_model_params[0].T, test_X[n]) + MIP_model_params[1]).round(3)

                        h[h < epsilon] = 0
                        h[h >= epsilon] = 1

                        k = 2
                        for l in range(1,L):

                            h = (np.dot(MIP_model_params[k].T, h) + MIP_model_params[k+1]).round(3)

                            h[h < epsilon] = 0
                            h[h >= epsilon] = 1

                            k += 2

                        h = (np.array(np.dot(MIP_model_params[k].T, h) + MIP_model_params[k+1])).round(3)

                        h[h < epsilon] = 0
                        h[h >= epsilon] = 1
                        
                        pred = np.argmax(h)

                        prescription_list.append(pred)
                        
                        if pred == (test_set['Discrete Noisy Learned Dose']).iloc[n]:
                            correct_pred += 1

                    performance_by_lambda[λ].append((correct_pred/N)*100)
    
                    print('λ=',λ,'Fold '+str(fold+1)+', OOSP = ', performance_by_lambda[λ][fold],'%\n')
                                                
            best_λ = max([(key,np.array(performance_by_lambda[key]).mean()) for key in performance_by_lambda.keys()],key=lambda x : x[1])[0]

            print('**************Best λ =',str(best_λ)+'**************\n')
    
            ### Use best_λ and run on full trainset ###
    
            print('Final Training for λ =',best_λ)

            obj, mip_gap, mip_runtime, output, params = PrescriptiveNetwork_L0(train, covariates, treatment, L, K, T, best_λ, weight_bounds, epsilon, seed1, seed2, prob, None)
    
            gap[seed2][prob] = mip_gap*100
            runtime[seed2][prob] = mip_runtime

            ### Testing ###

            T = sorted(df[treatment].unique())

            test_X = np.array(test[covariates])

            MIP_model_params = params

            N = test_X.shape[0]
            D = test_X.shape[1]

            prescription_list = []
            
            correct_pred = 0

            for n in range(N):

                h = (np.dot(MIP_model_params[0].T, test_X[n]) + MIP_model_params[1]).round(3)

                h[h < epsilon] = 0
                h[h >= epsilon] = 1

                k = 2
                for l in range(1,L):

                    h = (np.dot(MIP_model_params[k].T, h) + MIP_model_params[k+1]).round(3)

                    h[h < epsilon] = 0
                    h[h >= epsilon] = 1

                    k += 2

                h = (np.array(np.dot(MIP_model_params[k].T, h) + MIP_model_params[k+1])).round(3)

                h[h < epsilon] = 0
                h[h >= epsilon] = 1
        
                pred = np.argmax(h)
                
                prescription_list.append(int(pred))
                
                if pred == (test['Discrete Noisy Learned Dose']).iloc[n]:
                    correct_pred += 1
                        
            percent_correct = (correct_pred/N)*100

            oosp[seed2][prob] = percent_correct
            prescriptions[seed2][prob] = prescription_list
        
            print('Out-of-Sample Probability for architecture={}, p={}, seed={}: {}%'.format(str(architecture),str(prob),str(seed2),str(percent_correct)))
    
    with open(os.path.join('output',architecture,'oosp.json'), 'w') as outfile:
        json.dump(oosp, outfile)    
    with open(os.path.join('output',architecture,'gap.json'), 'w') as outfile:
        json.dump(gap, outfile)    
    with open(os.path.join('output',architecture,'runtime.json'), 'w') as outfile:
        json.dump(runtime, outfile)
    with open(os.path.join('output',architecture,'treatments.json'), 'w') as outfile:
        json.dump(prescriptions, outfile)

    return None

### Run Experiments ###

num_neurons = [3]

weight_bounds = [1,-1,1,-1]
epsilon = 0.01

L=1
    
for seed1 in [60]:
# for seed in [41,12,60,2,872]:
    for K in num_neurons:
        if not os.path.isdir(os.path.join('output')):
            os.mkdir(os.path.join('output'))
            print('\nOutput Seed File Created')
        else:
            print('Output Seed File Exists')
        run_experiment(L, K, weight_bounds, epsilon, seed1)
    