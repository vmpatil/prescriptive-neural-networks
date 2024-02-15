#!/usr/bin/env python
# coding: utf-8

### Experiments for Prescriptive Neural Networks sovled by MIPs 
# *Real Data from the Hypertension Project*

# _Last Update: September 6th 2022_

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

def PrescriptiveNetwork_L0(data, covariates, treatment, L, K, λ, weight_bounds, epsilon, version, fold):
    
    if treatment == 'bpmed.dc':
        assert version=='v1'
    elif treatment == 'dcDayPP':
        assert version=='v2'
    elif treatment =='cross_treatment':
        assert version=='v3'

    x = torch.tensor(data[covariates].values.tolist())

    N = x.shape[0]                                        # Length of input data
    D = x[0].shape[0]                                     # Dimension of input data
    T = sorted(data[treatment].unique())                  # Number of treatments
    
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
        -(1/N)*sum( sum( (h[(n,t,L)])*(data['psi_'+treatment+'_'+str(t)].iloc[n]) for t in T) 
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
        
#     m.params.LogFile = os.path.join('Output',version,architecture,'grb_output_'+str(λ)+'_'+str(fold)+'.log')
    m.params.LogFile = os.path.join(version,architecture,'grb_output_'+str(λ)+'_'+str(fold)+'.log')    
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
    model_params.append(np.array(weights))
    model_params.append(np.array(biases))
    
    for l in range(1,L):
        weights = []
        biases = []
        for k in range(K):
            weights.append([output_dict['alpha '+str((k_prime,k,l))] for k_prime in range(K)])
            biases.append(output_dict['beta '+str((k,l))])
        model_params.append(np.array(weights))
        model_params.append(np.array(biases))
        
    weights = []
    biases = []
    for t in T:
        weights.append([output_dict['alpha '+str((k_prime,t,L))] for k_prime in range(K)])
        biases.append(output_dict['beta '+str((t,L))])
    model_params.append(np.array(weights))
    model_params.append(np.array(biases))
    
#     np.save(os.path.join('Output',version,architecture,'params_'+str(λ)+'_'+str(fold)+ '.npy'), np.array(model_params, dtype=list),allow_pickle=True)
    np.save(os.path.join(version,architecture,'params_'+str(λ)+'_'+str(fold)+ '.npy'), np.array(model_params, dtype=list),allow_pickle=True)
    
    return(m.objVal, m.MIPGap, m.Runtime, output_dict, model_params)

### Run Experiments ###

def run_experiment(L, K, weight_bounds, epsilon, version):
        
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
    
    T = sorted(df[treatment].unique())

    architecture = str(D) + ('-'+str(K))*L + '-' +str(len(T))
    print(architecture)
    
#     if not os.path.isdir(os.path.join('Output',version,architecture)):
#         os.mkdir(os.path.join('Output',version,architecture))
#         print('Architecture File Created')
#     else:
#         print('Architecture File Exists')

    if not os.path.isdir(os.path.join(version,architecture)):
        os.mkdir(os.path.join(version,architecture))
        print('Architecture File Created')
    else:
        print('Architecture File Exists')

        
    ### K-fold Cross Validation for Hyper Parameter Tuning ###
    
    num_folds = 10
    fold_idx = [fold*floor(len(df)/num_folds) for fold in range(num_folds)]+[len(df)]
    
    lambda_set = [0,0.01,0.1,1,10]
    pi_by_lambda = {}
    all_outcomes = {}
    
    gap = {}
    runtime = {}
    
    for λ in lambda_set:
        
        pi_by_lambda[λ] = []
        all_outcomes[λ] = []
        
        gap[λ] = []
        runtime[λ] = []
        
        for fold in range(len(fold_idx)-1):
                        
            train_set = df[0:fold_idx[fold]].append(df[fold_idx[fold+1]:]).copy(deep=True).reset_index(drop=True)
            test_set = df[fold_idx[fold]:fold_idx[fold+1]].copy(deep=True).reset_index(drop=True)

            print('-'*75)
            print('On Fold', fold+1)

            ### Training ###

            print('Training for λ =',λ)

            obj, mip_gap, mip_runtime, output, params = PrescriptiveNetwork_L0(train_set, covariates, treatment, L, K, λ, weight_bounds, epsilon, version, fold)
            
            gap[λ].append(mip_gap)
            runtime[λ].append(mip_runtime)
                    
            ### Testing ###

            T = sorted(df[treatment].unique())

            test_X = np.array(test_set[covariates])
        
            MIP_model_params = params

            N = test_X.shape[0]
            D = test_X.shape[1]

            outcome = []

            for n in range(N):

                h = (np.dot(MIP_model_params[0], test_X[n]) + MIP_model_params[1]).round(3)

                h[h < epsilon] = 0
                h[h >= epsilon] = 1

                k = 2
                for l in range(1,L):

                    h = (np.dot(MIP_model_params[k], h) + MIP_model_params[k+1]).round(3)

                    h[h < epsilon] = 0
                    h[h >= epsilon] = 1

                    k += 2

                h = (np.array(np.dot(MIP_model_params[k], h) + MIP_model_params[k+1])).round(3)

                h[h < epsilon] = 0
                h[h >= epsilon] = 1

                if version == 'v1':
                    outcome.append(sum(h[t]*test_set['psi_'+treatment+'_'+str(t)].iloc[n] for t in T))

                elif version == 'v2' or version == 'v3':
                    outcome.append(sum(h[t-1]*test_set['psi_'+treatment+'_'+str(t)].iloc[n] for t in T))

            pi_by_lambda[λ].append(sum(outcome)/len(outcome))
            all_outcomes[λ].extend(outcome)
    
            print('λ=',λ,'Fold '+str(fold+1)+', Testing π(s) = ', pi_by_lambda[λ][fold],'\n')
                                    
    best_λ = min([(key,np.array(pi_by_lambda[key]).mean()) for key in pi_by_lambda.keys()],key=lambda x : x[1])[0]

    print('**************Best λ =',str(best_λ)+'**************\n')
                          
    var = np.mean([all_outcomes[best_λ][i]**2 for i in range(len(all_outcomes[best_λ]))]) - (np.mean(all_outcomes[best_λ])**2)
    CI = [np.mean(all_outcomes[best_λ]) - st.norm.ppf(0.95)*(var/len(all_outcomes[best_λ]))**0.5,
          np.mean(all_outcomes[best_λ]) + st.norm.ppf(0.95)*(var/len(all_outcomes[best_λ]))**0.5]
    print('Testing π(s) with best λ = {}: {}| 95% Confidence Interval = {}'.format(best_λ,np.mean(all_outcomes[best_λ]),CI))
    
#     with open(os.path.join('Output',version,architecture,'outcomes.json'), 'w') as outfile:
#         json.dump(all_outcomes, outfile)    
#     with open(os.path.join('Output',version,architecture,'gap.json'), 'w') as outfile:
#         json.dump(gap, outfile)    
#     with open(os.path.join('Output',version,architecture,'runtime.json'), 'w') as outfile:
#         json.dump(runtime, outfile)
    with open(os.path.join(version,architecture,'outcomes.json'), 'w') as outfile:
        json.dump(all_outcomes, outfile)    
    with open(os.path.join(version,architecture,'gap.json'), 'w') as outfile:
        json.dump(gap, outfile)    
    with open(os.path.join(version,architecture,'runtime.json'), 'w') as outfile:
        json.dump(runtime, outfile)


    return None

### Run Experiments ###

# versions = ['v1','v2','v3']
versions = ['v3']

# if not os.path.isdir(os.path.join('Output')):
#     os.mkdir(os.path.join('Output'))
#     print('\nOutput File Created')
# else:
#     print('Output File Exists')

num_neurons = [3,10]

weight_bounds = [1,-1,1,-1]
epsilon = 0.01

L=1
    
for version in versions:

#     if not os.path.isdir(os.path.join('Output',version)):
#         os.mkdir(os.path.join('Output',version))
#         print('\nVersion File Created')
#     else:
#         print('Version File Exists')

    if not os.path.isdir(os.path.join(version)):
        os.mkdir(os.path.join(version))
        print('\nVersion File Created')
    else:
        print('Version File Exists')
        
    for K in num_neurons:
        run_experiment(L=1, K=K, weight_bounds=weight_bounds, epsilon=epsilon, version=version)
    