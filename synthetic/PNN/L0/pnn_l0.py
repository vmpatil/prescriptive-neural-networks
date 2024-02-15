#!/usr/bin/env python
# coding: utf-8

### Experiments for Prescriptive Neural Networks sovled by MIPs 

# _Last Update: August 28th 2022_

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import os
import json
import re
import torch

### MIP-NN ###

def PrescriptiveNetwork_L0(data,D,K,λ): 
    
    x = torch.tensor(data.iloc[:,:D].values.tolist())

    N = x.shape[0]                                        # Length of input data
    T = len(data['t'].unique())                           # Number of treatments
    
    print('λ:',λ)
    
    print(datatype, N, D, K, T)
    
    # Establish the baseline policy
    
    # If baseline is the policy in data
    
#     baseline = []
#     target = np.array(df['t'])
#     for i in range(len(target)):
#         lr = np.arange(T)
#         one_hot = (lr==target[i]).astype(np.int)
#         baseline.append(one_hot)        
#     baseline = torch.tensor(baseline)
#     filename = 'historic_baseline_'
        
    # If baseline is no treatments
        
    baseline = []
    target = np.array([0 for i in range(len(df['t']))])
    for i in range(len(target)):
        lr = np.arange(T)
        one_hot = (lr==target[i]).astype(np.int)
        baseline.append(one_hot)
    baseline = torch.tensor(baseline)

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
        for t in range(T):
            alpha[(k,t,L)] = m.addVar(lb=w_lb, ub=w_ub, vtype=GRB.CONTINUOUS, name="alpha "+str((k,t,L)))
            
        for d in range(D):
            
            alpha[(d,k,0)] = m.addVar(lb=w_lb, ub=w_ub, vtype=GRB.CONTINUOUS, name="alpha "+str((d,k,0)))
            
        for k_prime in range(K):
            for l in range(1,L):
                alpha[(k_prime,k,l)] = m.addVar(lb=w_lb, ub=w_ub, vtype=GRB.CONTINUOUS, name="alpha "+str((k_prime,k,l)))
    
        for l in range(0,L):
            beta[(k,l)] = m.addVar(lb=b_lb, ub=b_ub, vtype=GRB.CONTINUOUS, name="beta "+str((k,l))) 
    
    for t in range(T):
        beta[(t,L)] = m.addVar(lb=b_lb, ub=b_ub, vtype=GRB.CONTINUOUS, name="beta "+str((t,L)))
        
    for k in range(K):
        for t in range(T):
            alpha_zero[(k,t,L)] = m.addVar(vtype=GRB.BINARY, name="alpha_zero "+str((k,t,L)))
            
        for d in range(D):
            alpha_zero[(d,k,0)] = m.addVar(vtype=GRB.BINARY, name="alpha_zero "+str((d,k,0)))
            
        for k_prime in range(K):
            for l in range(1,L):
                alpha_zero[(k_prime,k,l)] = m.addVar(vtype=GRB.BINARY, name="alpha_zero "+str((k_prime,k,l)))
    
        for l in range(0,L):
            beta_zero[(k,l)] = m.addVar(vtype=GRB.BINARY, name="beta_zero "+str((k,l))) 
    
    for t in range(T):
        beta_zero[(t,L)] = m.addVar(vtype=GRB.BINARY, name="beta_zero "+str((t,L)))
    
    for n in range(N):
        for t in range(T):
            h[(n,t,L)] = m.addVar(lb=-np.inf, vtype=GRB.BINARY, name="h "+str((n,t,L)))                    
            
        for k in range(K):
            for l in range(0,L):
                h[(n,k,l)] = m.addVar(vtype=GRB.BINARY, name="h "+str((n,k,l)))
                                
            for k_prime in range(K):
                for l in range(1,L):
                    z[(n,k_prime,k,l)] = m.addVar(lb=w_lb, ub=w_ub, vtype=GRB.CONTINUOUS, name= "z "+str((n,k_prime,k,l))) 
        
            for t in range(T):
                z[(n,k,t,L)] = m.addVar(lb=w_lb, ub=w_ub, vtype=GRB.CONTINUOUS, name= "z "+str((n,k,t,L)))        

    # Set objective
    m.setObjective(
        (1/N)*sum( sum( (h[(n,t,L)] - baseline[n][t])*(data['psi'+str(t)][n]) for t in range(T)) 
                  for n in range(N)) 
        + λ*(sum( sum(alpha_zero[(d,k,0)] for k in range(K)) for d in range(D)) 
             + sum( sum( sum(alpha_zero[(k_prime,k,l)] for k in range(K)) for k_prime in range(K)) for l in range(1,L))
             + sum( sum(alpha_zero[(k,t,L)] for t in range(T)) for k in range(K))
             + sum( sum(beta_zero[(k,l)] for l in range(0,L)) for k in range(K))
             + sum( beta_zero[(t,L)] for t in range(T))), GRB.MAXIMIZE)
    
    # Add constraints
    
    data_max = max((data.iloc[:,:D].min().abs().max(),data.iloc[:,:D].max().abs().max()))
    
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
            for t in range(T):
                m.addConstr(z[(n,k_prime,t,L)] <= alpha[(k_prime,t,L)] + (w_ub-w_lb)*(1.0-h[(n,k_prime,L-1)]), name="z-alpha UB "+str((n,k_prime,t,L)))
                m.addConstr(z[(n,k_prime,t,L)] >= alpha[(k_prime,t,L)] + (w_lb-w_ub)*(1.0-h[(n,k_prime,L-1)]), name="z-alpha LB "+str((n,k_prime,t,L)))
                m.addConstr(z[(n,k_prime,t,L)] <= (w_ub)*h[(n,k_prime,L-1)], name="z-h UB "+str((n,k_prime,t,L)))
                m.addConstr(z[(n,k_prime,t,L)] >= (w_lb)*h[(n,k_prime,L-1)], name="z-h LB "+str((n,k_prime,t,L)))
        
        M = (K*w_ub*1)+b_ub # updating M for the last layer
        
        for t in range(T):
            
            m.addConstr(sum(z[(n,k_prime,t,L)] for k_prime in range(K)) + beta[(t,L)]
                        <= (M + epsilon)*h[(n,t,L)], name='C1 Output '+str((n,t)))
            m.addConstr(sum(z[(n,k_prime,t,L)] for k_prime in range(K)) + beta[(t,L)]
                        >= epsilon + (-M - epsilon)*(1-h[(n,t,L)]), name='C2 Output '+str((n,t)))                                        

        m.addConstr(sum(h[(n,t,L)] for t in range(T)) == 1, name='One Treatment '+str((n)))
        
    for k in range(K):
        for d in range(D):
            m.addConstr(alpha[(d,k,0)] <= (w_ub)*(1-alpha_zero[(d,k,0)]), name='C1 alpha_zero '+str((d,k,0)))
            m.addConstr(alpha[(d,k,0)] >= (w_lb)*(1-alpha_zero[(d,k,0)]), name='C2 alpha_zero '+str((d,k,0)))            
            
        for k_prime in range(K):
            for l in range(1,L):
                m.addConstr(alpha[(k_prime,k,l)] <= (w_ub)*(1-alpha_zero[(k_prime,k,l)]), name='C1 alpha_zero '+str((k_prime,k,l)))
                m.addConstr(alpha[(k_prime,k,l)] >= (w_lb)*(1-alpha_zero[(k_prime,k,l)]), name='C2 alpha_zero '+str((k_prime,k,l)))            
        
        for t in range(T):
            m.addConstr(alpha[(k,t,L)] <= (w_ub)*(1-alpha_zero[(k,t,L)]), name='C1 alpha_zero '+str((k,t,L)))
            m.addConstr(alpha[(k,t,L)] >= (w_lb)*(1-alpha_zero[(k,t,L)]), name='C2 alpha_zero '+str((k,t,L)))            
            
        for l in range(0,L):
            m.addConstr(beta[(k,l)] <= (b_ub)*(1-beta_zero[(k,l)]), name='C1 beta_zero '+str((k,l)))
            m.addConstr(beta[(k,l)] >= (b_lb)*(1-beta_zero[(k,l)]), name='C2 beta_zero '+str((k,l)))

    for t in range(T):
        m.addConstr(beta[(t,L)] <= (b_ub)*(1-beta_zero[(t,L)]), name='C1 beta_zero '+str((t,L)))
        m.addConstr(beta[(t,L)] >= (b_lb)*(1-beta_zero[(t,L)]), name='C2 beta_zero '+str((t,L)))        
                            
    # Optimize model
    
    m.params.LogFile = os.path.join(datatype,architecture,'Output_'+str(N),'lambda_'+str(λ),'grb_output_' + str(prob) + '_' + str(dataset) + '.log')
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
    for t in range(T):
        weights.append([output_dict['alpha '+str((k_prime,t,L))] for k_prime in range(K)])
        biases.append(output_dict['beta '+str((t,L))])
    model_params.append(np.array(weights))
    model_params.append(np.array(biases))
    
    np.save(os.path.join(datatype,architecture,'Output_'+str(N),'lambda_'+str(λ), 'params_' + str(prob) + '_' + str(dataset) +'.npy'), np.array(model_params, dtype=list),allow_pickle=True)
    
    return(m.objVal, m.MIPGap, m.Runtime, output_dict, model_params)

### Run Experiments ###

# datatypes = ['1_Easy']
# datatypes = ['2_Easy']
datatypes = ['3_Easy']

train_lens = [100,500]
num_neurons = [3,10]
probs = [0.1, 0.25, 0.5, 0.75, 0.9]
datasets = [1, 2, 3, 4, 5]
lambda_set = [0,0.01,0.1,1,10]

for datatype in datatypes:
    if not os.path.isdir(datatype):
        os.mkdir(datatype)
    else:
        print('Data type File Exists')
            
    for K in num_neurons:    
        if datatype == '1_Easy':
            architecture = '2-'+str(K)+'-2'
            prefix = '1e'
        elif datatype == '1_Hard':
            architecture = '2-'+str(K)+'-2'        
            prefix = '1h'
        elif datatype == '2_Easy':
            architecture = '10-'+str(K)+'-2'
            prefix = '2e'
        elif datatype == '3_Easy':
            architecture = '20-'+str(K)+'-2'
            prefix = '3e'
    
        if not os.path.isdir(os.path.join(datatype, architecture)):
            os.mkdir(os.path.join(datatype, architecture))
        else:
            print('Architecture File Exists')
            
        for N in train_lens:
            if not os.path.isdir(os.path.join(datatype, architecture, 'Output_'+str(N))):
                os.mkdir(os.path.join(datatype, architecture, 'Output_'+str(N)))
            else:
                print('Output File Exists')
            
            for λ in lambda_set:
                if not os.path.isdir(os.path.join(datatype, architecture, 'Output_'+str(N),'lambda_'+str(λ))):
                    os.mkdir(os.path.join(datatype, architecture, 'Output_'+str(N),'lambda_'+str(λ)))
                else:
                    print('Lambda File Exists')            
            
                oosp = {}
                gap = {}
                runtime = {}            
                for prob in probs:
                    oosp[prob] = {}
                    gap[prob] = {}
                    runtime[prob] = {}
                    for dataset in datasets:
                        file_name = 'data_train_' + str(prob) + '_' + str(dataset) + '.csv'
                        file_path = prefix+'_athey_'+str(N)+'_learned'

                        df = pd.read_csv(os.path.join(file_path, file_name))

                        architecture_splitter = []
                        for substring in re.finditer('-', architecture):
                            architecture_splitter.append(substring.end())
                        
                        assert(K == int(architecture[architecture_splitter[-2]:architecture_splitter[-1]-1])), 'ERROR K:'+str(K)+' architecture:'+architecture+' '+datatype
                        D = int(architecture[:architecture_splitter[0]-1])
                        L = len(architecture_splitter) - 1

                        w_ub, w_lb, b_ub, b_lb = [1,-1,1,-1]
                        epsilon = 0.01

                        print('\nStarting Training '+ datatype + '_' + architecture + '_' + str(N) + '-' + str(prob) + '_' + str(dataset) + '...')

                        obj, mip_gap, mip_runtime, output, params = PrescriptiveNetwork_L0(df,D,K,λ)

                        print('\nFinished Training '+ datatype + '_' + architecture + '_' + str(N) + '-' + str(prob) + '_' + str(dataset) + '\n')

                        ### Testing ###

                        file_name = 'data_test_' + str(prob) + '_' + str(dataset) + '.csv'
                        file_path = prefix+'_athey_'+str(N)

                        test_df = pd.read_csv(os.path.join(file_path, file_name))

                        test_df['best_t'] = [1 if test_df['y1'][n] > test_df['y0'][n] else 0 for n in range(len(test_df))]

                        test_inputs = np.array(test_df.iloc[:,:D])
                        test_labels = test_df['best_t']

                        MIP_model_params = params

                        test_N = test_inputs.shape[0]

                        correct_pred = 0
                        for n in range(test_N):

                            h = (np.dot(MIP_model_params[0], test_inputs[n]) + MIP_model_params[1]).round(3)

                            h[h < epsilon] = 0
                            h[h >= epsilon] = 1

                            i = 2
                            for l in range(1,L):

                                h = (np.dot(MIP_model_params[i], h) + MIP_model_params[i+1]).round(3)

                                h[h < epsilon] = 0
                                h[h >= epsilon] = 1

                                i += 2

                            h = (np.array(np.dot(MIP_model_params[i], h) + MIP_model_params[i+1])).round(3)

                            h[h < epsilon] = 0
                            h[h >= epsilon] = 1

                            pred = np.argmax(h)
                            actual = test_labels[n]

                            if pred == actual:
                                correct_pred += 1

                        testing_acc = correct_pred/(test_N)*100    
                        print('Testing Accuracy for ' + str(prob) + '_' + str(dataset) + '=', testing_acc, '%\n')

                        oosp[prob][dataset] = testing_acc
                        gap[prob][dataset] = mip_gap
                        runtime[prob][dataset] = mip_runtime

                    avg_testing_acc = sum(oosp[prob][dataset] for dataset in datasets)/len(datasets)
                    print('Average Testing Accuracy for ' + str(prob) + '=', avg_testing_acc, '%\n')

                with open(os.path.join(datatype,architecture,'Output_'+str(N),'lambda_'+str(λ),'oosp.json'), 'w') as outfile:
                    json.dump(oosp, outfile)
                with open(os.path.join(datatype,architecture,'Output_'+str(N),'lambda_'+str(λ),'gap.json'), 'w') as outfile:
                    json.dump(gap, outfile)    
                with open(os.path.join(datatype,architecture,'Output_'+str(N),'lambda_'+str(λ),'runtime.json'), 'w') as outfile:
                    json.dump(runtime, outfile)
