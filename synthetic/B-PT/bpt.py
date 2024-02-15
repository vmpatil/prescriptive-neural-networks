#!/usr/bin/env python
# coding: utf-8

# ### Experiments for Prescriptive Trees adapted from Bertsimus et. al (2019) [Jo et. al]
# 
# _Reference: Jo et. al (2020), Bertsimus et. al (2019)_

# _Last Updated: August 18th 2022_

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from math import *
import pandas as pd
import os
import torch
import json

### Function Definition ###

def a(n): 
    '''
    Returns the ancestor of the node n ∈ B ∪ T of a binary tree
    '''
    if (n in B) or (n in T):
        return n // 2
    else: 
        raise Exception("Not a valid node")
    
def l(n): 
    '''
    Returns the left child of the node n ∈ B of a binary tree
    '''
    if n in B:
        return 2*n
    else: 
        raise Exception("Not a valid node")
    
def r(n): 
    '''
    Returns the right child of the node n ∈ B of a binary tree
    '''
    if n in B:
        return 2*n + 1
    else: 
        raise Exception("Not a valid node")
    
def A(n):
    '''
    Returns the set of all ancestors of node n ∈ B ∪ T
    '''
    ancestors = []
    if (n in B) or (n in T):
        current = n
        while current != 1:
            current = current // 2
            ancestors.append(current)
        return ancestors
    else: 
        raise Exception("Not a valid node")

### MIP Formulation ###

def B_PT(data, R, num_feats, theta=0.5, outcome='y', treatment='t'):
    
    x = torch.tensor(data.iloc[:,:num_feats].values.tolist())
    
    I = x.shape[0]                                        # Length of input data
    F = num_feats                                         # Number of binary features
    K = data[treatment].unique()                          # Number of treatments
    
    y_bar = data[outcome] - min(data[outcome])
    y_bar_max = max(y_bar)
    
    # Create a new model
    model = gp.Model('Bertsimus Adaptation')
    
    model.setParam('TimeLimit', 3600) # time limit of 1 hour
    model.setParam('MIPGap', 0.1/100) # gap limit of 0.1%

    # Create variables
    
    w = {}
    b = {}
    lamb = {}
    chi = {}
    nu = {}
    rho = {}
    g = {}
    beta = {}    
    
    for n in T:
        for k in K:
            w[(n,k)] = model.addVar(vtype=GRB.BINARY, name='w '+str((n,k))) # EC.2m
        
        for i in range(I):
            lamb[(i,n)] = model.addVar(vtype=GRB.BINARY, name='lamb '+str((i,n))) # EC.2o
            nu[(i,n)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='nu '+str((i,n))) # EC.2q
            
        rho[n] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='rho '+str((n))) # EC.2r
                                        
    for n in B:
        for f in range(1,F+1):
            b[(n,f)] = model.addVar(vtype=GRB.BINARY, name= 'b '+str((n,f))) # EC.2n
            
        for i in range(I):
            chi[(i,n)] = model.addVar(vtype=GRB.BINARY, name='chi '+str((i,n))) # EC.2p
            
    for i in range(I):
        g[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='g '+str((i))) # EC.2s
        
    for k in K:
        for n in T:
            beta[(n,k)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='beta '+str((n,k)))         
                    
    # Set objective
    model.setObjective(theta*sum( sum(nu[(i,n)] for n in T) for i in range(I))
                       - (1-theta)*sum((data[outcome][i] - g[i])**2 for i in range(I)), 
                       GRB.MAXIMIZE) # EC.2a
    
    # Add constraints

    for i in range(I):
        for n in T:            
            model.addConstr(lamb[(i,n)] >= 1 - sum(chi[(i,m)] for m in A(n) if R[(n,m)] == 1) 
                        + sum(-1 + chi[(i,m)] for m in A(n) if R[(n,m)] == -1), 
                        name='flow constr C2 '+str((i,n))) # EC.2c            
            
            model.addConstr(nu[(i,n)] <= y_bar_max*lamb[(i,n)], name='avg outcome C1 '+str((i,n))) # EC.2f
            
            model.addConstr(nu[(i,n)] <= rho[n], name='avg outcome C2 '+str((i,n))) # EC.2f
            
            model.addConstr(nu[(i,n)] >= rho[n] - y_bar_max*(1-lamb[(i,n)]), name='avg outcome C3 '+str((i,n))) # EC.2g

            for m in A(n):                
                model.addConstr(lamb[(i,n)] <= (1+R[(n,m)])/2 - R[(n,m)]*chi[(i,m)], 
                            name='flow constr C1 '+str((i,n,m))) # EC.2b

    for n in B:
        model.addConstr(sum(b[(n,f)] for f in range(1,F+1)) == 1, name='must branch '+str((n))) # EC.2d
        
        for i in range(I):
            model.addConstr(chi[(i,n)] == sum(b[(n,f)] for f in range(1,F+1) if x[i][f-1] == 0), 
                        name='chi def '+str((i,n))) # EC.2e
    
    M = y_bar_max*max(sum(([1]*I)[i] for i in range(I) if df[treatment][i] == k) for k in K)
    
    for n in T:
        for k in K:
            model.addConstr(sum((nu[(i,n)] - lamb[(i,n)]*y_bar[i]) for i in range(I) if data[treatment][i] == k)
                        <= M*(1-w[(n,k)]), name='linearize nu C1 '+str((n,k))) # EC.2h
            
            model.addConstr(sum((nu[(i,n)] - lamb[(i,n)]*y_bar[i]) for i in range(I) if data[treatment][i] == k)
                        >= -M*(1-w[(n,k)]), name='linearize nu C2 '+str((n,k))) # EC.2i

        model.addConstr(sum(w[(n,k)] for k in K) == 1, name='exactly one treatment '+str((n))) # EC.2j
    
    min_avg_outcome = min([df[outcome][i] for i in range(len(df))])
    max_avg_outcome = max([df[outcome][i] for i in range(len(df))])

    M = max_avg_outcome - min_avg_outcome

    for i in range(I):
        for n in T:
            model.addConstr(g[i] - beta[(n,data[treatment][i])] <= M*(1-lamb[(i,n)]), 
                        name = 'avg outcome for treatment node C1'+str((i,n))) # EC.2k

            model.addConstr(g[i] - beta[(n,data[treatment][i])] >= -M*(1-lamb[(i,n)]), 
                        name = 'avg outcome for treatment node C2'+str((i,n))) # EC.2l
                    
    # Optimize model
        
    model.params.LogFile = os.path.join(datatype,str(d),'Output_'+str(N),'grb_output_' + str(prob) + '_' + str(dataset) + '.log')
    model.optimize()

    output_dict = {}
    
    for v in model.getVars():
        output_dict[v.varName] = round(v.x, 5)

    print('Obj: %g' % model.objVal)
    
    branching_rule = []
    treatment_assignments = {}

    for n in B:
        branching_rule.append([round(output_dict['b '+str((n,f))],2) for f in range(1,F+1)])

    for n in T:
        for k in K:
            if round(output_dict['w '+str((n,k))]) == 1:
                treatment_assignments[n] = int(k)
    
    np.save(os.path.join(datatype,str(d),'Output_'+str(N), 'branching_rule_' + str(prob) + '_' + str(dataset) +'.npy'), np.array(branching_rule, dtype=list),allow_pickle=True)
    with open(os.path.join(datatype,str(d),'Output_'+str(N),'treatment_assignments_' + str(prob) + '_' + str(dataset) +'.json'), 'w') as outfile:
        json.dump(treatment_assignments, outfile)
    
    return model.objVal, model.MIPGap, model.Runtime, output_dict, branching_rule, treatment_assignments

### Run Experiments ###

# datatypes = ['1_Easy','1_Hard']
# datatypes = ['2_Easy']
datatypes = ['3_Easy']
train_lens = [100,500]
depths = [1,2]
datasets = [1, 2, 3, 4, 5]
probs = [0.1, 0.25, 0.5, 0.75, 0.9]

for datatype in datatypes:
    if not os.path.isdir(datatype):
        os.mkdir(datatype)
    else:
        print('Data type File Exists')
        
    if datatype == '1_Easy':
        num_feats = 20
        prefix = '1e'
    elif datatype == '1_Hard':
        num_feats = 20
        prefix = '1h'
    elif datatype == '2_Easy':
        num_feats = 100
        prefix = '2e'
    elif datatype == '3_Easy':
        num_feats = 200
        prefix = '3e'
    
    for d in depths:                
        if not os.path.isdir(os.path.join(datatype, str(d))):
            os.mkdir(os.path.join(datatype, str(d)))
        else:
            print('Depth File Exists')
        
        B = [n for n in range(1,2**d)]
        T = [n for n in range(2**d,2**(d+1))]

        ancestor_pathing = {}
        for p in B+T:
            for q in A(p):
                if r(q) in A(p) or r(q) == p:
                    ancestor_pathing[(p,q)] = 1
                else:
                    ancestor_pathing[(p,q)] = -1        
        
        for N in train_lens:
            if not os.path.isdir(os.path.join(datatype, str(d), 'Output_'+str(N))):
                os.mkdir(os.path.join(datatype, str(d), 'Output_'+str(N)))
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
                    file_name = 'data_train_enc_' + str(prob) + '_' + str(dataset) + '.csv'
                    file_path = prefix+'_athey_'+str(N)+'_learned'

                    df = pd.read_csv(os.path.join(file_path, file_name))
            
                    print('\nStarting Training '+ datatype + '_' + str(d) + '_' + str(N) + '-' + str(prob) + '_' + str(dataset) + '...')

                    obj, mip_gap, mip_runtime, output, rule, assignments = B_PT(df, ancestor_pathing, num_feats, theta=0.5, outcome='y', treatment='t')

                    print('\nFinished Training '+ datatype + '_' + str(d) + '_' + str(N) + '-' + str(prob) + '_' + str(dataset) + '\n')

                    for i in range(len(df)):
                        for n in T:
                            if int(output['lamb '+str((i,n))]) == 1:
                                assert round(output['g '+str(i)],3) == round(output['beta '+str((n,df['t'][i]))],3), (i, output['g '+str(i)], output['beta '+str((n,df['t'][i]))])
    
                    ### Testing ###

                    file_name = 'data_test_enc_' + str(prob) + '_' + str(dataset) + '.csv'
                    file_path = prefix+'_athey_'+str(N)

                    test_df = pd.read_csv(os.path.join(file_path, file_name))

                    test_df['best_t'] = [1 if test_df['y1'][n] > test_df['y0'][n] else 0 for n in range(len(test_df))]

                    test_x = np.array(test_df.iloc[:,:num_feats])
                    test_best_t = test_df['best_t']

                    I = test_x.shape[0]

                    correct_pred = 0
                    for i in range(I):
                        curr_node = 1
                        assigned = False
                        
                        if curr_node in assignments:
                            treatment = assignments[curr_node]
                            best = test_best_t[i]
        
                            assigned = True
        
                            if treatment == best:
                                correct_pred += 1

                        while not assigned:
        
                            h = np.dot(rule[curr_node-1], test_x[i])
                            curr_node = int((2*curr_node)+h)

                            if curr_node in assignments:
                                treatment = assignments[curr_node]
                                best = test_best_t[i]
            
                                assigned = True
            
                                if treatment == best:
                                    correct_pred += 1
        
                    testing_acc = correct_pred/(I)*100    
                    print("Testing Accuracy", testing_acc, "%\n")
        
                    oosp[prob][dataset] = testing_acc
                    gap[prob][dataset] = mip_gap
                    runtime[prob][dataset] = mip_runtime
    
                avg_testing_acc = sum(oosp[prob][dataset] for dataset in datasets)/len(datasets)
                print('Average Testing Accuracy for ' + str(prob) + '=', avg_testing_acc, '%\n')
    
            with open(os.path.join(datatype,str(d),'Output_'+str(N),'oosp.json'), 'w') as outfile:
                json.dump(oosp, outfile)
            with open(os.path.join(datatype,str(d),'Output_'+str(N),'gap.json'), 'w') as outfile:
                json.dump(gap, outfile)    
            with open(os.path.join(datatype,str(d),'Output_'+str(N),'runtime.json'), 'w') as outfile:
                json.dump(runtime, outfile)
