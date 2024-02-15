#!/usr/bin/env python
# coding: utf-8

### Experiments for Optimal Prescriptive Trees for Jo et. al
# 
# _Reference: Jo et. al (2020)_

# _Last Updated: 16th August 2022_

### Packages ###

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import os
import json
import re
import torch

### Function Definition ###

def a(n): 
    '''
    Returns the ancenstor of the node n ∈ B ∪ T of a binary tree
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

def J_PT(data, num_feats, outcome='y', treatment='t'):
    
    x = torch.tensor(data.iloc[:,:num_feats].values.tolist())
    
    I = x.shape[0]                                        # Length of input data
    F = x.shape[1]                                        # Number of binary features
    K = data[treatment].unique()                          # Number of treatments
    
    # Create a new model
    m = gp.Model('Jo et. al OPT')
    
    m.setParam('TimeLimit', 3600) # time limit of 1 hour
    m.setParam('MIPGap', 0.1/100) # gap limit of 0.1%


    # Create variables
    
    w = {}
    b = {}
    p = {}
    z = {}

    for k in K:
        for n in B+T:
            w[(n,k)] = m.addVar(vtype=GRB.BINARY, name='w '+str((n,k)))
            
    for i in range(I):
        for n in B+T:
            z[(i,a(n),n)] = m.addVar(vtype=GRB.BINARY, name= 'z '+str((i,a(n),n)))
            for k in K:
                z[(i,n,'t'+str(k))] = m.addVar(vtype=GRB.BINARY, name= 'z '+str((i,n,'t'+str(k))))
                    
    for n in B:
        for f in range(1,F+1):
            b[(n,f)] = m.addVar(vtype=GRB.BINARY, name= 'b '+str((n,f)))
            
    for n in B+T:
        p[n] = m.addVar(vtype=GRB.BINARY, name= 'p '+str((n)))
                    
    # Set objective
    m.setObjective(sum(sum(sum(z[(i,n,'t'+str(k))]*data['linear' + str(k)][i] for n in B+T) for k in K) for i in range(I))
                   + sum( sum((data[outcome][i] - data['linear' + str(data['t'][i])][i])*((z[(i,n,'t'+str(data[treatment][i]))])/data['prob_t_pred_tree'][i])
                              for n in B+T) 
                         for i in range(I)), GRB.MAXIMIZE) # can simplify this by pulling the summations out
    
    # Add constraints
    
    for n in B:            
        m.addConstr(sum(b[(n,f)] for f in range(1,F+1)) + p[n] + sum(p[m] for m in A(n))
                    == 1, name='branching const '+str((n))) # 7b
            
    for i in range(I):
        for n in T:
            m.addConstr(z[(i,a(n),n)] == sum(z[(i,n,'t'+str(k))] for k in K), name='flow conservation C2 '+str((i,n))) # 7e
            
        for n in B+T:
            for k in K:
                m.addConstr(z[(i,n,'t'+str(k))] <= w[(n,k)], name='sink '+str((i,n,k))) # 7i
        
        m.addConstr(z[(i,0,1)] == 1, name='source constr '+str((i))) # 7f
        
        for n in B:
            m.addConstr(z[(i,a(n),n)] == z[(i,n,l(n))] + z[(i,n,r(n))] + sum(z[(i,n,'t'+str(k))] for k in K), 
                        name='flow conservation C1 '+str((n,i))) # 7d
            m.addConstr(z[(i,n,l(n))] <= sum(b[(n,f)] for f in range(1,F+1) if x[i][f-1] == 0), name='left decision' +str((n,i))) # 7g
            m.addConstr(z[(i,n,r(n))] <= sum(b[(n,f)] for f in range(1,F+1) if x[i][f-1] == 1), name='right decision' +str((n,i))) # 7h
                                
    for n in T:
        m.addConstr(p[n] + sum(p[m] for m in A(n)) == 1, name='assign treatment '+str((n))) # 7c
        
    for n in B+T:
        m.addConstr(sum(w[(n,k)] for k in K) == p[n], name='exactly one treatment '+ str((n))) # 7j
    
    # Optimize model
    
    m.params.LogFile = os.path.join(datatype,str(d),'Output_'+str(N),'grb_output_' + str(prob) + '_' + str(dataset) + '.log')    
    m.optimize()
    
    output_dict = {}
    
    for v in m.getVars():
        output_dict[v.varName] = np.round(v.x, 5)

    print('Obj: %g' % m.objVal)
    
    branching_rule = []
    treatment_nodes = []
    treatment_assignments = {}

    for n in B:
        branching_rule.append([int(output_dict['b '+str((n,f))]) for f in range(1,F+1)])

    treatment_nodes = [n for n in B+T if output_dict['p '+str((n))] == 1]
    for treatment_node in treatment_nodes:
        for k in K:
            if round(output_dict['w '+str((treatment_node,k))]) == 1:
                treatment_assignments[treatment_node] = int(k)
    
    np.save(os.path.join(datatype,str(d),'Output_'+str(N), 'branching_rule_' + str(prob) + '_' + str(dataset) +'.npy'), np.array(branching_rule, dtype=list),allow_pickle=True)
    with open(os.path.join(datatype,str(d),'Output_'+str(N),'treatment_assignments_' + str(prob) + '_' + str(dataset) +'.json'), 'w') as outfile:
        json.dump(treatment_assignments, outfile)
    
    return m.objVal, m.MIPGap, m.Runtime, output_dict, branching_rule, treatment_assignments

### Run Experiments ###

datatypes = ['1_Easy','1_Hard','2_Easy','3_Easy']
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

    
        B = [i for i in range(1,2**d)]                        # Branching nodes
        T = [i for i in range(2**d,2**(d+1))]                 # Terminal Nodes
        
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

                    obj, mip_gap, mip_runtime, output, rule, assignments = J_PT(df, num_feats, outcome='y', treatment='t')

                    print('\nFinished Training '+ datatype + '_' + str(d) + '_' + str(N) + '-' + str(prob) + '_' + str(dataset) + '\n')
    
                    ### Testing

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