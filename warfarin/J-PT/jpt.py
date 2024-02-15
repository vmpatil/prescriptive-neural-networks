#!/usr/bin/env python
# coding: utf-8

### Experiments for Optimal Prescriptive Trees for Jo et. al
# 
# _Reference: Jo et. al (2020)_

### Packages ###

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import os
import json
import re
import torch

from math import floor, ceil

import scipy.stats as st


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

def J_PT(data, covariates, treatment, seed1, seed2, prob):
                
    x =  torch.tensor(data[covariates].values.tolist())
    
    I = x.shape[0]                                        # Length of input data
    F = len(covariates)                                   # Number of binary features
    K = sorted(data[treatment].unique())                  # Number of treatments
    
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
            w[(n,k)] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='w '+str((n,k)))
            
    for i in range(I):
        for n in B+T:
            z[(i,a(n),n)] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name= 'z '+str((i,a(n),n)))
            for k in K:
                z[(i,n,treatment+str(k))] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name= 'z '+str((i,n,treatment+str(k))))
                    
    for n in B:
        for f in range(1,F+1):
            b[(n,f)] = m.addVar(vtype=GRB.BINARY, name= 'b '+str((n,f)))
            
    for n in B+T:
        p[n] = m.addVar(vtype=GRB.BINARY, name= 'p '+str((n)))
                    
    # Set objective
    m.setObjective(sum(sum(sum(z[(i,n,treatment+str(k))]*data['dm_reg_'+str(k)].iloc[i] for n in B+T) for k in K) for i in range(I))
                   + sum( sum((data['y'].iloc[i] - data['dm_reg_'+str(data[treatment].iloc[i])].iloc[i])*((z[(i,n,treatment+str(data[treatment].iloc[i]))])/data['prop_score'].iloc[i])
                              for n in B+T) 
                         for i in range(I)), GRB.MAXIMIZE) # can simplify this by pulling the summations out
    
    # Add constraints
    
    for n in B:            
        m.addConstr(sum(b[(n,f)] for f in range(1,F+1)) + p[n] + sum(p[m] for m in A(n))
                    == 1, name='branching const '+str((n))) # 7b            
            
    for i in range(I):
        for n in T:
            m.addConstr(z[(i,a(n),n)] == sum(z[(i,n,treatment+str(k))] for k in K), name='flow conservation C2 '+str((i,n))) # 7e
            
        for n in B+T:
            for k in K:
                m.addConstr(z[(i,n,treatment+str(k))] <= w[(n,k)], name='sink '+str((i,n,k))) # 7i
        
        m.addConstr(z[(i,0,1)] == 1, name='source constr '+str((i))) # 7f
        
        for n in B:
            m.addConstr(z[(i,a(n),n)] == z[(i,n,l(n))] + z[(i,n,r(n))] + sum(z[(i,n,treatment+str(k))] for k in K), 
                        name='flow conservation C1 '+str((n,i))) # 7d
            m.addConstr(z[(i,n,l(n))] <= sum(b[(n,f)] for f in range(1,F+1) if x[i][f-1] == 0), name='left decision' +str((n,i))) # 7g
            m.addConstr(z[(i,n,r(n))] <= sum(b[(n,f)] for f in range(1,F+1) if x[i][f-1] == 1), name='right decision' +str((n,i))) # 7h
                                
    for n in T:
        m.addConstr(p[n] + sum(p[m] for m in A(n)) == 1, name='assign treatment '+str((n))) # 7c
        
    for n in B+T:
        m.addConstr(sum(w[(n,k)] for k in K) == p[n], name='exactly one treatment '+ str((n))) # 7j
    
    # Optimize model
    
    m.params.LogFile = os.path.join('output_'+str(seed1),str(depth),'grb_output_'+str(seed2)+'_'+str(prob)+'.log')
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
    
    np.save(os.path.join('output_'+str(seed1),str(depth),'branching_rule_'+str(seed2)+'_'+str(prob)+'.npy'), np.array(branching_rule, dtype=list),allow_pickle=True)
    with open(os.path.join('output_'+str(seed1),str(depth),'treatment_assignments_'+str(seed2)+'_'+str(prob)+'.json'), 'w') as outfile:
        json.dump(treatment_assignments, outfile)
    
    return m.objVal, m.MIPGap, m.Runtime, output_dict, branching_rule, treatment_assignments

### Run Experiments ###

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
            
            file_name = 'warfarin_p={}_enc.csv'.format(str(prob))

            df = pd.read_csv(os.path.join(file_path,file_name))
    
            print('\nOn Seed: '+str(seed1)+','+str(seed2))    
    
            train = df.sample(frac=.70, replace=False, weights=None, random_state=seed1, axis=None, ignore_index=False)
            test = df.iloc[df.index.difference(train.index)]    
        
            ### Define Parameters ###
    
            treatment = 'Random Treatment'
            covariates = list(df.columns[:83]) + list(df.columns[-9:])
            
            N = len(df)             # Number of individuals in dataset
            D = len(covariates)     # Input data dimensions         

            prescription_list = []
                
            ### Training ###

            obj, mip_gap, mip_runtime, output, rule, assignments = J_PT(train, covariates, treatment, seed1, seed2, prob)
    
            gap[seed2][prob] = mip_gap*100
            runtime[seed2][prob] = mip_runtime

            ### Testing ###

            T = sorted(df[treatment].unique())

            test_X = np.array(test[covariates])

            I = test_X.shape[0]

            correct_pred = 0
            for i in range(I):
                curr_node = 1
                assigned = False

                if curr_node in assignments:
                    prescription = assignments[curr_node]
                    prescription_list.append(prescription)

                    assigned = True

                while not assigned:

                    h = np.dot(rule[curr_node-1], test_X[i])
                    curr_node = int((2*curr_node)+h)

                    if curr_node in assignments:
                        prescription = assignments[curr_node]
                        prescription_list.append(prescription)              

                        assigned = True
                
                if prescription == (test['Discrete Noisy Learned Dose']).iloc[i]:
                    correct_pred += 1
                        
            percent_correct = (correct_pred/I)*100

            oosp[seed2][prob] = percent_correct
            prescriptions[seed2][prob] = prescription_list
            
            print('Out-of-Sample Probability for depth={}, p={}, seed={}: {}%'.format(str(depth),str(prob),str(seed2),str(percent_correct)))
        
    with open(os.path.join('output_'+str(seed1),str(depth),'oosp.json'), 'w') as outfile:
        json.dump(oosp, outfile)    
    with open(os.path.join('output_'+str(seed1),str(depth),'gap.json'), 'w') as outfile:
        json.dump(gap, outfile)    
    with open(os.path.join('output_'+str(seed1),str(depth),'runtime.json'), 'w') as outfile:
        json.dump(runtime, outfile)
    with open(os.path.join('output_'+str(seed1),str(depth),'treatments.json'), 'w') as outfile:
        json.dump(prescriptions, outfile)
                        
    return None

### Run Experiments ###

depths = [1,2]

for seed1 in [41]:
# for seed in [41,12,60,2,872]:
    if not os.path.isdir(os.path.join('output_'+str(seed1))):
        os.mkdir(os.path.join('output_'+str(seed1)))
        print('\nOutput File Created')
    else:
        print('Output File Exists')

    for depth in depths:

        B = [i for i in range(1,2**depth)]                            # Branching nodes
        T = [i for i in range(2**depth,2**(depth+1))]                 # Terminal Nodes

        if not os.path.isdir(os.path.join('output_'+str(seed1),str(depth))):
            os.mkdir(os.path.join('output_'+str(seed1),str(depth)))
            print('\nDepth File Created')
        else:
            print('Depth File Exists')
                
        run_experiment(seed1)