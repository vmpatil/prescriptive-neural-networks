#!/usr/bin/env python
# coding: utf-8

# ### Experiments for Prescriptive Trees adapted from Bertsimus et. al (2019) [Jo et. al]
# 
# _Reference: Jo et. al (2020), Bertsimus et. al (2019)_

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from math import *
import pandas as pd
import os
import torch
import json

from math import floor, ceil

import scipy.stats as st

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

def K_PT(R, data, covariates, treatment, seed1, seed2, prob):
                
    x =  torch.tensor(data[covariates].values.tolist())
    
    I = x.shape[0]                                        # Length of input data
    F = len(covariates)                                   # Number of binary features
    K = sorted(data[treatment].unique())                  # Number of treatments    
    
    y_bar = data['y'] - min(data['y'])
    y_bar_max = max(y_bar)
    
    # Create a new model
    model = gp.Model('Kallus Adaptation')
    
    model.setParam('TimeLimit', 3600) # time limit of 1 hour
    model.setParam('MIPGap', 0.1/100) # gap limit of 0.1%    

    # Create variables
    
    w = {}
    b = {}
    lamb = {}
    chi = {}
    nu = {}
    rho = {}

    for n in T:
        for k in K:
            w[(n,k)] = model.addVar(vtype=GRB.BINARY, name='w '+str((n,k))) # EC.1k
        
        for i in range(I):
            lamb[(i,n)] = model.addVar(vtype=GRB.BINARY, name='lamb '+str((i,n))) # EC.1m
            nu[(i,n)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='nu '+str((i,n))) # EC.1o
            
        rho[n] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='rho '+str((n))) # EC.1p
                                        
    for n in B:
        for f in range(1,F+1):
            b[(n,f)] = model.addVar(vtype=GRB.BINARY, name= 'b '+str((n,f))) # EC.1l
            
        for i in range(I):
            chi[(i,n)] = model.addVar(vtype=GRB.BINARY, name='chi '+str((i,n))) # EC.1n
                                
    # Set objective
    model.setObjective(sum( sum(nu[(i,n)] for n in T) for i in range(I)), GRB.MAXIMIZE) # EC.1a
    
    # Add constraints

    for i in range(I):
        for n in T:            
            model.addConstr(lamb[(i,n)] >= 1 - sum(chi[(i,m)] for m in A(n) if R[(n,m)] == 1) 
                        + sum(-1 + chi[(i,m)] for m in A(n) if R[(n,m)] == -1), 
                        name='flow constr C2 '+str((i,n))) # EC.1c            
            
            model.addConstr(nu[(i,n)] <= y_bar_max*lamb[(i,n)], name='avg outcome C1 '+str((i,n))) # EC.1f
            
            model.addConstr(nu[(i,n)] <= rho[n], name='avg outcome C2 '+str((i,n))) # EC.1f
            
            model.addConstr(nu[(i,n)] >= rho[n] - y_bar_max*(1-lamb[(i,n)]), name='avg outcome C3 '+str((i,n))) # EC.1g

            for m in A(n):                
                model.addConstr(lamb[(i,n)] <= (1+R[(n,m)])/2 - R[(n,m)]*chi[(i,m)], 
                            name='flow constr C1 '+str((i,n,m))) # EC.1b

    for n in B:
        model.addConstr(sum(b[(n,f)] for f in range(1,F+1)) == 1, name='must branch '+str((n))) # EC.1d
        
        for i in range(I):
            model.addConstr(chi[(i,n)] == sum(b[(n,f)] for f in range(1,F+1) if x[i][f-1] == 0), 
                        name='chi def '+str((i,n))) # EC.1e
    
    M = y_bar_max*max(sum(([1]*I)[i] for i in range(I) if data[treatment].iloc[i] == k) for k in K)
    
    for n in T:
        for k in K:
            model.addConstr(sum((nu[(i,n)] - lamb[(i,n)]*y_bar.iloc[i]) for i in range(I) if data[treatment].iloc[i] == k)
                        <= M*(1-w[(n,k)]), name='linearize nu C1 '+str((n,k))) # EC.1h
            
            model.addConstr(sum((nu[(i,n)] - lamb[(i,n)]*y_bar.iloc[i]) for i in range(I) if data[treatment].iloc[i] == k)
                        >= -M*(1-w[(n,k)]), name='linearize nu C2 '+str((n,k))) # EC.1i

        model.addConstr(sum(w[(n,k)] for k in K) == 1, name='exactly one treatment '+str((n))) # EC.1j
            
    # Optimize model
        
    model.params.LogFile = os.path.join('output_'+str(seed1),str(depth),'grb_output_'+str(seed2)+'_'+str(prob)+'.log')
    model.optimize()
    
    output_dict = {}
    
    for v in model.getVars():
        output_dict[v.varName] = np.round(v.x, 5)

    print('Obj: %g' % model.objVal)
    
    branching_rule = []
    treatment_assignments = {}

    for n in B:
        branching_rule.append([round(output_dict['b '+str((n,f))],2) for f in range(1,F+1)])

    for n in T:
        for k in K:
            if round(output_dict['w '+str((n,k))]) == 1:
                treatment_assignments[n] = int(k)
    
    np.save(os.path.join('output_'+str(seed1),str(depth),'branching_rule_'+str(seed2)+'_'+str(prob)+'.npy'), np.array(branching_rule, dtype=list),allow_pickle=True)
    with open(os.path.join('output_'+str(seed1),str(depth),'treatment_assignments_'+str(seed2)+'_'+str(prob)+'.json'), 'w') as outfile:
        json.dump(treatment_assignments, outfile)
    
    return model.objVal, model.MIPGap, model.Runtime, output_dict, branching_rule, treatment_assignments

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

            obj, mip_gap, mip_runtime, output, rule, assignments = K_PT(ancestor_pathing, train, covariates, treatment, seed1, seed2, prob)
    
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
    
        ancestor_pathing = {}
        for p in B+T:
            for q in A(p):
                if r(q) in A(p) or r(q) == p:
                    ancestor_pathing[(p,q)] = 1
                else:
                    ancestor_pathing[(p,q)] = -1
        
        if not os.path.isdir(os.path.join('output_'+str(seed1),str(depth))):
            os.mkdir(os.path.join('output_'+str(seed1),str(depth)))
            print('\nDepth File Created')
        else:
            print('Depth File Exists')
                        
        run_experiment(seed1)