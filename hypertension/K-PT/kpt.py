#!/usr/bin/env python
# coding: utf-8

# ### Experiments for Prescriptive Trees adapted from Bertsimus et. al (2019) [Jo et. al]
# 
# _Reference: Jo et. al (2020), Bertsimus et. al (2019)_

# _Last Updated: November 22nd 2022_

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

def K_PT(R, data, covariates, treatment, version, fold):
    
    if treatment == 'bpmed.dc':
        assert version=='v1'
    elif treatment == 'dcDayPP':
        assert version=='v2'
    elif treatment =='cross_treatment':
        assert version=='v3'
                
    x =  torch.tensor(data[covariates].values.tolist())
    
    I = x.shape[0]                                        # Length of input data
    F = len(covariates)                                   # Number of binary features
    K = sorted(data[treatment].unique())                  # Number of treatments    
    
    y_bar = data['max.sbp'] - min(data['max.sbp'])
    y_bar_max = max(y_bar)
    
    # Create a new model
    model = gp.Model('Kallus Adaptation')
    
    model.setParam('TimeLimit', 3600) # time limit of 1 hour

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
    model.setObjective(sum( sum(-nu[(i,n)] for n in T) for i in range(I)), GRB.MAXIMIZE) # EC.1a
    
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
    
    M = y_bar_max*max(sum(([1]*I)[i] for i in range(I) if data[treatment][i] == k) for k in K)
    
    for n in T:
        for k in K:
            model.addConstr(sum((nu[(i,n)] - lamb[(i,n)]*y_bar[i]) for i in range(I) if data[treatment][i] == k)
                        <= M*(1-w[(n,k)]), name='linearize nu C1 '+str((n,k))) # EC.1h
            
            model.addConstr(sum((nu[(i,n)] - lamb[(i,n)]*y_bar[i]) for i in range(I) if data[treatment][i] == k)
                        >= -M*(1-w[(n,k)]), name='linearize nu C2 '+str((n,k))) # EC.1i

        model.addConstr(sum(w[(n,k)] for k in K) == 1, name='exactly one treatment '+str((n))) # EC.1j
            
    # Optimize model
        
    model.params.LogFile = os.path.join('Output',version,str(depth),'grb_output_'+str(fold)+'.log')
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
    
    np.save(os.path.join('Output',version,str(depth),'branching_rule_'+str(fold)+'.npy'), np.array(branching_rule, dtype=list),allow_pickle=True)
    with open(os.path.join('Output',version,str(depth),'treatment_assignments_' +str(fold)+'.json'), 'w') as outfile:
        json.dump(treatment_assignments, outfile)
    
    return model.objVal, model.MIPGap, model.Runtime, output_dict, branching_rule, treatment_assignments

### Run Experiments ###

def run_experiment(version):
        
    ### Parse Data ###
    
    file_path = os.path.join('hypertension','hypertension_learned')
    file_name = 'hypertension_'+version+'_enc.csv'

    df = pd.read_csv(os.path.join(file_path,file_name))
    
    print('Training Version:',version)
    
    ### Define Parameters ###
    
    if version=='v1':
        
        treatment = 'bpmed.dc'        
        covariates = ['mode.del', 'chronicHTN', 'gestHTN', 'sipe', 'ethnicity', 
                      'feeding.br', 'DM.pregest', 'tobacco', 'DM.gest', 'pree', 
                      'pree.w.sf', 'eclampsia', 'hellp', 'dcDayPP_1', 'dcDayPP_2', 
                      'dcDayPP_3', 'dcDayPP_4'] \
        + ['bmi.prenatal_'+str(i) for i in range(1,11)] \
        + ['mom.age_'+str(i) for i in range(1,11)] \
        + ['gest.age_'+str(i) for i in range(1,11)] \
        + ['race_'+str(i) for i in range(3)] \
        + ['insurance_'+str(i) for i in range(3)]

        N = len(df)             # Number of individuals in dataset
        D = len(covariates)     # Input data dimensions 
        
    elif version=='v2':
        
        treatment = 'dcDayPP'
        covariates = ['bpmed.dc', 'mode.del', 'chronicHTN',
                      'gestHTN', 'sipe', 'ethnicity', 'feeding.br', 'DM.pregest', 'tobacco',
                      'DM.gest', 'pree', 'pree.w.sf', 'eclampsia', 'hellp'] \
        + ['bmi.prenatal_'+str(i) for i in range(1,11)] \
        + ['mom.age_'+str(i) for i in range(1,11)] \
        + ['gest.age_'+str(i) for i in range(1,11)] \
        + ['race_'+str(i) for i in range(3)] \
        + ['insurance_'+str(i) for i in range(3)]

        N = len(df)             # Number of individuals in dataset
        D = len(covariates)     # Input data dimensions 
    
    elif version=='v3':
        
        treatment = 'cross_treatment'
        covariates = ['mode.del', 'chronicHTN','gestHTN', 'sipe', 'ethnicity',
                      'feeding.br', 'DM.pregest', 'tobacco',
                      'DM.gest', 'pree', 'pree.w.sf', 'eclampsia', 'hellp'] \
        + ['bmi.prenatal_'+str(i) for i in range(1,11)] \
        + ['mom.age_'+str(i) for i in range(1,11)] \
        + ['gest.age_'+str(i) for i in range(1,11)] \
        + ['race_'+str(i) for i in range(3)] \
        + ['insurance_'+str(i) for i in range(3)]

        N = len(df)             # Number of individuals in dataset
        D = len(covariates)     # Input data dimensions         

    ### K-fold Cross Validation ###
    
    num_folds = 10
    fold_idx = [fold*floor(len(df)/num_folds) for fold in range(num_folds)]+[len(df)]
    
    all_outcomes = []
    performance = []
    
    gap = {}
    runtime = {}
                
    for fold in range(len(fold_idx)-1):

        train_set = df[0:fold_idx[fold]].append(df[fold_idx[fold+1]:]).copy(deep=True).reset_index(drop=True)
        test_set = df[fold_idx[fold]:fold_idx[fold+1]].copy(deep=True).reset_index(drop=True)

        print('-'*75)
        print('On Fold', fold+1)

        ### Training ###

        obj, mip_gap, mip_runtime, output, rule, assignments = K_PT(ancestor_pathing, train_set, covariates, treatment, version, fold)
        
        gap[fold] = mip_gap
        runtime[fold] = mip_runtime

        ### Testing ###

        T = sorted(df[treatment].unique())

        test_X = np.array(test_set[covariates])

        I = test_X.shape[0]

        outcome = []
        for i in range(I):
            curr_node = 1
            assigned = False

            if curr_node in assignments:
                prescription = assignments[curr_node]

                assigned = True

            while not assigned:

                h = np.dot(rule[curr_node-1], test_X[i])
                curr_node = int((2*curr_node)+h)

                if curr_node in assignments:
                    prescription = assignments[curr_node]

                    assigned = True

            outcome.append(test_set['psi_'+treatment+'_'+str(prescription)].iloc[i])

        performance.append(sum(outcome)/len(outcome))
        all_outcomes.extend(outcome)

        print('Fold '+str(fold+1)+', Testing π(s) = ', performance[fold],'\n')
        
    var = np.mean([all_outcomes[i]**2 for i in range(len(all_outcomes))]) - (np.mean(all_outcomes)**2)
    CI = [np.mean(all_outcomes) - st.norm.ppf(0.95)*(var/len(all_outcomes))**0.5,
          np.mean(all_outcomes) + st.norm.ppf(0.95)*(var/len(all_outcomes))**0.5]
    print('π(s) : {}| 95% Confidence Interval = {}'.format(np.mean(all_outcomes),CI))

    with open(os.path.join('Output',version,str(depth),'outcomes.json'), 'w') as outfile:
        json.dump(all_outcomes, outfile)    
    with open(os.path.join('Output',version,str(depth),'gap.json'), 'w') as outfile:
        json.dump(gap, outfile)    
    with open(os.path.join('Output',version,str(depth),'runtime.json'), 'w') as outfile:
        json.dump(runtime, outfile)
                                
    return(performance, all_outcomes)

### Run Experiments ###

versions = ['v1','v2','v3']

if not os.path.isdir(os.path.join('Output')):
    os.mkdir(os.path.join('Output'))
    print('\nOutput File Created')
else:
    print('Output File Exists')

depths = [1,2]

for version in versions:

    if not os.path.isdir(os.path.join('Output',version)):
        os.mkdir(os.path.join('Output',version))
        print('\nVersion File Created')
    else:
        print('Version File Exists')

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
    
        if not os.path.isdir(os.path.join('Output',version,str(depth))):
            os.mkdir(os.path.join('Output',version,str(depth)))
            print('\nDepth File Created')
        else:
            print('Depth File Exists')
                            
        run_experiment(version)