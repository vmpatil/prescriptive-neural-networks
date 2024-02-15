#!/usr/bin/env python
# coding: utf-8

### Experiments for Optimal Prescriptive Trees for Jo et. al
# 
# _Reference: Jo et. al (2020)_

# _Last Updated: 6th September 2022_

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

def J_PT(data, covariates, treatment, version, fold):
    
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
    m.setObjective(-(sum(sum(sum(z[(i,n,treatment+str(k))]*data['dm_reg_' +treatment+'_'+str(k)][i] for n in B+T) for k in K) for i in range(I))
                   + sum( sum((data['max.sbp'][i] - data['dm_reg_' +treatment+'_'+str(data[treatment][i])][i])*((z[(i,n,treatment+str(data[treatment][i]))])/data[treatment+'_prop'][i])
                              for n in B+T) 
                         for i in range(I))), GRB.MAXIMIZE) # can simplify this by pulling the summations out
    
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
    
    m.params.LogFile = os.path.join('Output',version,str(depth),'grb_output_'+str(fold)+'.log')
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
    
    np.save(os.path.join('Output',version,str(depth),'branching_rule_'+str(fold)+'.npy'), np.array(branching_rule, dtype=list),allow_pickle=True)
    with open(os.path.join('Output',version,str(depth),'treatment_assignments_' +str(fold)+'.json'), 'w') as outfile:
        json.dump(treatment_assignments, outfile)
    
    return m.objVal, m.MIPGap, m.Runtime, output_dict, branching_rule, treatment_assignments

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

        obj, mip_gap, mip_runtime, output, rule, assignments = J_PT(train_set, covariates, treatment, version, fold)
        
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
    
        if not os.path.isdir(os.path.join('Output',version,str(depth))):
            os.mkdir(os.path.join('Output',version,str(depth)))
            print('\nDepth File Created')
        else:
            print('Depth File Exists')
                            
        run_experiment(version)