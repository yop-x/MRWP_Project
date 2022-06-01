# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:54:22 2020

@author: Jakob
"""

import time
import pandas as pd
import pickle
import numpy as np
from multiprocessing import Pool
from analyse_network import analyse
from fast_model import run
from scipy import optimize

g_list = pickle.load(open(r"g_list.pkl", 'rb'))
X_list = pickle.load(open(r"x_list.pkl", 'rb'))

stat_names = ['degree', 'mut_prop', 'clustering', 
              'seg_sex', 'seg_race', 'seg_grade']
means = [stat_name + '_m' for stat_name in stat_names]
stds = [stat_name + '_std' for stat_name in stat_names]

# good paramter values
DELTA = 0.3
GAMMA = 0.5
C = 0.25
B1, B2, B3 = 0.05, 0.07, 0.09
#[weight/np.sum(weights) for weight in weights] # so weights add up to 1
settings = DELTA, GAMMA, C, B1, B2, B3
setting_names = ['DELTA', 'GAMMA', 'C', 'B1', 'B2', 'B3']


def sim(settings,X,n_iter):
    ''' Returns outcome measures of network simulation over n_iter iterations'''
    g_sim = run(settings,X)
    res = analyse(g_sim,X)
    for i in range(n_iter):
        g_sim = run(settings,X)
        res = res.append(analyse(g_sim,X), ignore_index=True)

    return res

ref_schools = [1,4,6] # schools used for validation

def validation_run(settings):
    ''' Returns a dataframe with differences in outcome measures (mean + std) 
    between simulated and observed networks '''
    k = 0
    for school in ref_schools:
        X = X_list[school]
        g = g_list[school]
    #    print('School {} with {} students:'.format(school,n_agents))
    
        n_sim = int(1/len(X)*200) # simulates more times for smaller schools
        
        n_sim = 1
        mean_res = pd.DataFrame(np.mean(sim(settings,X,n_sim))).T # average over n_sim simulations
        
        true_res = analyse(g,X)
        
        diff = true_res-mean_res
        diff['school'] = school
        
        if k == 0:
            diffs = diff
        else:
            diffs = diffs.append(diff.copy(), ignore_index=True)
            
        k += 1
        
    diffs.set_index(['school'], inplace=True)
    
#    obj = np.linalg.norm(diffs) # Euclidean distance of sim vs true g

#    print('Differences for {}:\n {} \n {}'.format(
#            [(setting_names[i],settings[i]) for i in range(len(setting_names))],
#            diffs, obj))
    
    return diffs


## Specify ranges to try for DELTA, GAMMA, C, B1, B2, B3 (in that order)
#rranges = (0.2, 0.4), (0.4,0.6), (0.2,0.3), (0.05,0.1), (0.05,0.1), (0.05,0.1)
#
## Grid search (Ns = number of values to try for each par)
#optimize.brute(validation_run, rranges, Ns=3, disp=True, workers=-1)


#def network_score(g_obs,g_sim):
#    n = len(g_obs)
#    g_diff = g_obs - g_sim
#    false_neg = np.count_nonzero(g_diff == 1)
#    false_pos = np.count_nonzero(g_diff == -1)
#    total_err = false_neg + false_pos
#    share_correct = 1-total_err/(n*(n-1))
#    n_obs_links = np.count_nonzero(g_obs == 1)
#    false_neg_rel = false_neg/n_obs_links
#    n_no_link = np.count_nonzero(g_obs == 0)
#    false_pos_rel = false_pos/n_no_link
##    total_err_per_node = total_err/n
#    return false_neg_rel

#def sim_score(settings,X,n_iter):
#    ''' Returns outcome measures of network simulation over n_iter iterations'''
#    score = []
#    for i in range(n_iter):
#        g_sim = run(settings,X)
#        score.append(network_score(g_obs,g_sim))
#
#    return np.mean(score)
    
#for school in ref_schools:
#    X = X_list[school]
#    n_agents = len(X)
#    if n_agents > 200:
#        continue
#    g_obs = g_list[school]
#    
#    n_sim = int(1/len(X)*200)
#    
#    print(sim_score(settings,X,n_sim))