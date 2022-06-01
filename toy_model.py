# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:01:21 2020
@author: Jakob
"""

###########################################################
### Imports
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import time 
from multiprocessing import Pool


### Global parameter inputs

n = 10 # Number of agents
delta = 0.3  # weight placed on indirect links
gamma = 0.8  # weight placed on additional utility derived from a mutual link
c = 0.02  # cost of forming and maintaining links
b = 0.7  # strength of preference for links to similar agents
sigma = 0.01  # standard deviation of the shocks to utility
alpha = 2  # convexity of costs (cost=c*d_i**alpha)
p_link_0 = 0.25  # Uniform initial link probability (to generate g_0)
beta = 0.1
rho = 0.2
omega = 0.05
z = 0.6

# shares of the types
share_red = 1/3
share_blue = 1/3
share_green = 1 - share_red - share_blue
possible_X = [1,2,3] # coding of the types
possible_Y = [1,2,3]

# Simulation parameters
T = 5000 # Maximum iterations
t_plot = 20 # periods between plots
t_conv = 10 # if g does not change for t_conv periods we have reached convergence

### Functions

def u(i, j, X, Y) :
    """ Returns the partial utility given X_i and X_j using the exp(-b*L1-norm
    of their difference)"""
    return math.exp(-b * np.linalg.norm((X[i] - X[j]), ord=1)) * math.exp(- z * np.linalg.norm((1- (Y[i] - Y[j]), ord=1)))
## where do we get X from? 


def U(i, g, X, Y ) :
    """ Returns the full utility of agent i given the current network structure
    g and the matrix of characteristics X """
    d_i = sum(g[i])  # degree of i

    direct_u = sum([g[i, j] * u(i, j, X) for j in range(n)])

    mutual_u = sum([g[i, j] * g[j, i] * u(i, j, X) for j in range(n)])

    indirect_u = 0
    for j in range(n) :
        for k in range(n) :
            if k == i or k == j :
                continue
            else :
                indirect_u += g[i, j] * g[j, k] * u(i, k, X)

    popularity_u = 0
    for j in range(n):
        for k in range(n):
            if k == i or k == j:
                continue
            else:
                popularity_u += g[i, j] * g[k, i] * u(k, j, X)

    fr_in_common_u= 0
    for j in range(n):
        for k in range(n):
            if k == i or k == j:
                continue
            else:
               fr_in_common_u += g[i, j] * (g[j, k] + g[k, j]) * (g[k, i] + g[i, k]) * u(i, j, X) * u(j, k, X) * u(k, i, X)

    return direct_u + gamma * mutual_u + delta * indirect_u + beta * popularity_u + rho * fr_in_common_u - d_i ** alpha * c

def V(i, g, X):

    cost = 0
    for i in range(n):
        cost += alpha * c * sum(g[i])  # degree of i
    direct_u = 0
    for i in range(n):
        for j in range(n):
            direct_u += g[i, j] * u(i, j, X)
    mutual_u = 0
    for i in range(n):
        for j in range(n):
            mutual_u += g[i, j] * g[j, i] * u(i, j, X)

    indirect_u = 0
    for i in range(n):
        for j in range(n) :
            for k in range(n) :
                if k == i or k == j :
                    continue
                else :
                    indirect_u += g[i, j] * g[j, k] * u(i, k, X)

    popularity_u = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if k == i or k == j:
                    continue
                else:
                    popularity_u += g[i, j] * g[k, i] * u(k, j, X)

    fr_in_common_u= 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if k == i or k == j:
                    continue
                else:
                    fr_in_common_u += g[i, j] * (g[j, k] + g[k, j]) * (g[k, i] + g[i, k]) * u(i, j, X) * u(j, k, X) * u(k, i, X)

    return direct_u + gamma * mutual_u + delta * indirect_u + beta * popularity_u + rho * fr_in_common_u/3 - cost


def step(g, X) :
    """ Randomly selects an agent i to revise their link with another random
    agent j. Returns the updated adjacency matrix and link revision tuple (i,j)
    as well as the outcome of the revision """
    i,j  = np.random.choice(range(n), size=2, replace=False)
    g_ij_initial = g[i,j]

    eps = np.random.normal(scale=sigma, size=2)  # Simulate two shocks from normal with std dev sigma

    g[i,j] = 1
    U_with_link = U(i,g,X) + eps[0]
    
    g[i,j] = 0
    U_without_link = U(i,g,X) + eps[1]
    
    if U_with_link > U_without_link :
        g[i,j] = 1
        
    if U_with_link == U_without_link :
        g[i,j] = g_ij_initial
        
    if U_with_link < U_without_link :
        g[i,j] = 0
    
    formed = 0
    if g[i,j] > g_ij_initial:
        formed = 1
    if g[i,j] < g_ij_initial:
        formed = 2
        
    return g, (i,j), formed


def analyse(connectivity, characteristics):
    """
        Calculate mean and standard deviation (output in tuples [mean,sd]) of 
        the following network measures:
        
        - degree
        - mut_prop
        - cluster_coef
        - segreg_ind    > list with length of the amount of characteristics (3 for sex, race, grade)
        
        INPUT: 
        - connectivity matrix with row-students nominating column-students as friends
        - characteristics matrix with row per student, with integers indicating every group for each characteristic (sex, race, grade)
    """
    
    # get amount of nodes and list of out going dyads for every individual
    nodes = connectivity.shape[0]
    out_d = np.count_nonzero(connectivity, axis=1)
    ##meaning shape?
    
    # determine degree nodes (outgoing connections)
    mean_degree = np.mean(out_d)
    std_degree = np.std(out_d)
    degree = [mean_degree, std_degree]

    
    # determine the mutual dyads proportion
    # create matrix with 2's on mutual dyads, 1's on asymmetric dyads and count occurrence
    added_up = connectivity + np.transpose(connectivity)
    mutual_d = np.count_nonzero(added_up == 2, axis=1)
    mut_prop = mutual_d / out_d
    # remove 'nan' individuals (with no out-going connections) from list
    mut_prop = [value for value in mut_prop if not math.isnan(value)]
    # calculate mean+std mutual dyads proportion
    mean_mut_prop = np.mean(mut_prop)
    std_mut_prop = np.std(mut_prop)
    mut_prop = [mean_mut_prop, std_mut_prop]
    
    
    # determine the local clustering coefficient
    clustering_coefficients = []
    for n_node, connections in enumerate(connectivity):
        # the amount of neighbours each node has
        n_neighbours = np.sum(connectivity[n_node])
        # only consider nodes with at least 2 neighbours
        if n_neighbours >= 2:
            # matrix of the nodes that are both neighbours of the node considered
            neighbour_matrix = np.dot(np.transpose([connectivity[n_node]]),[connectivity[n_node]])
            # the amount of connections between neighbours
            neighbour_connections = np.sum(connectivity*neighbour_matrix)
            # the amount of connections between neighbours divided by the possible amount of connections
            clustering_coefficients.append(neighbour_connections / (n_neighbours*(n_neighbours-1)))
    # calculate mean+std clustering coefficient
    mean_cluster_coef = np.mean(clustering_coefficients)
    std_cluster_coef = np.std(clustering_coefficients)
    cluster_coef = [mean_cluster_coef, std_cluster_coef]

    
    # determine the segregation index per characteristic (sex, race, grade)
    segreg_ind= []
    # iterate through different characteristics (sex, race, grade)
    for i in range(characteristics.shape[1]):
        # get different groups of this characteristic in dataset
        characs = sorted(list(set(characteristics[:,i])))
        amount = len(characs)
        # for every characteristic own tuple for mean and std
        segreg_ind_charac = []
        # iterate through different groups of this characteristic
        for j in range(amount):
            # indicate indices of members this group and save size group
            indices = np.where(characteristics[:,i] == characs[j])[0]
            # calculate ratio out-group individuals
            ratio_diff = 1 - len(indices) / nodes
            # create a submatrix of all nominations from this group and save amount
            submat_trait = connectivity[np.ix_(indices,)]
            # create submatrix outgoing connections to individuals different group
            mask = np.ones(connectivity.shape[0], np.bool)
            mask[indices] = 0
            submat_diff = submat_trait[:,mask]
            # calculate segregation index per individual of this group for this characteristic
            for ind in range(len(indices)):
                expect_out = submat_trait[ind].sum() * ratio_diff
                observ_out = submat_diff[ind].sum()
                seg_ind = (expect_out - observ_out) / expect_out
                if seg_ind < -1:
                    seg_ind = -1
                segreg_ind_charac.append(seg_ind)
        # remove 'nan' individuals from list
        segreg_ind_charac = [value for value in segreg_ind_charac if not math.isnan(value)]
        # calculate mean+std segregation index this characteristic
        mean_segreg_ind_charac = np.mean(segreg_ind_charac)
        std_segreg_ind_charac = np.std(segreg_ind_charac)
        segreg_ind.append([mean_segreg_ind_charac, std_segreg_ind_charac])


    return degree, mut_prop, cluster_coef, segreg_ind[0]


def nx_graph(g):
    ''' Returns an nx graph for an adjacency matrix g '''
    rows, cols = np.where(g == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph()  # Calling the DIRECTED graph method
    gr.add_nodes_from(range(len(g)))
    gr.add_edges_from(edges)

    return gr


def plot_network(g,ij=None,formed=None,node_positions=None,ax=None) :
    """ Uses nx to plot the directed network g """
    gr = nx_graph(g)
    # Add node colors according to X
    color_map = []
    for i in range(n) :
        if X[i,0] == possible_X[0]:
            color_map.append('red')
        if X[i,0] == possible_X[1]:
            color_map.append('blue')
        if X[i,0] == possible_X[2]:
            color_map.append('green')
#    pos = nx.spring_layout(gr)
    pos = nx.circular_layout(gr)
    nx.draw(gr, pos, node_color=color_map, with_labels=True, node_size=300, 
            arrowsize=20, ax=ax)
    
    if ij != None:
        if formed == 0:
            edge_color = 'b'
        if formed == 1:
            edge_color = 'g'
        if formed == 2:
            edge_color = 'r'
            
        nx.draw_networkx_edges(gr, pos, edgelist=[ij], edge_color=edge_color, 
                               arrowsize=20, width=3, ax=ax)


def plot_network_and_stats(g,X,stats_list,ij,formed,t,node_positions=None):
    ''' Plots the network and network statistics '''
    k = len(stats_list[0]) # number of network statistics

    fig = plt.figure(figsize=(15, 5))
    grid = plt.GridSpec(2, k, wspace=0.4, hspace=0.3)
    main_ax = fig.add_subplot(grid[0,:])
    small_axs = [fig.add_subplot(grid[1, i]) for i in range(k)]
    
    if ij == None:
        plot_network(g,node_positions=node_positions,ax=main_ax)
    else:
        plot_network(g,ij,formed,node_positions=node_positions,ax=main_ax)

    statistic_names = ['Avg. degree', 'Mutuality', 'Clustering', 'Segregation']

    for i in range(k):
        small_axs[i].plot(stats_list[0][i][0],'ro')
        small_axs[i].plot([stats[i][0] for stats in stats_list])
        small_axs[i].set_title(statistic_names[i])
        
    plt.suptitle('Network configuration at t={}'.format(t))
    print(f'The potential at this time is {V(n, g, X)}')
#    plt.savefig('toy_evolution' + str(t) + '.png') # save figures
    plt.show()

### Simulation
    
# Generate proportional green blue and reds
X = np.zeros((n,2))
X[:,0] = np.array([possible_X[0] for i in range(int(share_red * n))] +
             [possible_X[1] for i in range(int(share_blue * n))] +
             [possible_X[2] for i in range(n - int(share_red * n) - int(share_blue * n))])

# Randomly generate the initial network configuration
g_0 = np.random.choice([0, 1], size=(n, n), p=[1 - p_link_0, p_link_0])
np.fill_diagonal(g_0, 0)  # The diagonal elements of the adjacency matrix are 0 by convention
g_sequence = [g_0]  # Sequence of adjacency matrices

# Initialize lists to save results    
stats_list = [analyse(g_sequence[-1],X)]
ij_list = [None]
formed_list = [None]


for t in range(T - 1):
    # Perform a step and append the new network
    g_new, ij, formed = step(g_sequence[-1], X)
    g_sequence.append(g_new.copy())
    ij_list.append(ij)
    formed_list.append(formed)
    
    # Analyze new network
    stats = analyse(g_sequence[-1],X)
    stats_list.append(stats)
        
    if t % t_conv == 0 and t // t_conv > 0:
        # Calculate how many links have changed and stop if convergence has been reached
        network_change = np.linalg.norm((g_sequence[-1] - g_sequence[-t_conv]), ord=1)
        if network_change == 0:
            convergence_steps = t
            break
            
for t in range(len(g_sequence)-1):
    # Produce a plot and diagnostics every t_plot steps
    if t % t_plot == 0:
        plot_network_and_stats(g_sequence[t],X,stats_list[:t+1],ij_list[t+1],formed_list[t+1],t)


# Plot final network
plot_network_and_stats(g_sequence[-1],X,stats_list,None,None,len(g_sequence))

        
print('It took {} setps until convergence'.format(convergence_steps))

