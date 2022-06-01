
import matplotlib.pyplot as plt
import os
import pickle
from SALib.analyze import sobol
import numpy as np

_var = ['Delta', 'Gamma', 'C', 'Sigma', "b_1", 'b_2', 'b_3']

problem = {
        'num_vars': len(_var),
        'names': _var,
        'bounds': [[0, 3],
                    [0, 1],
                    [0, 3],
                    [0, 1],
                    [0, 3],
                    [0, 3],
                    [0, 3]]
    }

settings =  [0.05, 0.65, 0.175, 0.035, 0.1, 0.1, 0.2]

label = {'av_degree':'average degree',
         'mut_prop':'propotion of mutual dyads',
         'clustering_coefficient':'clustering coefficient',
         'segreg_ind2':'segregation grade',
         'segreg_ind1':'segregation race',
         'segreg_ind0':'segregation sex'}

_string = ['av_degree', 'mut_prop', 'clustering_coefficient', 'segreg_ind2', 'segreg_ind1', 'segreg_ind0']


# Uncertainty analysis
data = pickle.load(open('pickles\\100xschool4.pkl', 'rb'))

fig = plt.figure()
plt.boxplot(data.values(), labels=[label[i] for i in _string])
plt.ylim([0, 1])
plt.xticks(rotation=30, fontsize=13)

plt.ylabel('Normalized output', fontsize=13)
plt.title('Uncertainty analysis', fontsize=16)
plt.tight_layout()
fig.savefig('UA_plot.png')

# Experiment
extraruns = pickle.load(open('pickles\\100xschool4extra.pkl', 'rb'))
normalruns = pickle.load(open('pickles\\100xschool4.pkl', 'rb'))

fig = plt.figure()
plt.boxplot([normalruns['segreg_ind0'], extraruns['segreg_ind0'],
             normalruns['segreg_ind1'], extraruns['segreg_ind1'],
             normalruns['segreg_ind2'], extraruns['segreg_ind2']],
            labels=['norm ind0', 'extra ind0', 'norm ind1', 'extra ind1', 'norm ind2', 'extra ind2'])
plt.xticks(rotation=45)
plt.ylabel('Segregation value')
plt.title('Extra Charcaristics')
plt.tight_layout()
fig.savefig('ExtraChar.png')


# SOBOL
data = pickle.load(open('pickles\\SA_sobol.p', 'rb'))
para = pickle.load(open('pickles\\Par_sobol.p', 'rb'))
ind, Y = zip(*data)

for num in range(len(_string)) :
    Si = sobol.analyze(problem, np.array([i[num] if i[num] == i[num] else 0 for i in Y]))
    fig = plt.figure()
    plt.bar(_var, Si['S1'], yerr=Si['S1_conf'])
    plt.ylabel(label[_string[num]], fontsize=16)
    plt.title('First order sobol', fontsize=18)
    plt.tight_layout()
    fig.savefig(os.path.join('SA_figures', _string[num] + 'frist.png'))

    fig = plt.figure()
    plt.bar(_var, Si['ST'], yerr=Si['ST_conf'])
    plt.ylabel(label[_string[num]], fontsize=16)
    plt.title('Total order sobol', fontsize=18)
    plt.tight_layout()
    fig.savefig(os.path.join('SA_figures', _string[num] + 'total.png'))

plt.show()