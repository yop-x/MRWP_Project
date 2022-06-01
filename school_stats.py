# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 08:17:22 2020

@author: Jakob
"""
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

X_list = pickle.load(open(r"x_list.pkl", 'rb'))

school_list = [1,4,6,7,16,52,69]


X_conc = X_list[school_list[0]]
X_conc['community'] = school_list[0]

for com in school_list[1:]:
    X = X_list[com]
#    if all(X.school==1) and len(X) < 400:
    X['community'] = com
    X.grade.replace(6,7, inplace=True)
    X_conc = X_conc.append(X)

X_conc.groupby(['community']).sex.value_counts(normalize=True).iloc[:2]

yticks = [i/10 for i in range(0,12,2)]
fig, ax = plt.subplots(3,1, figsize=(8, 12))
for school in school_list:
    linestyle = '-'
    if school > 6:
        linestyle = '--'
    grp = X_conc.groupby(['community'])
    ax[0].plot(grp.sex.value_counts(normalize=True).loc[school].values,
            linestyle=linestyle, marker='o', label='School {} (n={})'.format(school,len(X_list[school])))
    ax[1].plot(grp.race.value_counts(normalize=True).loc[school].values,
            linestyle=linestyle, marker='o')
    grade_count = grp.grade.value_counts(normalize=True).loc[school].sort_index()
    for grade in [7,8,9,10,11,12]:
        if grade not in grade_count.index:
            grade_count = grade_count.append(pd.DataFrame([0], index=[grade]))
    grade_count.sort_index(inplace=True)
    ax[2].plot(grade_count,
        linestyle=linestyle, marker='o')
ax[0].set_xticks([0,1])
ax[0].set_xticklabels(['male','female'])
ax[1].set_xticks([0,1,2,3,4])
ax[1].set_xticklabels(['white','black','hispanic','asian','mixed/other'])
for axis in ax:
    axis.set_yticks(yticks)
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=int(len(school_list)/2)+1)
plt.savefig('school_stats.png', bbox_inches='tight', dpi=300)
plt.show()