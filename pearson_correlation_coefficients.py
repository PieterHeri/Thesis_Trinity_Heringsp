# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 22:28:11 2021

@author: piete
"""
import pandas as pd
import numpy as np
import scipy as sc
from scipy import stats

filename = 'redo_MoCA_for_pearson_L.xlsx'
df = pd.read_excel(filename, header=None)

dbase = df[1:] #chop header off
x = np.array(dbase[5])


r2_list = []
for i in range (19, 27):
    cogn = np.array(dbase[i])
    stat = sc.stats.pearsonr(x, cogn)
    r2_list.append(stat[0])
    

print(r2_list)




