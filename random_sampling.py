# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:26:42 2021

@author: Pieter
"""
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

dset = pd.read_excel('sampling_mlreg.xlsx')

train, validate = train_test_split(dset, test_size=0.2)

train.to_excel('train20.xlsx', index = False)
validate.to_excel('validate20.xlsx', index=False)