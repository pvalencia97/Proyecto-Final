#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:56:48 2019

@author: saul
"""
import numpy as np
import pandas as pd

#import seaborn as sns
#sns.set_style("darkgrid")
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import sklearn.linear_model as lm
import sklearn.discriminant_analysis as da
import sklearn.neighbors as nb
from sklearn.metrics import confusion_matrix, classification_report, precision_score, accuracy_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

#cargar el conjunto de datos
Weekly = pd.read_csv('Weekly.csv')
print(Weekly.head())

######mostrar datos estadisticos#####

#correlacion entre predictores
cm = Weekly.corr()
print(cm)
