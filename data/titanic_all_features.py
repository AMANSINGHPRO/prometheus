#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:13:52 2019

@author: aaman10
"""



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import matplotlib.pyplot as plt
import scikitplot as skplt

###############################################################################
# Read data file
###############################################################################
TRAIN_FILE = '/Users/aaman10/Desktop/Kaggle/titanic/train.csv'
TRAIN_ = pd.read_csv(TRAIN_FILE)

###############################################################################
# Remove nan
###############################################################################
TRAIN_ = TRAIN_.dropna(subset=['Embarked'])

###############################################################################
# Labelencoding for categorial features
###############################################################################
lb_make = LabelEncoder()
TRAIN_['SEX_CODE'] = lb_make.fit_transform(TRAIN_['Sex'])
TRAIN_['EMBARKED_CODE'] = lb_make.fit_transform(TRAIN_['Embarked'])
TRAIN_ = TRAIN_.drop(['Sex', 'Embarked'], axis = 1)

###############################################################################
# Seperate Target variable
###############################################################################
X = TRAIN_.drop('Survived', axis=1)
Y = TRAIN_['Survived']
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.2,
                                                    stratify=Y)

# Specify categorial features
cf = ['SEX_CODE', 'EMBARKED_CODE']
LGBM = lgb.LGBMClassifier(silent=False)
LGBM.fit(X, Y, categorical_feature=cf)

# Predict
YY = LGBM.predict(X_TEST)

###############################################################################
# Measure Performance
###############################################################################

# Accuracy
print(accuracy_score(Y_TEST, YY))

# Feature Importance
FEATURE_IMP = pd.DataFrame(zip(LGBM.feature_importances_, X.columns), 
                           columns=['Value','Feature']).sort_values(by='Value',
                                   ascending=False)
print(FEATURE_IMP)

# Confusion matrix
print(confusion_matrix(Y_TEST, YY))

# Classification Report
print(classification_report(Y_TEST, YY))

# ROC Curve
Y_PRED_PROB = LGBM.predict_proba(X_TEST)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_TEST, Y_PRED_PROB)

PROBAS = LGBM.predict_proba(X_TEST)

# Precision Recall Curve plot
skplt.metrics.plot_precision_recall_curve(Y_TEST, PROBAS)

# ROC Curve plot
skplt.metrics.plot_roc(Y_TEST, PROBAS)

# Feature Importance plot
lgb.plot_importance(LGBM)
