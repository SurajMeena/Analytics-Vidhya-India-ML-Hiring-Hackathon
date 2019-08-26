# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 23:59:42 2019
@author: Suraj
"""
import math
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
# Observation from dataset: all 1 valued target rows are in starting of the data
# so we will shuffle the data
df = df.sample(frac=1).reset_index(drop=True)
#Checking if shuffling was done successfully
idx_1 = np.where(df.m13 == 1) # Result : gives us the tuple of indices of m13=1
idx_0 = np.where(df.m13 == 0) # so basically you did good work

ntrain = df.shape[0]
ntest = df_test.shape[0]
y_train = df.m13.values
all_data = pd.concat([df, df_test], sort=False).reset_index(drop=True)
all_data.drop(['m13'], axis=1, inplace=True)

df['m13'].value_counts()  # Result: Classes are unbalanced only 636 1 classes
df['m13'].value_counts()  # Result: Classes are unbalanced only 636 1 classes

#z = df.select_dtypes(['object']) #this is used to get dataframe with only those columns which have datatype object

#This code can be used for finding correlation between categorical variables
def conditional_entropy(x, y):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    **Returns:** float
    Parameters
    ----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    """
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy
def theils_u(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x
corr = theils_u(all_data['source'], all_data['loan_purpose'])
#No significant correlations were obtained
    
corr = all_data.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
pos_filtered_corr = corr[ corr.iloc[:,:] >= 0.5] # from this we easily know highly positively correlated values
neg_filtered_corr = corr[ corr.iloc[:,:] <= -0.5] # no attribute is that highly negatively correlated
sns.heatmap(pos_filtered_corr,
            xticklabels=pos_filtered_corr.columns.values,
            yticklabels=pos_filtered_corr.columns.values) #only of the attributes which are correlated highly


df.isnull().sum()    # Result: No missing values
df_test.isnull().sum()    # Result: No missing values
df.info()
#all_data = all_data.drop(["co-borrower_credit_score", "m9", "m10", "m11", "m12"], axis=1) # based on correlation
all_data = all_data.drop(["origination_date", "co-borrower_credit_score"], axis=1) # based on correlation

all_data['first_payment_date'].unique() # Will treat it as categorical variable
all_data.replace("Apr-12", "04/2012", inplace = True)
all_data.replace("Mar-12", "03/2012", inplace = True)
all_data.replace("May-12", "05/2012", inplace = True)
all_data.replace("Feb-12", "02/2012", inplace = True)
all_data['loan_purpose'].unique()  # this is a categorical variable
all_data['source'].unique()  # this is a categorical variable
all_data['financial_institution'].value_counts()
#all_data['origination_date'].value_counts()
all_data.replace("2012-02-01", "01/02/12", inplace = True)
all_data.replace("2012-01-01", "01/01/12", inplace = True)
all_data.replace("2012-03-01", "01/03/12", inplace = True)
# convert all numerical valued categorical variables into str
all_data['loan_purpose'] = all_data['loan_purpose'].astype(str)
all_data['first_payment_date'] = all_data['first_payment_date'].astype(str)
all_data['financial_institution'] = all_data['financial_institution'].astype(str)
#all_data['origination_date'] = all_data['origination_date'].astype(str)
all_data['source'] = all_data['source'].astype(str)

Finalall_data = pd.get_dummies(all_data)
#we will do this below step only if a model is affected badly my dummy variables
#Finalall_data = Finalall_data.drop(["source_X", "financial_institution_OTHER", "loan_purpose_A23", "first_payment_date_02/2012", "origination_date_01/01/12"], axis=1) 
train = Finalall_data[:ntrain]
test = Finalall_data[ntrain:]

X = train.drop(["loan_id"], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#let's perform feature scaling
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

#sorted_df = df.loc[(df['loan_term'] == 240)]
#sorted_df = sorted_df.loc[(sorted_df['m9'] == 1) | (sorted_df['m11'] == 1) | (sorted_df['m10'] == 1) | (sorted_df['m12'] == 1)]
#sorted_df['loan_term'].value_counts()
#df['m7'].value_counts()
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
#you can do feature engineering inspired from housing price predictions like 
#log transformations on numerical attributes
#Correlated features
#borrower credit sccore and number of borowers
#m8 and m9
#m9 and m10
#m9 , m10 and m11
#m10, m11 and m12