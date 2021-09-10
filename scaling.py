#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 09:15:08 2021

@author: carolyndavis
"""

# =============================================================================
#                             SCALING EXERCISES
# =============================================================================

#IMPORTS USED FOR LESSON EXERCISES:
import pandas as pd
import numpy as np
import env
import wrangler2
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.model_selection import train_test_split

telco_df = wrangler2.get_telco_data()
telco_df = telco_df.drop_duplicates(subset=["customer_id"]).reset_index(drop=True)
#drop dupes bc of faulty SQL query


telco_df.head(10)
#

telco_df.columns

telco_df.info() 
quant_df = telco_df[['customer_id', 'tenure', 'monthly_charges', 'total_charges']].copy()
#this helps with later data manipulation
quant_df




# split the data in train, validate and test
train, test = train_test_split(quant_df, test_size = 0.2, random_state = 123)
train, validate = train_test_split(train, test_size = 0.25, random_state = 123)



#Looking at the shape:
train.shape, validate.shape, test.shape 
  #output: ((4225, 4), (1409, 4), (1409, 4))  
train.head()
# =============================================================================
# 1.)Apply the scalers we talked about in this lesson to your data and visualize
#  the results for the unscaled and scaled distribution .
# =============================================================================


#DEFINE THE THANG

scaler = sklearn.preprocessing.MinMaxScaler()

# Fit the thing
scaler.fit(train[['monthly_charges']])

#transform
scaled_month = scaler.transform(train[['monthly_charges']])


# single step to fit and transform
scaled_month = scaler.fit_transform(train[['monthly_charges']])

#Add a new scaled col to original train df 

train['scaled_month_charges'] = scaled_month
train.head()

                    #######  NOW VISUALIZE  #########
#plotting the total_charges and the scaled total charges...

plt.scatter(train.monthly_charges, scaled_month)
plt.xlabel('Monthly_Charges')
plt.ylabel('Scaled_Monthly_Charges')


                ##### Plotting the Distributuon of Monthky Charges ###########
plt.hist(train.monthly_charges)

                ###Plotting distrbution of monthly charges with scaled data
plt.hist(scaled_month)



fig = plt.figure(figsize = (12,6))

gs = plt.GridSpec(2,2)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1])

ax1.scatter(train.monthly_charges, scaled_month)
ax1.set(xlabel = 'Monthly_Charges', ylabel = 'Scaled_Monthly_Charges', title = 'Min/Max Scaler')

ax2.hist(train.monthly_charges)
ax2.set(title = 'Original')

ax3.hist(scaled_month)
ax3.set(title = 'Scaled_Charges')
plt.tight_layout();


#######################  USING MIN/MAX SCALER ####################################
def visualize_scaled_date(scaler, scaler_name, feature):
    scaled = scaler.fit_transform(train[[feature]])
    fig = plt.figure(figsize = (12,6))

    gs = plt.GridSpec(2,2)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    ax1.scatter(train[[feature]], scaled)
    ax1.set(xlabel = feature, ylabel = 'Scaled_' + feature, title = scaler_name)

    ax2.hist(train[[feature]])
    ax2.set(title = 'Original')

    ax3.hist(scaled)
    ax3.set(title = 'Scaled_Charges')
    plt.tight_layout();
    
    
    
# use the function created above for monthly charges

visualize_scaled_date(sklearn.preprocessing.MinMaxScaler(), 'Min_Max_Scaler', 'monthly_charges')



##############   VISUALIZE: Scaling Data for Tenure:
visualize_scaled_date(sklearn.preprocessing.MinMaxScaler(), 'Min_Max_Scaler', 'tenure')    



######################  VISUALIZE WITH STANDARD SCALER ####################
# visualize scaling for monthly charges

visualize_scaled_date(sklearn.preprocessing.StandardScaler(), 'Standard_Scaler', 'monthly_charges')


# visualize scaling for tenure 
visualize_scaled_date(sklearn.preprocessing.StandardScaler(), 'Standard_Scaler', 'tenure')


################### VISUALIZE WUTH ROBUST SCALER ###########################

visualize_scaled_date(sklearn.preprocessing.RobustScaler(), 'Standard_Scaler', 'monthly_charges')


#################### STANDARD SCALER: Tenure
visualize_scaled_date(sklearn.preprocessing.StandardScaler(), 'Standard_Scaler', 'tenure')


# =============================================================================
# 2.)Apply the .inverse_transform method to your scaled data. Is the resulting
#  dataset the exact same as the original data?
# =============================================================================

train.head()

scaler = sklearn.preprocessing.MinMaxScaler()
#fitting the scaled data to the train set
scaled = scaler.fit_transform(train[['monthly_charges', 'tenure']])
scaled

#output: array([[0.61791045, 0.65277778],
       # [0.65771144, 0.68055556],
       # [0.85273632, 0.75      ],
       # ...,
       # [0.87412935, 0.29166667],
       # [0.66268657, 0.55555556],
       # [0.34726368, 0.29166667]])



#make it into a df for manipulation:
scaled_df = pd.DataFrame(scaled, index = train.index, columns = ['monthly_charges', 'tenure'])
scaled_df.head()
# #output:       monthly_charges    tenure
# 440          0.617910  0.652778
# 67           0.657711  0.680556
# 600          0.852736  0.750000
# 4883         0.662189  0.013889
# 1258         0.023881  0.666667


#   USING THE INVERSE TRANFORM METHOD #########
scaler.inverse_transform(scaled_df)

# #output array([[ 80.35,  47.  ],
#        [ 84.35,  49.  ],
#        [103.95,  54.  ],
#        ...,
#        [106.1 ,  21.  ],
#        [ 84.85,  40.  ],
#        [ 53.15,  21.  ]])


#inverse produced array, changing to DataFrame:
unscaled_df = pd.DataFrame(scaler.inverse_transform(scaled), index = train.index, columns = ['monthly_charges', 'tenure'])
unscaled_df.head()
#output:
#       monthly_charges  tenure
# 440             80.35    47.0
# 67              84.35    49.0
# 600            103.95    54.0
# 4883            84.80     1.0
# 1258            20.65    48.0



# =============================================================================
# 3.)Read the documentation for sklearn's QuantileTransformer. Use normal for the
#  output_distribution and apply this scaler to your data. Visualize the result of
#  your data scaling.
# =============================================================================

#visualize monthly charges quantile transformation with /'normal'/ output

visualize_scaled_date(sklearn.preprocessing.QuantileTransformer(output_distribution='normal'), 'Quantile Scaler', 'monthly_charges')

#visualize monthly charges quantile transformation with /'uniform'/ output



#######################  VIZ FOR TENURE

visualize_scaled_date(sklearn.preprocessing.QuantileTransformer(output_distribution='normal'), 'Quantile Scaler', 'tenure')



# =============================================================================
# 4.)Use the QuantileTransformer, but omit the output_distribution argument.
#  Visualize your results. What do you notice?
# =============================================================================

####################### VIZ FOR MONTHLY_CHARGES
visualize_scaled_date(sklearn.preprocessing.QuantileTransformer(), 'Quantile Scaler', 'monthly_charges')

# =============================================================================
#                             SCALING TAKEAWAYS:
# =============================================================================
#   -HANDLE the outlier first unlessyou establish to use nonlinear
#   -MIN/MAX scaler transforms each valuue in the col within desireable range of (0,1)
#               (USE THIS AS YOUR FIRST CHOICE TO SCALE. PRESERVES SHAPE OF DISTRIBUTION..NO DISTORT)
#   -STANDARD SCALER transforms each value in the col to range mean of 0 and std of 1
#               (USE ONLY IF YOU KNOW DATA IS NORMALLY DISTRIBUTED)
#   -ROBUST SCALER: Have outlier you dont want to discard. USE ROBUST
#               (ALTERNATIVELY remove the outliers and use the two scaling methods above)
#   GOOD PRACTICE: visualize the distribution of vars after scaling.. make sure tranformation actually happened
#   USE NONLINEAR scalers when you realy have to (quanitle transformer, when u must have normally dist data)





# =============================================================================
# 5.)Based on the work you've done, choose a scaling method for your dataset.
#  Write a function within your prepare.py that accepts as input the train, validate,
#  and test data splits, and returns the scaled versions of each. Be sure to only learn
#  the parameters for scaling from your training data!
# =============================================================================
def Standard_Scaler(X_train, X_validate, X_test):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs
    """

    scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index, columns = X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    
    return scaler, X_train_scaled, X_validate_scaled, X_test_scaled



def Min_Max_Scaler(X_train, X_validate, X_test):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs 
    """
    scaler = sklearn.preprocessing.MinMaxScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index, columns = X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    
    return scaler, X_train_scaled, X_validate_scaled, X_test_scaled


X_train = train[['monthly_charges', 'tenure']]
X_validate = test[['monthly_charges', 'tenure']]
X_test = test[['monthly_charges', 'tenure']]




# Use the Standard_Scaler Function defined above

scaler, X_train_scaled, X_validate_scaled, X_test_scaled = Standard_Scaler(X_train, X_validate, X_test)

X_train_scaled.head()

# output:
#           monthly_charges    tenure
# 440          0.526056  0.599921
# 67           0.658841  0.681616
# 600          1.309488  0.885854
# 4883         0.673779 -1.279063
# 1258        -1.455763  0.640769