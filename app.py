#!/usr/bin/env python
# coding: utf-8

# # Our goal is to build a model that predicts if the client will subscribe a term deposit.
# 

# ### Step 1: Load the dataset

# In[1]:


import pandas as pd
df = pd.read_csv('bank.csv')
df = df.dropna()
df.info()


# Here we can see that a lot of columns are shown as object data type, which means categorical. Some categorical columns need to be dropped and some categorical columns can be convert to boolean type.

# ### Step 2: Data cleaning.

# We need to replace categorical columns with boolean or numerical values in order to use them in building the model.

# In[2]:


#Drop categorical variables
df = df.drop(['job', 'marital', 'education'], axis=1)
#Replace categorical columns with boolean or numerical values
df = df.replace({'default': {'yes': 1, 'no': 0, 'unknown': 2}})
df = df.replace({'housing': {'yes': 1, 'no': 0, 'unknown': 2}})
df = df.replace({'loan': {'yes': 1, 'no': 0, 'unknown': 2}})
df = df.replace({'contact': {'cellular': 1, 'telephone': 0}})
df = df.replace({'month': {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}})
df = df.replace({'day_of_week': {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5, 'sat':6, 'sun':7}})
df = df.replace({'poutcome': {'success': 1, 'failure': 0, 'nonexistent': 2}})
df = df.replace({'y': {'yes': 1, 'no': 0}})
df.head()


# Now we have all columns with numerical values and ready to create the model.

# ### Step 3: Split the dataset into 2 parts, 60% training data, 40% validation data.
# 

# In[3]:


len(df)


# Right now, the whole dataset contains 41188 observations. We can put 24712 observations into training data, and 16476 observations into validation data.

# In[4]:


import numpy as np
rows_for_training = np.random.choice( df.index, 24712, False )
training = df.index.isin(rows_for_training)
df_training = df[training]
df_validation = df[~training]


# This code is being created by first generating 24712 random rows to df_training dataframe. And the rest will be put into df_validation dataframe. This gives us about 60% training set and 40% validation set.

# ### Step 4: Create a model using the training dataset using the scikit-learn's logistic regression tool. And test the model using the validation dataset.

# In[5]:


from sklearn.linear_model import LogisticRegression
def fit_model_to (training):
    predictors = training.iloc[:,:-1]
    response = training.iloc[:,-1]
    model = LogisticRegression(max_iter=10000)
    model.fit(predictors, response)
    return model

def score_model (M, validation):
    predictions = M.predict(validation.iloc[:,:-1] )
    TP = (validation['y'] & predictions).sum()
    FP = (~validation['y'] & predictions).sum()
    FN = (validation['y'] & ~predictions).sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return 2 * precision * recall / (precision + recall)

model = fit_model_to(df_training)


# The values retured are the F1 score from training dataset and validation dataset. F1 score represents how good our model is by using the precision and recall values. We can see that both F1 score for training and validation datasets are relatively low. Possible reason is that we have high number of predictor variables. It's likely that some of predictors do not predict our response variable well. 

# ### Step 5: Create a better model to predict the response variable.

# In[6]:


def fit_model_to (training):
    # fit the model the same way as step 4
    predictors = training.iloc[:,:-1]
    response = training.iloc[:,-1]
    model = LogisticRegression(max_iter=10000)
    model.fit(predictors, response)
    # fit another model to standardized predictors in order to compare the coefficients with the same scale
    standardized = (predictors - predictors.mean()) / predictors.std()
    standard_model = LogisticRegression(max_iter=10000)
    standard_model.fit(standardized, response)
    # get that model's coefficients and display them
    coeffs = pd.Series(standard_model.coef_[0], index=predictors.columns)
    sorted = np.abs(coeffs).sort_values( ascending=False )  # sort the coefficients from the most important to the least
    coeffs = coeffs.loc[sorted.index]                        
    print(coeffs)
    # return the model fit to the actual predictors
    return model


model = fit_model_to(df_training)


# Here the coefficients are sorted in order of their importance to predict the model. We can perform a series of trial and error to see which columns to keep in the modeling in order to perform a higher F1 score.

# In[170]:


columns = ['duration','euribor3m','pdays','y']
model = fit_model_to( df_training.loc[:,columns] )


# Here are the final predictors that we decided to keep after experimenting various combinations. 

# In[176]:


import streamlit as st
st.title( 'Term Deposit Subscription Status vs. Potential Predictors' )
st.sidebar.markdown( '''
The heatmap shows the correlation between term deposit subscription status and its potential predictors. The predictors are choosen based on the logistic regression model built to predict the subscription status, 
which is also available on https://github.com/LiyaZhang-ziqing/Bank. Last column of the heatmap is the area that need to focus on. None of the predictors has strong correlation with subscription status, 
which explains why the F1 score from the modeling is relatively low. Moreover, the correlations between each predictors are weak. This helps avoid the situation where 
similar variables are counting multiple times during the modeling process.''' )
df_final = df.loc[:, columns]


# In[174]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
correlation_coefficients = np.corrcoef(df_final, rowvar=False )
sns.heatmap( correlation_coefficients, annot=True )
plt.yticks( np.arange(4)+0.5, df_final.columns, rotation=0 )
plt.xticks( np.arange(4)+0.5, df_final.columns, rotation=90 )
plt.title('The correlation between term deposit subscription status and its predictors')
st.pyplot(plt.gcf())


# In[ ]:





# In[ ]:




