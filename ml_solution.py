#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[26]:


from sklearn.metrics import roc_curve # for predicting error y_tested and y_predicted


# In[27]:


from sklearn.metrics import auc # area under curb finding the accuracy 


# In[4]:


#dataset=pd.read_csv("Downloads/New Bank_loan_data.csv")


# In[75]:


df=pd.read_excel("Downloads/New Bank_loan_data.xlsx") #reading excell file


# In[29]:


df.head(n=5)


# In[76]:


nulls=df.isnull().sum()


# In[77]:


nulls=nulls[nulls>0]


# In[78]:


nulls


# In[79]:


per=nulls*100/df.shape[0]


# In[80]:


pd.concat([nulls, per], axis=1, 
          keys=['nulls', 'Per']).\
          sort_values(by="nulls", ascending=False)


# In[36]:


df['Gender'].fillna('M', inplace=True)   # filling null values with M 


# In[97]:


df['home_ownership'].fillna('Rent', inplace=True)# filling missing values with rent due to majority number of rent there

df['Online'].fillna(1, inplace=True)  #filling missing values with 1 due to majority number of Online there 


# In[98]:


df['Income'].fillna(df['Income'].mean(), inplace=True)# filling missing value with mean of the system


# In[81]:


# dropping the unnecessary content
del df['ID']

del df["ZIP Code"]


# In[99]:


del df['Gender']


# In[100]:



df.head(10)


# In[83]:


df.groupby('Online').Online.count()# classifing the o and 1 in the column.


# In[102]:


nulls=df.isnull().sum()
nulls


# In[84]:


df


# In[87]:


df.rename(columns={"Personal Loan": "Personal_Loan","Securities Account":"Sec_Acc","Home Ownership":"home_ownership",'CD Account':'cd_account'},inplace=True)


# In[88]:


df  # renaming  above


# In[48]:


df.dtypes  # EDA trying to identify the nature of the data


# In[50]:


df.describe() #quick summary of data

sns.countplot(data=df,x='Personal_Loan') 
# In[52]:


indexAge = df[ (df['Age'] >= 100) ].index
df.drop(indexAge , inplace=True)
indexAge = df[ (df['Experience'] < 0) ].index
df.drop(indexAge , inplace=True)
df.describe()                                          # elliminating the given row which has age beyond 100 year and experience less 
                                                        #less than 0.


# In[53]:


# visualising the image and trying to identify their relation


# In[54]:


sns.countplot(data=df,x='Personal_Loan') 


# In[55]:


plt.hist(df['Experience'])


# In[56]:


# experience is well balance here


# In[57]:


df['Experience']


# In[58]:


plt.bar(df['Experience'],df['Age'])


# In[ ]:


# increasing experience with age


# In[59]:


plt.bar(df['Experience'],df['Income'])


# In[48]:


# income vs experience is also balance


# In[49]:


df['Income'],


# In[50]:


plt.hist(df['Income'])


# In[51]:


plt.scatter(df['Income'],df['Experience'])


# In[52]:


# normalizing the data to remove its baisedness


# In[89]:


from sklearn import preprocessing


# In[90]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['Income'] = scaler.fit_transform(df['Income'].values.reshape(-1,1))


# In[91]:


df['Mortgage'] = scaler.fit_transform(df['Mortgage'].values.reshape(-1,1))
#Ys = scaler.fit_transform(df['Income'].reshape(-1,1))

#B = vec2matrix(df['Income'],ncol=2)


# In[92]:


df['Income'] = df['Income'].values.reshape(-1, 1)


# In[93]:


df['Income']


# In[103]:


df.groupby('home_ownership').home_ownership.count()


# In[72]:





# In[95]:


df['home_ownership']


# In[ ]:



    
    


# In[ ]:





# In[70]:


# doing one hot encoding


# In[104]:


df1 = pd.get_dummies(df, columns = ['home_ownership'])

print(df1)


# In[106]:


df1


# In[105]:


df.corr() # checking correlation of the data with the target variable


# In[109]:


df['Personal_Loan'].unique()


# In[131]:


#we have to elliminate the empty string


# In[152]:


#df_1=df
df_2 = df1.drop(df1[df1['Personal_Loan'] == ' '].index)


# In[153]:


df_2['Personal_Loan'].unique()


# In[ ]:





# In[154]:


df_2


# In[158]:


x=df_2.drop('Personal_Loan',axis=1)


# In[123]:


#columns = ['Age', 'Experience', 'Income',  'Family', 'CCAvg',
#   'Education', 'Mortgage', 
#   'cd_account', 'Online', 'CreditCard']


# In[159]:


x


# In[160]:


y=df_2['Personal_Loan']
#y=y.drop(1)


# In[162]:


#from sklearn.ensemble import ExtraTreesClassifier
#model=ExtraTreesClassifier()
#model.fit(x,y)


# In[ ]:


#from sklearn.linear_model import LogisticRegression
#lr = LogisticRegression()
#lr.fit(x,df_1)


# In[170]:


y.value_counts()  # this is the skewed data we have to do oversampling to elliminate its effect


# In[164]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[165]:


X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.20,random_state=0)


# In[171]:


np.bincount(Y_train)


# In[307]:


#columns = ['Age', 'Experience', 'Income',  'Family', 'CCAvg',
 #      'Education', 'Mortgage', 
  #     'CD Account', 'Online', 'CreditCard']


# In[148]:


x=df_1[columns]


# In[311]:


#y=df_2['Personal_Loan']


# In[167]:


x_feature=np.array(x)
y_feature=np.array(y,dtype=np.int32)


# In[168]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()


# In[172]:


model.fit(x_feature,y_feature)


# In[173]:


X_test = x.iloc[70:1111]
y_test = y.iloc[70:1111]

X_test = np.array(X_test)
y_test = np.array(y_test)


# In[174]:


y_hat = model.predict(X_test)


# In[188]:


a_test = np.array(y_test)
y_test = np.array(y_hat) 


# In[189]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_hat)
cm


# In[337]:


#from here we can see that our model is trained as the 100 accuracy.


# In[185]:





# In[338]:





# In[ ]:





# In[343]:





# In[137]:


# !pip install imblearn


# In[136]:


#from imblearn.over_sampling import RandomOverSampler
#ros=RandomOverSampler(random_state=0)
#x_resampled,y_resampled=ros.fit_resample(x_feature,y_feature)


# In[134]:


#pip install imblearn


# In[135]:


#!pip install imblearn


# In[133]:


#from imblearn.over_sampling import RandomOverSampler


# In[132]:


#from imblearn.over_sampling import RandomOverSampler

