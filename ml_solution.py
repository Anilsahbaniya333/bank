

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve # for predicting error y_tested and y_predicted

from sklearn.metrics import auc # area under curb finding the accuracy 

#dataset=pd.read_csv("Downloads/New Bank_loan_data.csv")

df=pd.read_excel("Downloads/New Bank_loan_data.xlsx") #reading excell file

nulls=df.isnull().sum()

nulls=nulls[nulls>0]

nulls

per=nulls*100/df.shape[0]

pd.concat([nulls, per], axis=1, 
          keys=['nulls', 'Per']).\
          sort_values(by="nulls", ascending=False)

df['Gender'].fillna('M', inplace=True)   # filling null values with M 

df['home_ownership'].fillna('Rent', inplace=True)# filling missing values with rent due to majority number of rent there

df['Online'].fillna(1, inplace=True)  #filling missing values with 1 due to majority number of Online there 


# In[98]:


df['Income'].fillna(df['Income'].mean(), inplace=True)# filling missing value with mean of the system


# In[81]:


# dropping the unnecessary content
del df['ID']

del df["ZIP Code"]



del df['Gender']


df.head(10)

df.groupby('Online').Online.count()# classifing the o and 1 in the column.

nulls=df.isnull().sum()
nulls


df

df.rename(columns={"Personal Loan": "Personal_Loan","Securities Account":"Sec_Acc","Home Ownership":"home_ownership",'CD Account':'cd_account'},inplace=True)

df  # renaming  above

df.dtypes  # EDA trying to identify the nature of the data

df.describe() #quick summary of data

sns.countplot(data=df,x='Personal_Loan') 

indexAge = df[ (df['Age'] >= 100) ].index
df.drop(indexAge , inplace=True)
indexAge = df[ (df['Experience'] < 0) ].index
df.drop(indexAge , inplace=True)
df.describe()                                          # elliminating the given row which has age beyond 100 year and experience less 
                                                        #less than 0.


# visualising the image and trying to identify their relation


# In[54]:


sns.countplot(data=df,x='Personal_Loan') 

plt.hist(df['Experience'])

# experience is well balance here

df['Experience']

plt.bar(df['Experience'],df['Age'])

# increasing experience with age

plt.bar(df['Experience'],df['Income'])

# income vs experience is also balance

df['Income'],

plt.hist(df['Income'])

plt.scatter(df['Income'],df['Experience'])

# normalizing the data to remove its baisedness

from sklearn import preprocessing


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['Income'] = scaler.fit_transform(df['Income'].values.reshape(-1,1))

df['Mortgage'] = scaler.fit_transform(df['Mortgage'].values.reshape(-1,1))
#Ys = scaler.fit_transform(df['Income'].reshape(-1,1))

#B = vec2matrix(df['Income'],ncol=2)


df['Income'] = df['Income'].values.reshape(-1, 1)

df['Income']


df.groupby('home_ownership').home_ownership.count()

df['home_ownership']

# doing one hot encoding


df1 = pd.get_dummies(df, columns = ['home_ownership'])

print(df1)

df1

df.corr() # checking correlation of the data with the target variable

df['Personal_Loan'].unique()

#we have to elliminate the empty string

#df_1=df
df_2 = df1.drop(df1[df1['Personal_Loan'] == ' '].index)

df_2['Personal_Loan'].unique()


df_2

x=df_2.drop('Personal_Loan',axis=1)

#columns = ['Age', 'Experience', 'Income',  'Family', 'CCAvg',
#   'Education', 'Mortgage', 
#   'cd_account', 'Online', 'CreditCard']

x
y=df_2['Personal_Loan']
#y=y.drop(1)


y.value_counts()  # this is the skewed data we have to do oversampling to elliminate its effect



from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.20,random_state=0)



np.bincount(Y_train)


# In[307]:


x_feature=np.array(x)
y_feature=np.array(y,dtype=np.int32)


# In[168]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_feature,y_feature)

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


#from here we can see that our model is trained as the 100 accuracy.

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

