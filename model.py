import pandas as pd

dataset=pd.read_csv('train.csv')

dataset.head()

dataset.shape

dataset.tail()

dataset=dataset.drop('id',axis=1)

dataset.head(2)

X=dataset.iloc[:,1]

X
y=dataset['label']

y.head()

X.replace("[^a-zA-Z]"," ",regex=True, inplace=True)


# In[13]:


X.head()


# In[14]:
tweets=list(X)


# In[21]:


tweets[1]


# In[22]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[36]:


countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(tweets)

pickle.dump(countvector,open('transform.pkl','wb'))


# In[24]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()


# In[25]:


model.fit(traindataset,y)


# for testing

# In[26]:



import pickle

from sklearn.externals import joblib

filename = 'tweet_model.pkl'
pickle.dump(model, open(filename, 'wb'))







