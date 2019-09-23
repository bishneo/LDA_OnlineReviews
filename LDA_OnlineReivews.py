#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import nltk
# from nltk import FreqDist
# nltk.download('stopwords') # run this one time


# In[3]:


import pandas as pd
import numpy as np
import re
import spacy

import gensim
from gensim import corpora

# Libraries for visualization

import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns


# nltk libraries
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

## get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_colwidth", 200)


# In[4]:


# Read the data file and store in a Pandas DataFrame

df = pd.read_json('data/Automotive_5.json', lines=True)
df.head()


# In[5]:


def getFrequencyDistribution(words, num = 20):
    
    fdist = FreqDist(words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    # selecting top 'num' most frequent words
    d = words_df.nlargest(columns="count", n = num) 
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()
    
    return words_df


# In[6]:


def removeStopWords(text):
    stop_words = set(stopwords.words('english'))
    return [w for w in text if not w in stop_words and len(w) > 2]


# In[7]:


def removeNumbersAndSymbols(words):
    return [w.lower() for w in words if w.isalpha()]


# In[8]:


def runLemmatization(text): # filter noun and adjective
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in text]


# In[9]:


def executeMethod(reviews, method):
    return [method(rev) for rev in reviews]


# In[18]:


def getAllWords(reviews):
    return [w for rev in reviews for w in rev]


# In[10]:


tokenized_reviews = [word_tokenize(rev) for rev in df['reviewText']]

print(tokenized_reviews[1])

# freqDist = getFrequencyDistribution(all_words, 30)


# In[11]:


filtered_revs = executeMethod(tokenized_reviews, removeNumbersAndSymbols)
filtered_revs = executeMethod(filtered_revs, removeStopWords)

#freqDist = getFrequencyDistribution(sum(filtered_revs,[]), 30)


# In[12]:


lemmatized_revs = executeMethod(filtered_revs, runLemmatization)
#freqDist = getFrequencyDistribution(lemmatized_words, 30)


# In[21]:


all_words = getAllWords(lemmatized_revs)
freqDist = getFrequencyDistribution(all_words)


# In[26]:


# Create Dictionary
dictionary = corpora.Dictionary(lemmatized_revs)

#convert to document term matrix
doc_term_matrix = [dictionary.doc2bow(rev) for rev in lemmatized_revs]


# In[ ]:


# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=7, random_state=100,
                chunksize=1000, passes=50)


# In[ ]:


lda_model.print_topics()

