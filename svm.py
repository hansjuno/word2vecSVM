
import numpy as np
import json
import pandas as pd
from pandas.io.json import json_normalize
from statistics import mean
import argparse

from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

import sys
import os
import time

from preprocessing import cleansing
from preprocessing import plot_confusion_matrix
from mae import count_mae


# In[11]:

embeddings_path = 'Word2Vec_trained_1.bin'
# Read the data
with open('sentimen_own_full_repaired.json','r',encoding='utf-8') as json_data:
    tweet = json.load(json_data)

# Convert to dataframe
df = pd.io.json.json_normalize(data=tweet, record_path=['RECORDS'])
# Convert to matrix
data = df.as_matrix()

models = {
'tf-idf': 'tf-idf',
}

parser = argparse.ArgumentParser(description='Train and evaluate Indonesian text classifier')
parser.add_argument('-m','--model', default=models['tf-idf'], choices=models.values(), help='Pipeline model')
args = parser.parse_args()
# In[12]:


# Initiation data train and target
X = np.array(data[:,4])
y = np.array(data[:,1])
kf = KFold(n_splits=10)


# In[13]:


final_test_labels = []
final_prediction = []

precision_val = []
recall_val = []
fscore_val = []


# In[14]:

def create_folds_embeddings(self, embeddings, args):
    emb_init_values = []
    exist_word_vec = 0
    unexist_word_vec = 0
    embeddings.embed_dim = 300
    
    args.embeddings_dim = embeddings.embed_dim
    for i in range(self.idx_to_vocab.__len__()): #untuk memastikan urut
      word = self.idx_to_vocab.get(i)
      if word == '<unk>':
        emb_init_values.append(np.random.uniform(-0.25, 0.25, args.embeddings_dim).astype('float32'))
      elif word == '<pad>':
        emb_init_values.append(np.zeros(args.embeddings_dim).astype('float32'))
      elif word in embeddings.word2vec.wv.vocab:
        emb_init_values.append(embeddings.word2vec.wv.word_vec(word))
        exist_word_vec +=1
      else:
        emb_init_values.append(np.random.uniform(-0.25, 0.25, args.embeddings_dim).astype('float32'))
        unexist_word_vec += 1
    
    print('--- WORD VECTOR STATISTIC ---')
    print('Exist Word in vector   : {} word'.format(exist_word_vec))
    print('Unexist Word in vector : {} word'.format(unexist_word_vec))
    print('')

    return emb_init_values   
    
for train_index, test_index in kf.split(X):
    
    X_trainSet, X_testSet = X[train_index], X[test_index]
    y_trainSet, y_testSet = y[train_index], y[test_index]
    
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    
    # Change label train
    for i in range(len(X_trainSet)):  
        train_data.append(cleansing(X_trainSet[i]))
        if int(y_trainSet[i]) == 2:
            train_labels.append(1)
        elif int(y_trainSet[i]) == 1:
            train_labels.append(1)
        elif int(y_trainSet[i]) == -2:
            train_labels.append(-1)
        elif int(y_trainSet[i]) == -1:
            train_labels.append(-1)
        elif int(y_trainSet[i]) == 0:
            train_labels.append(0)
            
    # Change label test    
    for i in range(len(X_testSet)):
        test_data.append(cleansing(X_testSet[i]))
        if int(y_testSet[i]) == 2:
            test_labels.append(1)
        elif int(y_testSet[i]) == 1:
            test_labels.append(1)
        elif int(y_testSet[i]) == -2:
            test_labels.append(-1)
        elif int(y_testSet[i]) == -1:
            test_labels.append(-1)
        elif int(y_testSet[i]) == 0:
            test_labels.append(0)
    
    # Convert data to vector tf-idf

        
    if args.model ==models['tf-idf']:
        vectorizer = TfidfVectorizer(min_df=5,
                                max_df = 0.8,
                                sublinear_tf=True,
                                use_idf=True)
        train_vectors = vectorizer.fit_transform(train_data)
        test_vectors = vectorizer.transform(test_data)

    elif args.model ==models['w2v']:
        embedding_index = {}
        wv = KeyedVectors.load_word2vec_format(embeddings_path, binary=False, unicode_error='ignore')
        embeddings_dim = 300
            
    # Start Train
    classifier = svm.SVC(kernel='linear').fit(train_vectors, train_labels)
    prediction = classifier.predict(test_vectors)
    
    precision, recall, fscore, Null_Value = precision_recall_fscore_support(test_labels, prediction, average='macro')
    print(precision_recall_fscore_support(test_labels, prediction, average='macro'))
    
    precision_val.append(precision)
    recall_val.append(recall)
    fscore_val.append(fscore)
    
    final_test_labels.extend(test_labels)
    final_prediction.extend(prediction)


# In[15]:


print("recall ", mean(recall_val))
print("presisi ", mean(precision_val))
print("akurasi ", accuracy_score(final_test_labels, final_prediction))
print("mae", count_mae(final_test_labels, final_prediction))


# In[10]:


class_names = ['positive', 'neutral', 'negative']
cm = confusion_matrix(final_test_labels, final_prediction)
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix, Naive Bayes')

