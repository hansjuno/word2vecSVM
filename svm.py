import numpy as np
import json
import pandas as pd
from pandas.io.json import json_normalize
from statistics import mean
import argparse
import prepare_data #create fold embeddings

from collections import defaultdict
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

from math import ceil

import torchtext.data
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

def cross_validate(fold, data, embeddings, args):
  actual_counts    = defaultdict(int)
  predicted_counts = defaultdict(int)
  match_counts     = defaultdict(int)

  split_width = int(ceil(len(data.examples)/fold))

  for i in range(fold):
    print('FOLD [{}]'.format(i + 1))

    train_examples = data.examples[:] 
    del train_examples[i*split_width:min(len(data.examples), (i+1)*split_width)]
    test_examples = data.examples[i*split_width:min(len(data.examples), (i+1)*split_width)]

    total_len = len(data.examples)
    train_len = len(train_examples)
    test_len  = len(test_examples)

    train_counts = defaultdict(int)
    test_counts  = defaultdict(int)

    for example in train_examples:
      train_counts[example.label] += 1

    for example in test_examples:
      test_counts[example.label] += 1

    output['fold_{}'.format(i+1)]['total_examples'] = total_len
    output['fold_{}'.format(i+1)]['detail_examples'] = []
   
    # Training Label
    high_pos = train_counts['Highly_Positive']
    pos = train_counts['Positive']
    neg = train_counts['Negative']
    high_neg = train_counts['Highly_Negative']
    neu = train_counts['Neutral']
    output['fold_{}'.format(i+1)]['detail_examples'].append({
      'examples': 'train',
      'total': train_len,
      'high_positive': high_pos,
      'positive': pos,
      'neutral': neu,
      'negative': neg,
      'high_negative': high_neg,
    })

    # Testing Label
    high_pos = test_counts['Highly_Positive']
    pos = test_counts['Positive']
    neg = test_counts['Negative']
    high_neg = test_counts['Highly_Negative']
    neu = test_counts['Neutral']

    output['fold_{}'.format(i+1)]['detail_examples'].append({
      'examples': 'test',
      'total': test_len,
      'high_positive': high_pos,
      'positive': pos,
      'neutral': neu,
      'negative': neg,
      'high_negative': high_neg,
    })
    
    fields = data.fields
    train_set = torchtext.data.Dataset(examples=train_examples, fields=fields)
    test_set  = torchtext.data.Dataset(examples=test_examples, fields=fields)

    # Building the vocabs
    text_field  = None
    label_field = None

    for field_name, field_object in fields:
      if field_name == 'text':
        text_field = field_object
      elif field_name == 'label':
        label_field = field_object
    
    text_field.build_vocab(train_set)
    label_field.build_vocab(train_set)

    data.vocab_to_idx = dict(text_field.vocab.stoi)
    data.idx_to_vocab = {v: k for k, v in data.vocab_to_idx.items()}

    data.label_to_idx = dict(label_field.vocab.stoi)
    data.idx_to_label = {v: k for k, v in data.label_to_idx.items()}

    embed_num = len(text_field.vocab)
    label_num = len(label_field.vocab)

    # Loading pre-trained embeddings
    emb_init_values = np.array(data.create_folds_embeddings(embeddings, args))

    # Ini gunanya untuk mengukur training performance dari model
    # (lawannya generalization performance, forget what it's called)
    train_bulk_dataset = train_set,
    train_bulk__size   = len(train_set),
    train_bulk_iter    = torchtext.data.Iterator.splits(datasets=train_bulk_dataset, 
                                                        batch_sizes=train_bulk__size,
                                                        device=-1, 
                                                        repeat=False)[0]


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
       # wv = KeyedVectors.load_word2vec_format(embeddings_path, binary=False, unicode_error='ignore')
        embeddings_dim = 300
            
    # Start Train
    nested_dict = lambda : defaultdict(nested_dict)
    output = nested_dict()

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

