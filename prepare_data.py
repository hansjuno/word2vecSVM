from gensim.models.keyedvectors import KeyedVectors
import time
import numpy as np
import random
import re
import torchtext
import read_sentences
from read_embeddings import Embeddings

class DataPreparation:

  def __init__(self):
    self.emb_init_values = None
    self.vocab_to_idx    = None
    self.idx_to_vocab    = None
    self.label_to_idx    = None
    self.idx_to_label    = None
    self.embed_num       = 0
    self.label_num       = 0
    self.train_set       = None
    self.dev_set         = None
    
  def polarity_to_label_substask_a(self, polarity):
    # Positive, Negative
    if int(polarity) < 0 :
      return 'Negative'
    elif int(polarity) > 0 :
      return 'Positive'
      
  def polarity_to_label_substask_b(self, polarity):
    # Positive, Netral, Positive
    if int(polarity) < 0 :
      return 'Negative'
    elif int(polarity) > 0 :
      return 'Positive'
    elif int(polarity) == 0 :
      return 'Neutral'
    
  def polarity_to_label_substask_c(self, polarity):
    # Highly Negative, Negative, Neutral, Positive, Highly Positive
    if int(polarity) == -2 :
      return 'Highly_Negative'
    elif int(polarity) == -1 :
      return 'Negative'
    elif int(polarity) == 0 :
      return 'Neutral'
    elif int(polarity) == 1 :
      return 'Positive'
    elif int(polarity) == 2 :
      return 'Highly_Positive'
    
  def read_dataset(self, substask, type_file):

    # Read in data 
    file_name = 'sentimen_own_full_repaired.json'

    # politic_data is an iterator object returning the id, label, and text of data
    politic_data = read_sentences.ReadingIndonesianPolitic(file_name, type_file)

    # Create instructions for processing the text 
    id_field = torchtext.data.Field(use_vocab=False, sequential=False)
    label_field = torchtext.data.Field(sequential=False)
    text_field = torchtext.data.Field()

    fields = [('twt_id',id_field),
              ('label',label_field),
              ('text', text_field)]

    self.fields = fields

    # Populate the list of training and test examples
    if substask == 'subtaskA':
      examples = [torchtext.data.Example.fromlist([twt_id, self.polarity_to_label_substask_a(polarity), text], fields)
                for twt_id, polarity, text in politic_data]
    elif substask == 'subtaskB':
      examples = [torchtext.data.Example.fromlist([twt_id, self.polarity_to_label_substask_b(polarity), text], fields)
                for twt_id, polarity, text in politic_data]
    elif substask == 'subtaskC':
      examples = [torchtext.data.Example.fromlist([twt_id, self.polarity_to_label_substask_c(polarity), text], fields)
                for twt_id, polarity, text in politic_data]
      
    self.examples = examples
    
  def create_folds_embeddings(self, embeddings, args):
    emb_init_values = []
    exist_word_vec = 0
    unexist_word_vec = 0
    
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

    self.exist_word_vec = exist_word_vec
    self.unexist_word_vec = unexist_word_vec
    self.emb_init_values = emb_init_values

    return emb_init_values
          