import time
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.wrappers import FastText

class Embeddings():
  
  def __init__(self):
    self.word2vec = None
    self.embed_dim = 0
  
  def read_model(self, tipe):

    start = time.time()

    if tipe == 'own-model1':
      print('Loading {} '.format(str(tipe)))
      embeddings_path = '/home/pras/Embeddings/word_embedding.vec'
      word2vec_model = KeyedVectors.load_word2vec_format(embeddings_path, binary=False, unicode_errors="ignore")
    elif tipe == 'own-model2':
      print('Loading {} '.format(str(tipe)))
      embeddings_path = '/home/pras/Embeddings/model_arif'
      word2vec_model = KeyedVectors.load_word2vec_format(embeddings_path, binary=False, unicode_errors="ignore")
    elif tipe == 'own-model3':
      print('Loading {} '.format(str(tipe)))
      embeddings_path = '/home/adrian/new_cnn/modelapik_cbows.bin'
      word2vec_model = KeyedVectors.load_word2vec_format(embeddings_path, binary=False, unicode_errors="ignore")
    elif tipe == 'bojanowski':
      print('Loading {} model'.format(str(tipe)))
      embeddings_path = '/home/pras/Embeddings/wiki.bin'
      word2vec_model = FastText.load_fasttext_format(embeddings_path)
    else:
      print('Error Embeddings')

    end = time.time()
    print('Loading {} done in {} Seconds'.format(str(tipe), (end-start)))
    print('')

    self.word2vec = word2vec_model
    self.embed_dim = 300

    return word2vec_model
