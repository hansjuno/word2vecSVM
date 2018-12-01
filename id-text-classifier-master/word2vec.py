"""
Train word2vec model from raw dataset
@Author yohanes.gultom@gmail.com
"""

import gensim, logging
import csv
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# config
RAW_DATASET_FILE = 'dataset_labeled.csv'
STOPWORDS_FILE = 'stopwords-id.txt'
OUT_PATH = 'w2v.pkl'
VECTOR_SIZE = 300
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Dataset(object):
    """
    Yield splitted sentence from TSV dataset
    Dataset content example:

    first sentence  0
    second sentence 1
    """
    def __init__(self, csv_file, stem=True, stopwords=True, verbose=True):
        self.csv_file = csv_file
        self.stemmer = StemmerFactory().create_stemmer() if stem else None
        self.stopwords = []
        if stopwords:
            with open(STOPWORDS_FILE, 'r') as f:
                self.stopwords = f.read().splitlines()
        
    
    def __iter__(self):
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                sentence = []
                if self.stemmer:
                    for token in self.stemmer.stem(row[0]).split():
                        if token not in self.stopwords:
                            sentence.append(token)
                else:
                    for token in row[0].lower().split():
                        if token not in self.stopwords:
                            sentence.append(token)
                if sentence:
                    yield(sentence)


if __name__ == '__main__':
    sentences = Dataset(RAW_DATASET_FILE, stem=False)
    model = gensim.models.Word2Vec(sg = 1, # 1=skipgram, 0=CBOW
    seed = seed, 
    workers = num_workers,
    size = num_features,
    min_count = min_word_count,
    window = context_size,
    iter = iteration)

    # quick check
    assert model.wv['bawang'].shape[0] == VECTOR_SIZE

    # save to file
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    with open(OUT_PATH, 'w') as f:
        pickle.dump(w2v, f)
        print('Word2vec model saved in {}'.format(OUT_PATH))
