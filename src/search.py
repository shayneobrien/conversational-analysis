# EFFICIENT SEARCH
import re
from sklearn.feature_extraction.text import CountVectorizer
from utils import load_kathy_corpus, flatten, get_file_list
from scipy.sparse import find
from collections import defaultdict

class Search:
    def __init__(self, corpus):
        """ Efficiently search the corpus for occurrences of a unigram or bigram token.
        Input: token (unigram, bigram)
        Output: dictionary where keys are document filenames and values are untokenized lines with that token in it
        Usage:
            >> search = Search(corpus)
            >> search.find_occurrences('Donald Trump') # case insensitive
        """
        self.corpus = corpus
        self.cv, self.vectors = self._vectorize_corpus(ngram = 2)
        
    def __call__(self, *args):
        return self.find_occurrences(*args)

    def find_occurrences(self, token):
        """ Find all occurrences of a case-insensitive token. 
        Clean token before searching. Return dictionary of line, occurrence """
        token = self._clean_token(token)
        word_idx = self.cv.vocabulary_[token]
        line_idx, _, _ = find(self.vectors[:, word_idx])
        
        flat_corpus = flatten([[line for line in doc.full] for doc in corpus.full])
        lines = [flat_corpus[idx] for idx in line_idx]
        
        dic = defaultdict(list)
        for line in lines:
            dic[line.file].append(line.text)
        return dic
    
    def _vectorize_corpus(self, ngram):
        """ Vectorize the corpus into a sparse matrix based on binary occurrence """
        cv = CountVectorizer(ngram_range=(1, ngram), binary = True)
        vectors = cv.fit_transform(flatten(self.corpus.tokenized))
        return cv, vectors
    
    def _clean_token(self, token):
        """  Remove everything but whitespace, the alphabet. Separate apostrophes for stopwords  """
        token = re.sub(r"[^a-z\s]", '', token.lower())
        token = re.sub(r"[']+", ' ', token)
        return token
