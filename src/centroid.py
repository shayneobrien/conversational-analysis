# KEYPHRASE EXTRACTION
import re, string
import networkx as nx
import numpy as np
from utils import load_kathy_corpus, flatten, stop_words, Document
from copy import deepcopy

from sklearn.metrics.pairwise import cosine_similarity as cossim
from sklearn.feature_extraction.text import TfidfVectorizer


class Centroid:
    """ Summarize a document using the revisited centroid method.
    Method must be chosen from ['first', 'best', 'tfidf', 'all']
    
    Usage:
        >> corpus = load_kathy_corpus('../data/')
        >> document = corpus[12]
        >> centroid = Centroid(corpus, method='tfidf')
        >> centroid(document, 10) 
        """
    def __init__(self, corpus, method=None, N=250):
        if method == 'first':
            self.method = self._n_first
        elif method == 'best':
            self.method = self._n_best
        elif method == 'tfidf':
            self.method = self._new_tfidf
        elif method == 'all':
            self.method = self._all
        else:
            print('Invalid method ["first", "best", "tfidf", "all"]. \         
            Defaulting to n_first.')
            self.method = self._n_first
        
        self.N = N
        self.corpus = Document(flatten(corpus))
        self.tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stop_words)
        self.matrix = self.tfidf.fit_transform(self.corpus.tokenized)

    def __call__(self, *args, **kwargs):
        return self.summarize(*args, **kwargs)
    
    def summarize(self, document, sum_len=5):
        """ Centroid defined as the sum of all tfidf vectors of the document.
        Summaries are made by greedily adding in sentences such that summed cossim
        between the summary vector and the centroid is maximized at each iteration,
        until sum_len is reached.  
        
        Input:
            document: Document class containing list of Sentence instances
            sum_len: length for the summary to be (int)
        Output:
            list of sentences that 'summarize' the input document.
        """
        self.document = document
        
        self.submatrix = self.matrix[document.i1:document.i2]
        self.centroid = sum(self.submatrix[:])
        self.submatrix = self.method()

        summary, sum_idx = [], []

        while len(summary) < sum_len:
            sims = np.ravel([cossim(self.centroid, sum(summary + [i]))
                             for i in self.submatrix])
            summary.append(self.submatrix[np.argmax(sims)])
            sum_idx.append(np.argmax(sims))
        
        return [self.document.text[i] for i in sum_idx]
    
    def _n_first(self):
        """ Only consider the N first sentences """
        return self.submatrix[:self.N]
    
    def _n_best(self):
        """ Only consider the N most similar sentences to the centroid """
        sims = np.ravel([cossim(self.centroid, i) for i in self.submatrix])
        args = np.argsort(sims)[::-1]
        return self.submatrix[args[:self.N]]
    
    def _new_tfidf(self):
        """ Only consider the N most contributive sentences, where contributive
        is defined as the summed tfidf weights of the vocab that sentence introduces
        """
        vocab, new_idfs = set(), []
        for sent in self.document.tokenized:
            tokens = sent.split()
            new_idfs.append(sum([self._token_to_tfidf(i) 
                               for i in tokens 
                               if i not in vocab]))
            vocab.update(tokens)
        
        args = np.argsort(new_idfs)[::-1]
        return self.submatrix[args[:self.N]]
    
    def _all(self):
        """ Do not prune sentences at all when adding to the summary (best, but
        computationally inefficient) """
        return self.submatrix
    
    def _token_to_tfidf(self, token, threshold=10):
        """ Lookup the tfidf value of a vocabulary term. Return 0 if the term
        is not in the vocabulary, or if it is below the threshold for inclusion """
        try:
            idx = self.tfidf.vocabulary_[token]
        except KeyError:
            return 0
        value = self.tfidf.idf_[idx]
        if value > threshold:
            return value
        return 0
