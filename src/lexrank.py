# KEYPHRASE EXTRACTION
import re, string
import networkx as nx
import numpy as np
from utils import load_kathy_corpus, flatten, parse_doc_speakers
from math import log10

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import binarize

class LexRank:
    def __init__(self, method):
        """ Extract keyphrase from an input document using LexRank (pagerank on cosine similarity graph)
        Cosine similarity method must be chosen from ['naive', 'idf-mod', 'tfidf']
        
        Usage:
            >> corpus = load_kathy_corpus('../data/')
            >> keyphrases = LexRank('idf-mod')
            >> keyphrases(speakers = ['k', 'K', 'kathy', 'Kathy', 'cathy', 'Cathy'], use_main = False)
        """
        if method == 'naive':
            self.method = self._naive_sim
        elif method == 'idf-mod':
            self.method = self._idf_mod_sim
        elif method == 'tfidf':
            self.method = self._tfidf_sim
        else:
            print("Invalid method chosen from ['naive', 'idf-mod', 'tfidf']. Defaulting to naive.")
            self.method = self._naive_sim
        
    def __call__(self, *args, **kwargs):
        """ Return keyphrases """
        return self.get_keyphrases(*args, **kwargs)
    
    def init_counts(self, pattern=r"(?u)\b\w+\b"):
        """ Initialize matrix optimizations. Pattern is whether to consider 1 character tokens or not """
        self.cv, self.tfidf = CountVectorizer(token_pattern=pattern), TfidfTransformer()
        self.counts = self.cv.fit_transform(self.document.tokenized)
        self.b_counts, self.matrix = binarize(self.counts), self.tfidf.fit_transform(self.counts)
        
    def get_keyphrases(self, document, speakers = None, use_main = False, topX = 5, maxlen = 50, include_scores = False):
        """ Get keyphrases from a document using LexRank
        
        Speakers, use_main use case similar to in keynet.py """
        self.document = document
        
        if speakers:
            main, others = parse_doc_speakers(self.document, speakers)
            if use_main:
                self.document = main
            else:
                self.document = others
                
        self.init_counts()
        
        network_graph = self._build_graph()

        ranked = nx.pagerank_scipy(network_graph)
        ranked = [(val, text) for val, text in ranked.items()]
                
        sort_ranked = sorted(ranked, key=lambda t: t[1], reverse=True)

        if maxlen:
            sort_ranked = [t for t in sort_ranked if len(t[0].split()) < maxlen]

        if not include_scores:
            sort_ranked = [s[0] for s in sort_ranked]
            
        return sort_ranked[:topX]
        
    def _build_graph(self):
        """ Build graph """
        network_graph = nx.Graph()
        network_graph.clear()

        # add all the nodes
        network_graph.add_nodes_from(self.document.text)
        
        # build edges using the similarities data
        edges = self.method()

        # add in edges
        network_graph.add_edges_from(edges)

        return network_graph
    
    def _idf_mod_sim(self, eps=1e-6):
        """ Fast idf-modified-cosine. Gives same results as naive??? """
        tf_xidf = np.dot(self.counts, self.tfidf._idf_diag) # term frequencies * their inverse doc freqs (not tf-idf)

        numerator = np.dot(tf_xidf, tf_xidf.T) # tf_{w,x} * tf_{w,y} * (idf_{w}**2), for the upper triangular

        denominator = np.sqrt(np.dot(self.counts, tf_xidf.T) ** 2).diagonal() + eps # np.sqrt((tf_{w}*idf_{w})^2)

        indices = np.triu_indices_from(numerator, k=1)
        sims = np.asarray(numerator[indices]).flatten()
        
        edges = [[self.document.text[i1], # text sentence 1
                  self.document.text[i2], # text sentence 2
                  {'similarity': sim / (denominator[i1]*denominator[i2])}] # similarity
                for i1, i2, sim in zip(*indices, sims) if sim > 0]

        return edges
    
    def _tfidf_sim(self):
        tf_idf = np.dot(self.matrix, self.matrix.T)
        indices = np.triu_indices_from(tf_idf, k=1)
        sims = np.asarray(tf_idf[indices]).flatten()   
        
        edges = [[self.document.text[i1], # text sentence 1
                  self.document.text[i2], # text sentence 2
                  {'similarity': sim}] # similarity
                for i1, i2, sim in zip(*indices, sims) if sim > 0]
        
        return edges
    
    def _naive_sim(self):
        """ Cosine similarity defined as the length of the set of words in both sentence i and sentence j """
        cooccurs = np.dot(self.b_counts, self.b_counts.T)
        indices = np.triu_indices_from(cooccurs, k=1)
        sims = np.asarray(cooccurs[indices]).flatten()
        
        edges = [[self.document.text[i1], # text sentence 1
                  self.document.text[i2], # text sentence 2
                  {'similarity': self._safe_naive_edge(sim)}] # similarity
                for i1, i2, sim in zip(*indices, sims) if sim > 0]
        
        return edges
    
    def _safe_naive_edge(self, sim):
        """ Safely compute the similarity score for naive, 
            _naive_sim / log length of each sentence added together """
        if sim != 1:
            return sim / log10(sim)
        return 0

    def _prune_nodes(self, network_graph):
        """ Remove all nonzero nodes """
        nodes_to_remove = list(nx.isolates(network_graph))
        network_graph.remove_nodes_from(nodes_to_remove)
        return network_graph
