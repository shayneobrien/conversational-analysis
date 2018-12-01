import os, re, operator
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils import load_kathy_corpus, concat_corpus_docs, flatten, parse_speakers

from collections import defaultdict
from scipy.sparse import find
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords = ['em', 'gonna', 'huh', 'well', 'im', 'ill', 'ive', 'thats', 'hes', 'yeah', 'yep', 'yup', 'youre', 'ok', 'goin', 'hi', 'hey', 'inaudible', 'crosstalk', 'laughter', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'good']

class KeyNet:
    def __init__(self):
        """ Compute keywords and keyword co-occurrence network
        TODO: Needs method for separating speakers. In this case it is critical to output quality!
        
        Usage: # for example, this would give the keywords network for the locals.
            >> keys = KeyNet()
            >> keys.build_keywords(corpus, main_speakers = ['K', 'k', 'Kathy', 'kathy', 'Cathy', 'cathy'], use_main = False, context_window = 10, force = 1.5, display = True)

        """
        pass
        
    def __call__(self, *args):
        return self.build_keywords(*args)
        
    def build_keywords(self, corpus, speakers = None, use_main = True, context_window = 10, force = 1.5, display = True):
        """ display keyword network. force is for forced directed layout, 
        display is display centrality values or not
        speakers is if we want to parse the corpus into just locals or just main speakers
        use_main == True indicates build keyword network for main_speakers. if false, build it for everyone else"""
        if main_speakers:
            main, others = parse_speakers(corpus, speakers)
            if use_main:
                corpus = main
            else:
                corpus = others
                
        corpus = corpus.tokenized
        keyword_scores, keywords = self._display_scores(topX = 50, corpus = corpus, display = display)
        
        # initialize graph
        network_graph = nx.Graph()
        network_graph.clear()

        # add all the nodes
        network_graph.add_nodes_from(keywords)

        # get keyword edge values
        combo_counts = self._get_cooccurrences_context(keywords, corpus, context_window)

        # parse edge values
        edges = []
        normalize_constant = sum(combo_counts.values())
        for keywords_combo in combo_counts.keys():
            individual_keywords = keywords_combo.split('_')
            weight = {'weight': 400 * float(combo_counts[keywords_combo]) / normalize_constant}
            individual_keywords.append(weight)
            edges.append(tuple(individual_keywords))

        # add edges    
        network_graph.add_edges_from(edges)

        # get sorted eigenvector degree centralities, which is the influence of the node in the network. Note polysemy of body
        try: degree_centralities, s = nx.eigenvector_centrality(network_graph, weight = 'weight'), '\nEigenvector Centrality'
        except: degree_centralities, s = nx.degree_centrality(network_graph), 'Degree Centrality'
        sorted_centralities = sorted(degree_centralities.items(), key=operator.itemgetter(1))[::-1]
        if display:
            print(s)
            for i, item in enumerate(sorted_centralities[:50]):
                print("{0}: {1:50} Score: {2}".format(i+1, item[0], item[1]))

        # draw the graph
        plt.clf
        plt.figure(figsize=(15,10))
        spring_pos = nx.spring_layout(network_graph, k=force, iterations=100)
        nx.draw(network_graph, 
                pos=spring_pos,
                nodelist=degree_centralities.keys(), 
                node_size=[(v**2) * 5000 for v in degree_centralities.values()], # eigenvector centrality for node size
                width = [w[2]['weight'] / 4.5 for w in network_graph.edges(data = True)],
                with_labels=True, 
                node_color='#80B3FF')
        plt.show()
        return sorted_centralities[::-1], network_graph  
    
    def _get_tfidf_keywords(self, corpus):
        """ get TFIDF scores from corpus """
        # initialize vectorizer, train it  
        vectorizer = TfidfVectorizer(ngram_range=(1,1), stop_words=stopwords, norm='l2', min_df = 0.10, max_df = 0.80, use_idf=True, smooth_idf=False, sublinear_tf=True)
        try: tfidf_result = vectorizer.fit_transform(corpus)
        except AttributeError: tfidf_result = vectorizer.fit_transform(concat_corpus_docs(corpus))

        # get the scores
        scores = zip(vectorizer.get_feature_names(), tfidf_result.toarray()[0].tolist())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return sorted_scores
    
    def _get_NER_keywords(self, corpus, min_count = 5, topX = 75):
        """ get topX most common Named Entities above a specified count. """
        keywords = NER_stats(corpus)
        keywords = [word[0].lower() for word in keywords[:topX] if (word[1] >= min_count and word[0].lower() not in stopwords)]
        return list(set(keywords))

    def _shift_corpus(self, corpus, context_window):
        """ add padding between documents """
        return [doc + (['null'] * context_window) for doc in corpus]

    def _display_scores(self, topX, corpus, display = True):
        """ display the top X most important words from a vectorized corpus """
        sorted_scores = self._get_tfidf_keywords(corpus)
        if display:
            print('TFIDF Keywords')
            for i, item in enumerate(sorted_scores[:topX]):
                print("{0}: {1:20} Score: {2}".format(i+1, item[0], item[1]))

        keywords = [str(keyword[0]) for keyword in sorted_scores]
        return sorted_scores[:topX], keywords[:topX]

    def _get_cooccurrences_context(self, keywords, corpus, context_window):
        """ given some context window, get co-occurrences for keyword network """
        cv = CountVectorizer(vocabulary = np.sort(keywords), ngram_range=(1,2))
        matrix = cv.fit_transform(flatten(self._shift_corpus(corpus, context_window)))
        vocab = cv.vocabulary

        occurrence_dict, values_dict = defaultdict(list), defaultdict(list)
        row_entry, keyword_id, value = find(matrix)
        occurrences = zip(keyword_id, row_entry, value) # keyword_id, row entry, value
        for entry in occurrences: occurrence_dict[vocab[entry[0]]].append((entry[1])), values_dict[vocab[entry[0]]].append((entry[2]))

        combos = defaultdict()
        for idx, key1 in enumerate(vocab):
            for key2 in vocab[idx+1:]:
                key1_matrix, key2_matrix = np.array(occurrence_dict[key1]), np.array(occurrence_dict[key2]) # get sentence ids where keys occur
                dist = np.abs(key1_matrix[:, np.newaxis] - key2_matrix) # compute exhaustive distances (in sentence count) between mentions
                args = np.argwhere( dist < context_window ) # find where distance less than context_window size
                if len(args) > 0:
                    cooccur_vals = [val1*val2 for val1,val2 in zip([values_dict[key1][x] for x in args[:,0]], [values_dict[key2][y] for y in args[:,1]])] # retrieve their co-occurrence values
                    combos['_'.join([key1, key2])] = combos.get('_'.join([key1, key2]), 0) + sum(cooccur_vals) # update occurrences

        return combos
