# CORPUS STATS
import nltk
import numpy as np
from nltk import word_tokenize, pos_tag_sents, ne_chunk_sents
from utils import load_kathy_corpus, flatten
from collections import Counter

""" Corpus Statistics:
    - Number of documents 
    - Number of words: total, unique (in document, in corpus)
    - Number of utterances (in document, in corpus)
    - Percentiles for utterance lengths
    - Number of unique speakers
    - Distribution for who is doing how much talking
    - Named Entity Recognition, most frequent (in document, in corpus)
"""



""" Basic, high-level corpus statistics """
def count_docs(corpus):
    """ Counts number of documents in a corpus """
    return len(corpus.full)
    
def count_doc_words(document):
    """ Count number of total and unique words in a document from corpus.full """
    split_sentences = [sentence.text.split() for sentence in document]
    all_words = flatten(split_sentences)
    unique_words = set(all_words)
    return len(all_words), len(unique_words)

def count_corp_words(corpus):
    """ Count number of total and unique words in a corpus """
    flat_corpus = flatten(corpus.full)
    num_words, num_unique = count_doc_words(flat_corpus)
    return num_words, num_unique

def count_doc_lines(document):
    """ Count number of utterances in a document from corpus.full. 
    Equivalent to count_docs if calling count_docs(document) """
    return len(document)

def count_corp_lines(corpus):
    """ Count number of utterances in a corpus """
    return sum([len(doc) for doc in corpus.full])

def utterance_stats(corpus, percentiles = [25,50,75,95,100]):
    """ Compute percentile ranks for uttance lengths """
    flat_corpus = flatten(corpus.full)
    split_sentences = [sentence.text.split() for sentence in flat_corpus]
    lengths = [len(sentence) for sentence in split_sentences]
    return np.percentile(lengths, percentiles)


""" Speaker statistics """
def count_doc_speakers(document, solo = False):
    """ Count number of speakers in a document from corpus.full. 
        If solo == True, return their sorted counts. """
    doc_speakers = [sentence.speaker for sentence in document]
    counts = Counter(doc_speakers)
    if solo:
        sorted_counts = sorted(counts.items(), key = lambda x: x[1], reverse = True)
        return sorted_counts
    
    return counts
    
def count_corp_speakers(corpus):
    """ Count corpus speakers """
    doc_speaker_counts = [count_doc_speakers(doc) for doc in corpus.full]
    corp_speakers = np.sum(doc_speaker_counts)
    sorted_counts = sorted(corp_speakers.items(), key = lambda x: x[1], reverse = True)
    return sorted_counts

def separate_speakers(speaker_counts, main_names):
    primary_count = [name[1] for name in speaker_counts if name[0] in main_names]
    return sum(primary_count)


class NER:
    def __init__(self, corpus):
        """ Get all named entities from a corpus. This is the slowest method by far (~1 min locally)
        Usage:
            >> NER = NER(corpus)
            >> stats = NER()
        """
        self.corpus = corpus
    
    def __call__(self):
        return self.count_NER()
    
    def count_NER(self):
        """ Retrieve named entities for entire corpus. Returns sorted list of tuples (NER, Count)"""
        entities = flatten([self._extract_doc_entities(doc) for doc in corpus.text])
        corp_NER = Counter(entities)
        sorted_corp_NER = sorted(corp_NER.items(), key = lambda x: x[1], reverse = True)
        return sorted_corp_NER

    def _extract_doc_entities(self, document, solo = False):
        """ Extract document entities. If solo is true, then we are just operating on that doc and return counts """
        tokenized_sentences = [word_tokenize(sentence) for sentence in document]
        tagged_sentences = pos_tag_sents(tokenized_sentences)
        entity_trees = ne_chunk_sents(tagged_sentences, binary=True)
        doc_NER = flatten([self._extract_sent_entities(tree) for tree in entity_trees])
        if solo:
            return sorted(Counter(doc_NER).items(), key = lambda x: x[1], reverse = True)

        return doc_NER
    
    def _extract_sent_entities(self, tree):
        """ Extract sentence entities """
        entity_names = []
        if hasattr(tree, 'label') and tree.label:
            if tree.label() == 'NE':
                entity_names.append(' '.join([child[0] for child in tree]))
            else:
                for child in tree:
                    entity_names.extend(self._extract_sent_entities(child))
        return entity_names

