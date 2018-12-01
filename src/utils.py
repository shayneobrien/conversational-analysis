import re
from os import listdir
from os.path import isfile, join

stop_words = ['em', 'gonna', 'huh', 'yep',  'goin', 'hi', 'inaudible', 'crosstalk', 'laughter', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

class Corpus:
    """ Object for raw and tokenized versions of corpus. 
    Expects list of documents, where each document is a list of separated sentences. """
    def __init__(self, corpus):
        self.full = corpus
        self.text = [document.text for document in corpus]  
        self.tokenized = [document.tokenized for document in corpus] 
        
    def __getitem__(self, idx):
        return self.full[idx]
    
    def __repr__(self):
        return 'Corpus of %d transcripts' % len(self.full)


class Document:
    def __init__(self, document, file=None, i1=None, i2=None):
        """ Object for documents.  
        """
        self.full = document
        self.text = [line.text for line in document]
        self.tokenized = [line.tokenized for line in document]
        self.file = file
        self.i1, self.i2 = i1, i2
        
    def __getitem__(self, idx):
        return self.full[idx]
    
    def __repr__(self):
        return 'Document of %d utterances' % len(self.text)


class Sentence:
    """ Object to hold text, tokenized text, speaker, score (if applicable), 
    and source filename of sentence.
    """
    def __init__(self, text, tokenized=None, speaker = None, file = None, score = -1):
        self.text = text.strip()
        self.tokenized = tokenized.strip()
        self.speaker = speaker
        self.file = file
        self.score = score
        
    def __getitem__(self):
        return self

    def __repr__(self):
        if self.speaker:
            return self.speaker + ': ' + self.text
        return self.text
    
    def __len__(self):
        return len(self.text.split())

    
def load_kathy_corpus(dirname):
    """ Load in corpus from a directory name. 
    Expects a dialogue format where each line is an utterance, and each line is denoted by a speaker, e.g.
        (Kathy: ...)
        (Janet: ...)
        (John: ...)
    """
    corpus, idx_counter = [], 0
    file_list = get_file_list(dirname)

    # For every file in the corpus
    for file in file_list:
        
        # Record the starting index of the document
        start = idx_counter

        # Open the file
        with open(file, 'r') as fh:
            dataframe = fh.read().splitlines()
            
        # Record the ending index of the document
        idx_counter += len(dataframe)
        end = idx_counter

        # Parse all speakers
        split_dataframe = [line.split(':', maxsplit = 1) for line in dataframe]

        # align speakers with their words if they are not already aligned
        aligned_dataframe = [entry if len(entry) == 2 else ['UNK'] + entry for entry in split_dataframe]

        # tidy up the dataframe
        tidy_dataframe = [Sentence(speaker=line[0].strip(), 
                                   text=line[1], 
                                   tokenized=clean_string(line[1]), 
                                   file=file) 
                          for line in aligned_dataframe]

        # add it to the corpus
        corpus.append(Document(document=tidy_dataframe, file=file, i1=start, i2=end))
    
    return Corpus(corpus)


def get_file_list(dirname):
    """ Retrieve all file names in a directory """
    return [dirname + f for f in listdir(dirname) if isfile(join(dirname,f)) if f != '.DS_Store']

def clean_string(string):
    """  Remove everything but whitespace, the alphabet. Separate apostrophes for stopwords  """
    string = re.sub(r"[^a-z\s']", '', string.lower())
    string = re.sub(r"[']+", ' ', string)
    return string

def flatten(list):
    """ Flatten a list of lists into one list """
    return [item for sublist in list for item in sublist]

def remove_stopwords(list):
    """ Remove all stopwords from a list (this input would be like ['hello', 'my', 'name', 'is', '...']) """
    return [word for word in list if word not in stop_words]

def concat_corpus_docs(corpus):
    """ Concatenate all documents in a corpus into one sentence """
    return [concat_doc_sentences(doc) for doc in corpus]

def concat_doc_sentences(document):
    """ Concatenate all sentences in a document into one sentence """
    return ' '.join([word for word in document])

def parse_speakers(corpus, speakers):
    """ Parse a corpus given speakers into two corpora: one containing just those speakers, and another containing all other speakers 
    Usage:
        >> kathy, locals = parse_speakers(corpus, ['K', 'k', 'Cathy', 'cathy', 'Kathy', 'kathy])
    """
    main_speakers = [Document([sentence for sentence in document.full if sentence.speaker in speakers], document.file) for document in corpus.full]
    other_speakers = [Document([sentence for sentence in document.full if sentence.speaker not in speakers], document.file) for document in corpus.full]
    return Corpus(main_speakers), Corpus(other_speakers)

def parse_doc_speakers(document, speakers):
    """ Messy wrap around to parse document speakers """
    main, others = parse_speakers(Corpus([document]), speakers)
    return main.full[0], others.full[0]

def separate_by_year(file_list):
    """ Kathy corpus only: Separate file list into years and return a dictionary with keys being years and values being files for that year """
    dictionary = {}
    years = set([f[-8:-4] for f in file_list])
    for year in years:
        files = [filename for filename in file_list if year in filename]
        dictionary[year] = files
    return dictionary