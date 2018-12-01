# CORPUS RELATIONS
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import ward, dendrogram

from utils import load_kathy_corpus, get_file_list, flatten

plt.style.use('ggplot')
STOPWORDS = ['em', 'gonna', 'huh', 'yep',  'goin', 'hi', 'inaudible', 'crosstalk', 'thats', 'um', 'uh', 'laughter', 'i', 'im', 'ppl', 'pause', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

class Cluster:
    def __init__(self, corpus, dirname = None):
        """ Clustering algorithms for determining corpus relations. KMeans, Ward Hierarchical.
        """
        self.corpus = corpus
        self.dirname = dirname
    
    def k_means_cluster(self, num_clusters = 8):
        """ KMeans Cluster at the document level. Separates samples into n groups of equal variance by
        within-cluster sum-of-squares. The square is taken of the tfidf matrix row entries, then summed 
        for distance to each centroid. Clusters are initialized semi-randomly (inital clusters far from each other) """
        
        tfidf_vectorizer = TfidfVectorizer(max_df=0.50, max_features=200000, min_df=0.02, stop_words=STOPWORDS, use_idf=True, ngram_range=(1,2))
        tfidf_matrix = tfidf_vectorizer.fit_transform([' . '.join(doc) for doc in self.corpus.tokenized]) #fit the vectorizer to synopses

        dist = 1 - cosine_similarity(tfidf_matrix)
        km = KMeans(n_clusters=num_clusters, max_iter = 300, init='k-means++')
        km = km.fit(tfidf_matrix)
        terms = tfidf_vectorizer.get_feature_names()

        print("Top terms per cluster:")
        order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

        for cluster in range(num_clusters):
            print("Cluster {0} words: {1}".format(cluster+1, ' | '.join([terms[ind] for ind in order_centroids[cluster, :3]])))

        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1) # PCA is a method of MDS
        positions = mds.fit_transform(dist) 

        xs, ys = positions[:, 0], positions[:, 1]

        labels = km.labels_.tolist()
        if self.dirname:
            titles = get_file_list(dirname)
            titles = [label[-8:-4] for label in labels] # KATHY ONLY: get year
        else:
            titles = ['  ' + str(int(label)) for label in labels]

        df = pd.DataFrame(dict(x=xs, y=ys, label = labels, title = titles))

        #group by cluster
        groups = df.groupby('label')

        # set up plot
        fig, ax = plt.subplots(figsize=(17, 9)) # set size
        ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

        #iterate through groups to layer the plot
        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=8, 
                    label='Cluster ' + str(name),
                    mec='none')

            ax.set_aspect('auto')

            ax.tick_params(                axis= 'x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)

            ax.tick_params(                axis= 'y',         # changes apply to the y-axis
                which='both',      # both major and minor ticks are affected
                left=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelleft=False)

        ax.legend(numpoints=1) 

        
        #add label in x,y position with the label as the film title
        for i in range(len(df)):
            ax.text(df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i]['title'], size=8)  

        plt.show() 

    def ward_cluster(self, dist = None):
        """ Ward hierarchical clustering. See the Incremental Algorithm for variance minimization """
        
        if not dist:
            tfidf_vectorizer = TfidfVectorizer(max_df=0.50, max_features=200000, min_df=0.02, stop_words=STOPWORDS, use_idf=True, ngram_range=(1,2))
            tfidf_matrix = tfidf_vectorizer.fit_transform([' . '.join(doc) for doc in self.corpus.tokenized]) #fit the vectorizer to synopses
            dist = 1 - cosine_similarity(tfidf_matrix)
        
        if self.dirname:
            labels = get_file_list(dirname)
            labels = [label[-8:-4] for label in labels] # KATHY ONLY: get year
        else:
            labels = None

        linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

        fig, ax = plt.subplots(figsize=(8, 15)) # set size
        ax = dendrogram(linkage_matrix, orientation="right", labels=labels, leaf_font_size = 10);

        plt.tick_params(
            axis= 'x',          # changes apply to the x-axis
            which='both',       # both major and minor ticks are affected
            bottom= False,      # ticks along the bottom edge are off
            top= False,         # ticks along the top edge are off
            labelbottom= False,
            length= 10,
            )

        plt.tight_layout() #show plot with tight layout
        plt.show()
