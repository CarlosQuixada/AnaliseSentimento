from __future__ import print_function

from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from time import time

dataPath = 'C:/Users/Carlos/PycharmProjects/Analise/arquivoTreino/store_reviews.csv'

class Cluster (object):
    def __init__(self):
        self.n_cluster = 8

    def getBase(self):
        Dados = []
        print('========= Buscando Dados Treino =============')
        with open(dataPath, 'rb') as file:
            reader = csv.reader(file)
            for row in reader:
                Dados.append(row[0])
        print('=========== DONE =============================')
        return Dados

    def prepararCluster(self):
        dataset = self.getBase()

        t0 = time()
        pt_stop_words = set(stopwords.words('portuguese'))
        pt_stop_words.add('pra')
        pt_stop_words.add('para')
        vectorizer = TfidfVectorizer(max_df=0.75, max_features=5000,min_df=2, stop_words=pt_stop_words)
        baseVetorizada = vectorizer.fit_transform(dataset)
        #print("done in %fs" % (time() - t0))
        print("n_samples: %d, n_features: %d" % baseVetorizada.shape)
        #print()

        km = KMeans(n_clusters= self.n_cluster, init='k-means++', max_iter=100, n_init=1)

        t0 = time()
        km.fit(baseVetorizada)

        data = km.predict(baseVetorizada)
        clusters = []
        for i in range(len(data)):
            clusters.append((dataset[i],data[i]))

        #print("done in %0.3fs" % (time() - t0))
        #print()
        #print("Top terms per cluster:")


        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()

        return order_centroids,terms,clusters

    def getCluster(self):
        orderCentroids,terms,clusters = self.prepararCluster()

        topicos = []
        for i in range(self.n_cluster):
            texto =''
            for ind in orderCentroids[i, :10]:
                texto = texto + ' ' + terms[ind]
                #texto.append(terms[ind])
            topicos.append(texto)
        return topicos,clusters