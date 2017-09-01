# -*- coding: utf-8 -*-
from __future__ import print_function
import nltk
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
class Analise (object):
    def __init__(self):

        self.dataPath_treino = 'arquivoTreino/database2.csv'
        self.stopwordsnltk = nltk.corpus.stopwords.words('portuguese')
        self.stopwordsnltk.append('vou')
        self.stopwordsnltk.append('tão')
        self.stopwordsnltk.append('vai')

        self.base = self.getBase()
        self.treino = self.aplicastemmer(self.base)
        self.palavrastreinamento = self.buscapalvras(self.treino)
        self.frequenciatreinamento = self.buscafrequencia(self.palavrastreinamento)
        self.palavrasunicastreinamento = self.buscapalavrasunicas(self.frequenciatreinamento)

    def getBase(self):
        dados = []
        print('========= Buscando Dados Treino =============')
        with open(self.dataPath_treino, 'rb') as file:
            reader = csv.reader(file)
            good = 0
            bad = 0
            neutral = 0
            for row in reader:
                if row[3] == 'Good' and good < 500:
                    dados.append((row[0], row[3]))
                    good =good+1
                if row[3] == 'Bad':
                    dados.append((row[0], row[3]))
                    bad = bad+1
                #if row[3] == 'Neutral':
                 #   dados.append((row[0], row[3]))
                  #  neutral = neutral+1
        print('=========== DONE =============================')
        print('Good: %d | Bad: %d '%(good,bad))
        return dados

    def aplicastemmer(self,texto):
        stemmer = nltk.stem.RSLPStemmer()
        frasesstemming =[]
        for (palavras,emocao) in texto:
            comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in self.stopwordsnltk]
            frasesstemming.append((comstemming,emocao))
        return frasesstemming

    def buscapalvras(self,frases):
        todaspalvras=[]
        for (palavras,emocao) in frases:
            todaspalvras.extend(palavras)
        return todaspalvras

    def buscafrequencia(self,palavras):
        palavras = nltk.FreqDist(palavras)
        return palavras

    def buscapalavrasunicas(self,frequencia):
        freq = frequencia.keys()
        return freq

    def extratorpalavrasTreino(self,documento):
        doc = set(documento)
        caracteristicas = {}

        for palavras in self.palavrasunicastreinamento:
            caracteristicas['%s' %palavras] = (palavras in doc)
        return caracteristicas

    def treinar(self):
        basecompletaTreino = nltk.classify.apply_features(self.extratorpalavrasTreino, self.treino)
        classificador = nltk.NaiveBayesClassifier.train(basecompletaTreino)
        print(classificador.labels())
        return classificador

    def classificarTopicos(self,classificador,topicos):
        print("=========== CLASSIFICAR TOPICOS ==============")
        for frases in topicos:
            topicostemming = []
            stemmer = nltk.stem.RSLPStemmer()
            for (palavrastreinamento) in frases.split():
                comstem = [p for p in palavrastreinamento.split()]
                topicostemming.append(str(stemmer.stem(comstem[0])))
            print(frases)

            novoTopico = self.extratorpalavrasTreino(topicostemming)
            # print(novo)
            print(classificador.classify(novoTopico))
            distribuicao = classificador.prob_classify(novoTopico)
            for classe in distribuicao.samples():
                print("%s: %f" % (classe, distribuicao.prob(classe)))
        print("============ DONE CLASSIFICAR TOPICOS ============")

    def classificarClusters(self,classificador,clusters):
        print("=========== CLASSIFICAR CLUSTER ==============")
        clustersClassificacao =[]
        for frases in clusters:
            clusterStem = []
            stemmer = nltk.stem.RSLPStemmer()
            for palavraFrase in frases[0].split():
                comStem = [p for p in palavraFrase.split()]
                clusterStem.append(str(stemmer.stem(comStem[0])))
            #print(frases[0])
            frasesTratada = self.extratorpalavrasTreino(clusterStem)
            # print(novo)
            classificacao = classificador.classify(frasesTratada)
            #print(classificacao)

            clustersClassificacao.append((classificacao,frases[1]))
            distribuicao = classificador.prob_classify(frasesTratada)
           # for classe in distribuicao.samples():
            #    print("%s: %f" % (classe, distribuicao.prob(classe)))
        return clustersClassificacao

        print("============ DONE CLASSIFICAR CLUSTER ============")

    def classificar(self, topicos, clusters):
        print('========== CLASSIFICANDO ==========')
        classificador = self.treinar()
        classificadoTopicos = self.classificarTopicos(classificador,topicos)
        classificarClusters = self.classificarClusters(classificador,clusters)
        clusters =[]
        for i in range(len(topicos)):
            grupo = []
            for classificacao in classificarClusters:
                if(classificacao[1] == i):
                    grupo.append(classificacao)
            clusters.append(grupo)

        for cluster in clusters:
            bad = 0
            good =0
            for frase in cluster:
                totalFrases = len(cluster)
                if(frase[0] == 'Bad'):
                    bad = bad + 1
                if(frase[0] == 'Good'):
                    good = good + 1
            pctGood = (good*100)/totalFrases
            pctBad = (bad*100)/totalFrases
            print('Cluster: Good: %d percent (%d qtd)| Bad: %d percent (%d qtd)'%(pctGood,good,pctBad,bad))
        print("========== DONE CLASSIFICANDO ==========")

