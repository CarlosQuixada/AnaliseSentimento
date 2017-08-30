# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from Cluster import Cluster
from Analise import Analise
c = Cluster()
a = Analise()
topicos,clusters= c.getCluster()
a.classificar(topicos,clusters)