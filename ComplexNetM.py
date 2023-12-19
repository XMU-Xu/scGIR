
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy import sparse

def degreeCentrality(G):
    if len(G) <= 1:
        return np.array([n for n in G])
    s = 1.0/(len(G) - 1.0)
    centrality = np.array([d * s for n, d in G.degree()])
    return centrality

def eigenvectorCentrality(G):
    dict = nx.eigenvector_centrality(G)
    ret = []
    for _, v in dict.items():
        ret.append(v)
    return np.array(ret)

def betweennessCentrality(G):
    dict = nx.betweenness_centrality(G)
    ret = []
    for _, v in dict.items():
        ret.append(v)
    return np.array(ret)

def pageRankCentrality(G):
    dict = nx.pagerank(G)
    ret = []
    for _, v in dict.items():
        ret.append(v)
    return np.array(ret)

def closenessCentrality(G):
    dict = nx.closeness_centrality(G)
    ret = []
    for _, v in dict.items():
        ret.append(v)
    return np.array(ret)



def geneCentralityRank(X, path, savePath, centralityDir, cells, cls = 'PageRank'):
    #有加权的重要性计算
    switch = {
        'Degree' : degreeCentrality,
        'Eigenvector' : eigenvectorCentrality,
        'Betweenness' : betweennessCentrality,
        'PageRank' : pageRankCentrality,
        'Closeness' : closenessCentrality,
    }
    for k in tqdm(range(cells)):
        cellsSP = sparse.load_npz(path + 'csn' + str(k) + '.npz') # 读取
        cellsMat = cellsSP.toarray()
        geneExp = X[:,k]
        M = np.multiply(cellsMat, geneExp)
        sumM = np.sum(M, axis = 1) + 0.0001
        broSumM = np.ones(M.shape)*sumM
        edgeW = M / broSumM.T
        l1,l2 = np.nonzero(edgeW)
        DG = nx.from_numpy_array(cellsMat, create_using=nx.DiGraph)
        for i in range(l1.shape[0]):
            DG.add_weighted_edges_from([(l1[i], l2[i], edgeW[l1[i]][l2[i]])])
        n = switch[cls](DG).reshape(1,-1)
        with open(centralityDir, 'a') as f:
            np.savetxt(f, n, fmt = '%f', delimiter = ',')

def martrixCentrality(path, savePath, centralityDir, cells, cls = 'Degree'):
    #无加权的重要性计算
    switch = {
        'Degree' : degreeCentrality,
        'Eigenvector' : eigenvectorCentrality,
        'Betweenness' : betweennessCentrality,
        'PageRank' : pageRankCentrality,
        'Closeness' : closenessCentrality,
    }
    for k in tqdm(range(cells)):
        cellsSP = sparse.load_npz(path + 'csn' + str(k) + '.npz') # 读取
        cellsMat = cellsSP.toarray()
        DG = nx.from_numpy_array(cellsMat, create_using=nx.DiGraph)
        n = switch[cls](DG).reshape(1,-1)
        with open(centralityDir, 'a') as f:
            np.savetxt(f, n, fmt = '%f', delimiter = ',')