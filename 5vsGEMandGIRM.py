#%%
import os
import sys
import time
import stat
import numpy as np
import pandas as pd
import scanpy as sc
import utils as tool
import ComplexNetM as CNet
import CsnConstruct as Csn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn_extra.cluster import KMedoids
from sklearn import metrics
import umap
import umap.plot
import matplotlib.pyplot as plt
workDir = sys.path[0]
os.chdir(workDir)
#10X_PBMC\GSE75748sctime\mouse_bladder_cell\GSE82187\GSE90047\GSE81861\Buettner\Jiaxu2017
#mouse_ES_cell\Trapnell\GSE102066
#三个字典存放指标结果，建议用jupyter看
Raw = {}
Gir = {}
Ndm = {}
#%%
name = ['GSE75748','mouse_bladder_cell','GSE102066','GSE90047','GSE82187','GSE71585','Trapnell','10X_PBMC','GSE81861','GSE75748sctime','mouse_ES_cell']
# name = ['GSE75748','mouse_bladder_cell','GSE102066','GSE90047','GSE82187','GSE71585','Trapnell']
# 10X PBMC	Mouse bladder cells	Li	Chu-time	Wang	Gokce	Klein
# Chu-type	Mouse bladder cells	Wang	Yang	Gokce	Tasic	Trapnell
#%%
from tqdm import tqdm
def clusterKD(matrix, label, cls = 'KMeans', n_clusters = 4, n_components = 8, decompose = 'PCA', target = 'ARI',random_state=1):
    switch = {
        'KMeans' : KMeans(n_clusters= n_clusters,random_state=random_state).fit,
        'Hierarchical' : AgglomerativeClustering(n_clusters= n_clusters).fit,
        'KMedoids' : KMedoids(n_clusters= n_clusters).fit,   
        'Spectral' : SpectralClustering(n_clusters= n_clusters).fit,
    }
    if decompose == 'TSNE':
        newx = TSNE(n_components = n_components).fit_transform(matrix)
    elif decompose == 'PCA': 
        newx = PCA(n_components = n_components).fit_transform(matrix)
    else: 
        newx = matrix
    y_pred = switch[cls](newx)
    
    if target == 'ARI':
        score = metrics.adjusted_rand_score(y_pred.labels_,label)
    elif target == 'FM':
        score = metrics.fowlkes_mallows_score(y_pred.labels_,label)
    elif target == 'NMI':
        score = metrics.normalized_mutual_info_score(y_pred.labels_,label)
    return score

for GSEname in tqdm(name):
    GSEdir = './data/' + GSEname +'.h5ad'
    adata = tool.scDataCluster(GSEdir, cls = 'h5', min_cells = 10, highlyVarGene= 2000)
    label = adata.obs['Day']
    dataName = GSEname + '2000'
    csnDataDir = workDir + '/' + dataName + 'csnData/'
    savePath = csnDataDir + 'result'
    mtx = adata.X #(cell,gene)

    # X = np.array(mtx).T
    cls = 'GIR'
    centralityDir = savePath + '/' +cls +'Matrix.csv'
    GirM = np.array(pd.read_csv(centralityDir, header = None))

    cls = 'Degree'
    centralityDir = savePath + '/' +cls +'Matrix.csv'
    DegreeM = np.array(pd.read_csv(centralityDir, header = None))
    n_clusters = len(np.unique(label))
    decompose = 'PCA'
    taget = 'ARI'
    n_com = 8
    Raw[GSEname] = []
    Gir[GSEname] = []
    Ndm[GSEname] = []

    Raw[GSEname].append(clusterKD(mtx, label, cls= 'KMeans', n_clusters = n_clusters, n_components = n_com,decompose = decompose,target =taget))
    Ndm[GSEname].append(clusterKD(DegreeM, label, cls= 'KMeans', n_clusters = n_clusters, n_components = n_com,decompose = decompose,target =taget))
    Gir[GSEname].append( clusterKD(GirM, label, cls= 'KMeans', n_clusters = n_clusters, n_components = n_com,decompose = decompose,target =taget))

    Raw[GSEname].append(clusterKD(mtx, label, cls= 'KMeans', n_clusters = n_clusters, n_components = 2,decompose = 'TSNE',target =taget))
    Ndm[GSEname].append(clusterKD(DegreeM, label, cls= 'KMeans', n_clusters = n_clusters, n_components = 2,decompose = 'TSNE',target =taget))
    Gir[GSEname].append( clusterKD(GirM, label, cls= 'KMeans', n_clusters = n_clusters, n_components = 2,decompose = 'TSNE',target =taget))

    Raw[GSEname].append(clusterKD(mtx, label, cls= 'Hierarchical', n_clusters = n_clusters, n_components = n_com,decompose = decompose,target =taget))
    Ndm[GSEname].append(clusterKD(DegreeM, label, cls= 'Hierarchical', n_clusters = n_clusters, n_components = n_com,decompose = decompose,target =taget))
    Gir[GSEname].append( clusterKD(GirM, label, cls= 'Hierarchical', n_clusters = n_clusters, n_components = n_com,decompose = decompose,target =taget))

    Raw[GSEname].append(clusterKD(mtx, label, cls= 'Hierarchical', n_clusters = n_clusters, n_components = 2,decompose = 'TSNE',target =taget))
    Ndm[GSEname].append(clusterKD(DegreeM, label, cls= 'Hierarchical', n_clusters = n_clusters, n_components = 2,decompose = 'TSNE',target =taget))
    Gir[GSEname].append( clusterKD(GirM, label, cls= 'Hierarchical', n_clusters = n_clusters, n_components = 2,decompose = 'TSNE',target =taget))

    Raw[GSEname].append(clusterKD(mtx, label, cls= 'KMedoids', n_clusters = n_clusters, n_components = n_com,decompose = decompose,target =taget))
    Ndm[GSEname].append(clusterKD(DegreeM, label, cls= 'KMedoids', n_clusters = n_clusters, n_components = n_com,decompose = decompose,target =taget))
    Gir[GSEname].append( clusterKD(GirM, label, cls= 'KMedoids', n_clusters = n_clusters, n_components = n_com,decompose = decompose,target =taget))

    Raw[GSEname].append(clusterKD(mtx, label, cls= 'Spectral', n_clusters = n_clusters, n_components = n_com,decompose = decompose,target =taget))
    Ndm[GSEname].append(clusterKD(DegreeM, label, cls= 'Spectral', n_clusters = n_clusters, n_components = n_com,decompose = decompose,target =taget))
    Gir[GSEname].append( clusterKD(GirM, label, cls= 'Spectral', n_clusters = n_clusters, n_components = n_com,decompose = decompose,target =taget))
    
    
    # sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    # sc.tl.leiden(adata)
    # Raw[GSEname].append(adata.obs['leiden'])
    # adata.X = GirM
    # sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    # sc.tl.leiden(adata)
    # Gir[GSEname].append(adata.obs['leiden'])
#%%

# %%
#seurat要单独跑
for GSEname in tqdm(name):
    GSEdir = '../CSNData/final_data/' + GSEname +'.h5ad'
    adata = tool.scDataCluster(GSEdir, cls = 'h5', min_cells = 10, highlyVarGene= 2000)
    label = adata.obs['Day']
    dataName = GSEname + '2000'
    csnDataDir = workDir + '/' + dataName + 'csnData/'
    savePath = csnDataDir + 'result'
    mtx = adata.X #(cell,gene)

    # X = np.array(mtx).T
    cls = 'GIR'
    centralityDir = savePath + '/' +cls +'Matrix.csv'
    GirM = np.array(pd.read_csv(centralityDir, header = None))

    cls = 'Degree'
    centralityDir = savePath + '/' +cls +'Matrix.csv'
    DegreeM = np.array(pd.read_csv(centralityDir, header = None))
    n_clusters = len(np.unique(label))
    decompose = 'PCA'
    taget = 'ARI'
    n_com = 8
    Raw[GSEname] = []
    Gir[GSEname] = []
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata)
    Raw[GSEname].append(metrics.adjusted_rand_score(adata.obs['leiden'],label))
    adata = tool.scDataCluster(GSEdir, cls = 'h5', min_cells = 10, highlyVarGene= 2000)
    adata.X = GirM
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata)
    Gir[GSEname].append(metrics.adjusted_rand_score(adata.obs['leiden'],label))
# %%

