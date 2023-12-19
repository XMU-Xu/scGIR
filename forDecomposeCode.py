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
# name = ['10X_PBMC','GSE81861','GSE75748sctime','mouse_ES_cell','GSE75748','GSE102066','GSE90047','GSE82187','Trapnell']
name = ['10X_PBMC','GSE75748sctime','GSE75748','GSE102066']

def clusterKD(matrix, label, cls = 'KMeans', n_clusters = 4, n_components = 8, decompose = 'PCA', target = 'ARI'):
    switch = {
        'KMeans' : KMeans(n_clusters= n_clusters).fit,
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
    print(y_pred.labels_)
    if target == 'ARI':
        score = metrics.adjusted_rand_score(y_pred.labels_,label)
    elif target == 'FM':
        score = metrics.fowlkes_mallows_score(y_pred.labels_,label)
    elif target == 'NMI':
        score = metrics.normalized_mutual_info_score(y_pred.labels_,label)
    return score
# score_PCA = {}
# taget = 'ARI'
# n_com = 8
# n_clusters = len(np.unique(label))
# # decompose = 'PCA'
# # score_PCA['Raw'] = clusterKD(mtx, label, cls= 'KMeans', n_clusters = n_clusters, n_components = n_com,decompose = decompose,target =taget)
# # score_PCA['scGIR'] = clusterKD(GirM, label, cls= 'KMeans', n_clusters = n_clusters, n_components = n_com,decompose = decompose,target =taget)
# # score_PCA['NDM'] = clusterKD(DegreeM, label, cls= 'KMeans', n_clusters = n_clusters, n_components = n_com,decompose = decompose,target =taget)
# # score_PCA
#%%

def drawUmap(
    mtx,
    name,
    ylabel,
    cls = 'KMeans',
    embedding_way = 'UMAP',
    savePath = None,
    size = [5, 5],
    n_clusters = 4,
    component = 8,
    n_neighbors = 6,
    min_dist = 0.1,
    u_components = 2,
    c = 'cet_rainbow4',
    random_state = 99,
):
    switch = {
        'KMeans' : KMeans(n_clusters= n_clusters).fit,
        'Hierarchical' : AgglomerativeClustering(n_clusters= n_clusters).fit,
        'KMedoids' : KMedoids(n_clusters= n_clusters).fit,   
        'Spectral' : SpectralClustering(n_clusters= n_clusters).fit,
        
    }
    fig = plt.figure(figsize=(size[0],size[1]))
    pca = PCA(n_components= component)
    dmtx = pca.fit_transform(mtx)
    # y_pred = switch[cls](dmtx)
    if embedding_way == 'UMAP':
        embedding = umap.UMAP(n_components= u_components).fit_transform(mtx)
    elif embedding_way == 'TSNE':
        
        embedding = TSNE(n_components = u_components).fit_transform(dmtx)
    else:
        embedding = PCA(n_components = u_components).fit_transform(dmtx)

    # unique_labels = list(range(len(np.unique(ylabel))))
    unique_labels = np.unique(ylabel)
    # colors = plt.get_cmap(c)(np.linspace(0, 1, len(unique_labels)))
    #设置颜色
    colors = ['#83639f','#c22f2f', '#3490de','#449945', '#1f70a9', '#ea7827', '#F8766D','#abedd8','#f6416c','#ffd460','#6c5b7b','#53354a','#e84545']#NDM

    
    for i, label in enumerate(unique_labels):
        plt.scatter(embedding[ylabel == label, 0], embedding[ylabel == label, 1], c=colors[i], label=f'Cluster {label}', s = 0.6)
    # plt.title(name, size=12)
    plt.xticks([])
    plt.yticks([])
    if embedding_way == 'PCA':
        plt.xlabel('PC1',size = 8)
        plt.ylabel('PC2',size = 8)
    elif embedding_way == 'UMAP':
        plt.xlabel('UMAP1',size = 8)
        plt.ylabel('UMAP2',size = 8)
    else:
        plt.xlabel('t-SNE1',size = 8)
        plt.ylabel('t-SNE2',size = 8)
    plt.tight_layout()
    # plt.legend()
    plt.rcParams['figure.dpi'] = 300
    plt.savefig(savePath+name+'.eps',format = 'eps')
    plt.show()
    plt.close(fig)

from tqdm import tqdm
for GSEname in tqdm(name):
    GSEdir = './data/' + GSEname +'.h5ad'
    adata = tool.scDataCluster(GSEdir, cls = 'h5', min_cells = 10, highlyVarGene= 2000)
    if GSEname == 'Jiaxu2017':
        label = adata.obs['Day']
    else:
        label = adata.obs['cell type']
    dataName = GSEname + '2000'
    csnDataDir = workDir + '/' + dataName + 'csnData/'
    savePath = csnDataDir + 'result'
    mtx = adata.X #(cell,gene)

    cls = 'GIR'
    centralityDir = savePath + '/' +cls +'Matrix.csv'
    GirM = np.array(pd.read_csv(centralityDir, header = None))

    cls = 'Degree'
    centralityDir = savePath + '/' +cls +'Matrix.csv'
    DegreeM = np.array(pd.read_csv(centralityDir, header = None))


    n_clusters = len(np.unique(label))
    savePath = '../CSNData/cluster_data/pca1/'+GSEname
    drawUmap(mtx, name = 'GEM',ylabel = label,cls = 'KMeans', embedding_way = 'PCA',size = [2,2], n_clusters = n_clusters,savePath = savePath, component = 4, u_components = 2)
    drawUmap(DegreeM, name = 'NDM',ylabel = label,cls = 'KMeans', embedding_way = 'PCA',size = [2,2], n_clusters = n_clusters,savePath = savePath, component = 50, u_components = 2)
    drawUmap(GirM, name = 'GIR',ylabel = label,cls = 'KMeans', embedding_way = 'PCA',size = [2,2], n_clusters = n_clusters,savePath = savePath, component = 50, u_components = 2)

    n_clusters = len(np.unique(label))
    savePath = '../CSNData/cluster_data/tsne1/'+GSEname

    drawUmap(mtx, name = 'GEM',ylabel=label,cls = 'KMeans', embedding_way = 'TSNE',size = [2,2], n_clusters = n_clusters,savePath = savePath, component =16, u_components = 2)
    drawUmap(DegreeM, name = 'NDM',ylabel=label,cls = 'KMeans', embedding_way = 'TSNE',size = [2,2], n_clusters = n_clusters,savePath = savePath, component = 100, u_components = 2)
    drawUmap(GirM, name = 'GIR',ylabel=label,cls = 'KMeans', embedding_way = 'TSNE',size = [2,2], n_clusters = n_clusters,savePath = savePath, component = 64, u_components = 2)
    
    
    n_clusters = len(np.unique(label))
    savePath = '../CSNData/cluster_data/umap1/'+GSEname
    drawUmap(mtx, name = 'GEM',ylabel=label,cls = 'KMeans', embedding_way = 'UMAP',size = [2,2], n_clusters = n_clusters,savePath = savePath, component = 2, u_components = 2)
    drawUmap(DegreeM, name = 'NDM',ylabel=label,cls = 'KMeans', embedding_way = 'UMAP',size = [2,2], n_clusters = n_clusters,savePath = savePath, component = 8, u_components = 2)
    drawUmap(GirM, name = 'GIR',ylabel=label,cls = 'KMeans', embedding_way = 'UMAP',size = [2,2], n_clusters = n_clusters,savePath = savePath, component = 8, u_components = 2)


