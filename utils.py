import umap
import umap.plot
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn_extra.cluster import KMedoids
from sklearn import metrics

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
    
    if target == 'ARI':
        score = metrics.adjusted_rand_score(y_pred.labels_,label)
    elif target == 'FM':
        score = metrics.fowlkes_mallows_score(y_pred.labels_,label)
    elif target == 'NMI':
        score = metrics.normalized_mutual_info_score(y_pred.labels_,label)
    return score
def save2Csv(
    matrix, 
    name,
    dir = './',
    header=None,
    index= None
):
    """
        save matrix to csv file
    Args:
        matrix
            numpy array or Dataframe. 
        dir
            directory of file, Default = './'
        header
            Write out the column names. If a list of strings is given it is
            assumed to be aliases for the column names.
        index
            Write row names.
    """
    if isinstance(matrix, pd.DataFrame):
        matrix.to_csv(dir+name+'.csv',header= header, index= index)
    elif isinstance(matrix, np.ndarray):
        if len(matrix.shape) == 2:
            np.savetxt(dir+name+'.csv', matrix, delimiter=",")
        elif len(matrix.shape) == 3:
            with open(dir+name+'.csv', 'w') as f:
                for slice_2d in matrix:
                    np.savetxt(f, slice_2d, fmt = '%f', delimiter = ',')
    else: raise NameError('Ensure that the data types are numpy of dataframe')

def draw(
    x,
    y,
    xlabel = 'x',
    ylabel = 'y',
    title = 'title',
    xfontsize = 12,
    yfontsize = 12,
    tfontsize = 15,
    cls = 'plot',
    color = 'red',
    legend = 'best',
    scatter_s = 10,
):
    plt.title(title, fontsize = tfontsize)
    plt.xlabel(xlabel, fontsize = xfontsize)
    plt.ylabel(ylabel, fontsize = yfontsize)
    if cls == 'plot':
        plt.plot(x,y,c = color)
    elif cls == 'scatter':
        plt.scatter(x,y,c = color, s=scatter_s, label = legend)
    plt.legend(loc = legend)
    plt.show()

def drawHist(
    x,
    xlabel = 'p',
    ylabel = 'Density',
    title = 'title',
    figsize = [12, 8],
    dpi = 80,
    xfontsize = 12,
    yfontsize = 12,
    tfontsize = 15,
    color = 'red',
    range = None,
    save = False,
    fit = True,
    dir = './hist.png',
    width = 100,
):
    plt.figure(figsize=figsize, dpi = dpi)
    plt.title(title, fontsize = tfontsize)
    plt.xlabel(xlabel, fontsize = xfontsize)
    plt.ylabel(ylabel, fontsize = yfontsize)
    n, bins, patches = plt.hist(x, bins=width, range = range)
    if fit:
        X = bins[0:width] + (bins[1] - bins[0])/2.0
        Y = n
        plt.plot(X,Y, color = 'green')
        p1 = np.polyfit(X, Y, 7)
        Y1 = np.polyval(p1, X)
        plt.plot(X, Y1, color = 'red')
    if save: plt.savefig(dir)
    plt.show()

def drawUmap(
    mtx,
    name,
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
    y_pred = switch[cls](dmtx)
    if embedding_way == 'UMAP':
        embedding = umap.UMAP(n_components= u_components).fit_transform(mtx)
    elif embedding_way == 'TSNE':
        
        embedding = TSNE(n_components = u_components).fit_transform(dmtx)
    else:
        embedding = PCA(n_components = u_components).fit_transform(mtx)
        
    unique_labels = np.unique(y_pred.labels_)
    colors = plt.get_cmap(c)(np.linspace(0, 1, len(unique_labels)))
    colors = ['#83639f','#c22f2f', '#3490de','#449945', '#1f70a9', '#ea7827', '#F8766D','#abedd8','#f6416c','#ffd460']#
    
    for i, label in enumerate(unique_labels):
        plt.scatter(embedding[y_pred.labels_ == label, 0], embedding[y_pred.labels_ == label, 1], c=colors[i], label=f'Cluster {label}', s = 0.6)
    plt.title(name, size=12)
    plt.xticks([])
    plt.yticks([])
    if embedding_way == 'PCA':
        plt.xlabel('PC1',size = 8)
        plt.ylabel('PC2',size = 8)
    else:
        plt.xlabel('t-SNE1',size = 8)
        plt.ylabel('t-SNE2',size = 8)
    plt.tight_layout()
    plt.rcParams['figure.dpi'] = 300
    plt.savefig(savePath+name)
    plt.show()
    plt.close(fig)

def scDataCluster(path, cls = 'h5', min_cells = None, percent = None, highlyVarGene = None):
    if cls == 'h5':
        adata = sc.read_h5ad(path)
    elif cls == 'csv':
        adata = sc.read_csv(path)
    elif cls == 'loom':
        adata = sc.read_loom(path)
    else:
        adata = sc.read_text(path,delimiter = ',')
    if min_cells:
        sc.pp.filter_genes(adata, min_cells= min_cells)
    elif percent:
        sc.pp.filter_genes(adata, min_cells= adata.X.shape[0] * percent)
    
    sc.pp.log1p(adata)
    if highlyVarGene: 
        sc.pp.highly_variable_genes(adata, n_top_genes = highlyVarGene)
        adata = adata[:,adata.var['highly_variable']]
    
    return adata

