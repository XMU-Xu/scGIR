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
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
#10X_PBMC\GSE75748sctime\mouse_bladder_cell\GSE82187\GSE90047\GSE81861\Buettner\Jiaxu2017
#mouse_ES_cell\Trapnell\GSE102066\GSE71585
#%%
GSEname = 'GSE75748'

GSEdir = './data/' + GSEname +'.h5ad'

# adata = tool.scDataCluster(GSEdir, cls = 'h5',min_cells=10, highlyVarGene= 2000)
adata = tool.scDataCluster(GSEdir, cls = 'h5')
# label = adata.obs['cell type']
adata.X.shape
#%%
# label = adata.obs['cell type']
dataName = GSEname + '2000'
csnDataDir = workDir + '/' + dataName + 'csnData/'
savePath = csnDataDir + 'result'
mtx = adata.X #(cell,gene)
adata2 = adata

# X = np.array(mtx).T
cls = 'GIR'
centralityDir = savePath + '/' +cls +'Matrix.csv'
GirM = np.array(pd.read_csv(centralityDir, header = None))

cls = 'Degree'
centralityDir = savePath + '/' +cls +'Matrix.csv'
DegreeM = np.array(pd.read_csv(centralityDir, header = None))
#%%
min_val = np.min(GirM)
max_val = np.max(GirM)

# 计算数值范围的大小
range_val = max_val - min_val

# 对矩阵中的每个元素进行缩放
scaled_matrix = (GirM - min_val) / range_val
adata.X = scaled_matrix
#%%

adata.X = scaled_matrix
sc.pp.neighbors(adata,use_rep='X', n_neighbors=50, n_pcs=50)

sc.tl.leiden(adata,resolution = 0.8)
sc.tl.umap(adata)
sc.pl.umap(adata, color='cell type')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.show()
#%%
adata.X = mtx
adata.rename_categories('leiden', np.unique(adata.obs['cell type']))
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
# %%
adata.X = scaled_matrix
adata.write('../CSNData/chutype.h5ad', compression='gzip') 
#%%
adata.X = mtx
import matplotlib.cm as cm

sc.tl.rank_genes_groups(adata, 'cell type', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
marker_genes = list(adata.uns['rank_genes_groups']['names'][0])+list(adata.uns['rank_genes_groups']['names'][1])

sc.pl.dotplot(adata, marker_genes, cmap = 'coolwarm',groupby='cell type', figsize=[10,5],dendrogram=True);
# %%

adata.X = scaled_matrix
sc.pl.umap(adata, color='CER1')

# %%
adata.X = mtx
sc.pl.umap(adata, color=marker_genes)
#%%
adata.X = DegreeM
sc.pl.umap(adata, color=marker_genes)

# %%
'PTN''TERF1''SERPINB9''RHOBTB3'
sc.pp.neighbors(adata,use_rep='X', n_neighbors=50, n_pcs=50)
plt.figure(figsize = (5,5))
plt.rcParams['figure.dpi'] = 300
# sc.tl.leiden(adata,resolution = 0.8)
sc.tl.tsne(adata)
sc.pl.tsne(adata, color='cell type',save = 'ccccccccccccccccccccccc.eps')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

#%%
marker_genes = ['UBE2L6','HDDC2','COLEC12', 'RPS4Y1', 'LIN28B', 'CD9', 'VCAN', 'PTN','TERF1','SERPINB9','RHOBTB3']
gname = 'CER1'
plt.figure(figsize = (5,5))
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
sc.tl.tsne(adata)
adata.X = scaled_matrix
sc.pl.tsne(adata, color=gname,save = gname +'GIM.eps')
adata.X = mtx
sc.pl.tsne(adata, color=gname,save = gname +'GEM.eps')
# adata.X = DegreeM
# sc.pl.tsne(adata, color=gname,save = gname +'NDM.eps')
#%%
adata.X = scaled_matrix
sc.pl.violin(adata, keys=['NLRP2'], groupby='cell type')
#%%
#h1 100, h9 300 DEC 500 EC 600 HFF 700 NPC 800 TB 1000
cellN = 808
netName = adata.obs['cell type'][cellN]
adata.obs['cell type'][cellN]

# %%
mean_G = {}
for i in np.unique(adata.obs['cell type']):
    mean_G[i] =[]
mean_G

for i in range(len(adata.obs['cell type'])):
    mean_G[adata.obs['cell type'][i]].append(i)
# %%
mean_GM = []
for cell in np.unique(adata.obs['cell type']):
    c_l = []
    for i in range(mtx.shape[1]):
        g = []
        for c in mean_G[cell]:
            g.append(mtx[c,i])
        c_l.append(np.mean(np.array(g)))
    mean_GM.append(c_l)


#%%
all_genes = adata.var_names.tolist()
gene_name = 'RPS4Y1'
gene_index = all_genes.index(gene_name)

marker_genes = ['EFEMP1','DHRS3','TERF1', 'RPS4Y1', 'LIN28B', 'CD9', 'IGFBP4', 'LOXL3','CBR1','COLEC12','EIF1AY','DSC2','ZNF572','TPM1']
import networkx as nx
for c in ['NPC']:
    #下面这个列表放细胞
    cell_list = [5,6,2,3,4,100,101,102,103,104]
    for i in cell_list:
        netName = c
        print(mean_G[c][i])
        cellN = mean_G[c][i]
        k = cellN
        X = mtx.T
        from scipy import sparse
        path = csnDataDir
        cellsSP = sparse.load_npz(path + 'csn' + str(k) + '.npz') # 读取
        cellsMat = cellsSP.toarray()
        # np.savetxt('csn0.csv', cellsMat, delimiter=",")

        geneExp = X[:,k]
        M = np.multiply(cellsMat, geneExp)
        sumM = np.sum(M, axis = 1) + 0.0001
        broSumM = np.ones(M.shape)*sumM
        edgeW = M / broSumM.T
        l1,l2 = np.nonzero(edgeW)
        DG = nx.from_numpy_array(cellsMat, create_using=nx.DiGraph)
        for i in range(l1.shape[0]):
            DG.add_weighted_edges_from([(l1[i], l2[i], edgeW[l1[i]][l2[i]])])


        gi = []
        for g in marker_genes:
            gi.append(adata.var_names.get_loc(g))

        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        colormap = cm.coolwarm

        # G = nx.from_numpy_array(np.array(adjacency_matrix))
        nodes_to_draw = gi
        node_name = marker_genes
        node_label = {}
        for i in range(len(node_name)):
            node_label[nodes_to_draw[i]] = node_name[i]

        G = DG.subgraph(nodes_to_draw)
        nx.set_node_attributes(G, node_label, "label")
        pos = nx.layout.circular_layout(G)

        NodeId=list(G.nodes())
        node_size=[G.degree(i)**1.2*90 for i in NodeId]
        node_color=[mtx[k,all_genes.index(i)]**1.2*90 for i in marker_genes]

        options={'linewidths':10,'width':0.4,'style':'solid','nodelist':NodeId,'node_color':node_size,'font_color':'w'}

        fig,ax=plt.subplots(figsize=(6,6))
        nx.draw(G,pos = pos,ax = ax, node_shape='o', with_labels=False,node_size=node_size,**options)
        nx.draw_networkx_labels(G, pos=pos, labels=node_label,font_size = 14)

        min_weight = min(d['weight'] for (u, v, d) in G.edges(data=True))
        max_weight = max(d['weight'] for (u, v, d) in G.edges(data=True))
        norm = Normalize(vmin=min_weight, vmax=max_weight)
        cmap = colormap
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        edge_colors = [sm.to_rgba(d['weight']*10) for (u, v, d) in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        plt.savefig('../CSNData/cluster_data/NPC/'+netName+str(cellN)+'.eps',format = 'eps')
        plt.show()
# %%
