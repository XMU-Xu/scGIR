#%%
import os
import sys
import time
import stat
import numpy as np
import pandas as pd
import scanpy as sc
workDir = sys.path[0]
os.chdir(workDir)

import utils as tool
import ComplexNetM as CNet
import CsnConstruct as Csn

def scGIR(dataDir, name, min_cells, highlyVarGene):

    adata = tool.scDataCluster(dataDir, cls = 'h5', min_cells = min_cells, highlyVarGene = highlyVarGene)
    mtx = adata.X
    dataName = name + str(highlyVarGene)
    csnDataDir = workDir + '/' + dataName + 'csnData/'
    savePath = csnDataDir + 'result'
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    os.chmod(savePath, stat.S_IWRITE)

    X = np.array(mtx).T

    Csn.csnConstruct(X, path=csnDataDir)
    cls = 'GIR'
    centralityDir = savePath + '/' +cls +'Matrix.csv'
    CNet.geneCentralityRank(X, path = csnDataDir, savePath = savePath, centralityDir = centralityDir,  cells = X.shape[1])
    cls = 'PageRank'
    centralityDir = savePath + '/' +cls +'Matrix.csv'
    CNet.martrixCentrality(path = csnDataDir, savePath = savePath, centralityDir = centralityDir,  cells = X.shape[1], cls = cls)
    cls = 'Degree'
    centralityDir = savePath + '/' +cls +'Matrix.csv'
    CNet.martrixCentrality(path = csnDataDir, savePath = savePath, centralityDir = centralityDir,  cells = X.shape[1], cls = cls)
    

scGIR('./data/GSE75748.h5ad', 'GSE75748', 10, 2000)



# %%
