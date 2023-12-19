"""
Construction of cell-specific networks
"""
import os
import stat
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy import sparse
from tqdm import tqdm
from utils import *

class CustomTask:
    def __init__(self, func):
        self._result = []
        self._func = func

    def run(self, *args):
        result = self._func(*args)
        self._result.append(result)
    
    def getResult(self):
        return self._result

def neighborhoodDef(
    rows, #n1
    cols, #n2
    boxsize,
    mat,
)->np.array:
    """
    Define the neighborhood of each plot
    ----------
    Args:
        rows
            Gene expression
        cols 
            Cells
        boxsize
            Size of neighborhood.
        mat 
            Gene expression matrix, rows = genes, columns = cells.
    Returns:
        gene_k boundary
    """
    upper = np.zeros((rows, cols)) #box boundary
    lower = np.zeros((rows, cols))
    for i in range(rows): #每个基因x_k都要定一个上下限，所以是从
        geneIdx = np.argsort(mat[i,:])#s2
        geneX = np.sort(mat[i,:]) #s1
        epsCells = np.sum(np.sign(geneX))
        unepsCells = cols - epsCells #n3
        h = boxsize/2 * epsCells #side of box, epscells * boxsize = n_x or n_y = 0.1n
        if h == 0.5:
            h = 1
        else:
            h = np.round(h)
        k = 0
        while k < cols:
            s = 0
            while k+s+1 < cols and geneX[k+s+1] == geneX[k]: #跳过表达量相同的，一般用来跳过0
                s = s + 1

            if s >= h:
                for j in range(s+1):
                    upper[i, geneIdx[k+j]] = mat[i, geneIdx[k]]
                    lower[i, geneIdx[k+j]] = mat[i, geneIdx[k]]
            else:
                for j in range(s+1):
                    upper[i, geneIdx[k+j]] = mat[i, geneIdx[int(min(cols-1, k+s+h))]]
                    lower[i, geneIdx[k+j]] = mat[i, geneIdx[int(max(unepsCells*(unepsCells>h), k-h))]]
                    
            k = k + s + 1
    
    return upper, lower

class CustomTask:
    def __init__(self, func):
        self._result = []
        self._func = func

    def run(self, *args):
        result = self._func(*args)
        self._result.append(result)
    
    def getResult(self):
        return self._result
def mutilTask(start, end, mat, upper, lower, B, p, weighted):
    csn = []
    prox = []
    rows =  mat.shape[0] # genes
    cols = mat.shape[1] # cells
    for k in range(start, end):
        for j in range(cols):
            for g in range(rows):
                if mat[g,j] <= upper[g,k] and mat[g,j] >= lower[g,k]:
                    B[g, j] = 1
                else: B[g, j] = 0
        colSumMat = np.sum(B, axis=1)
        colSumMat = colSumMat.reshape((np.shape(colSumMat)[0],1))
        eps = np.finfo(float).eps

        distence = (np.dot(B,B.T)*cols-np.dot(colSumMat,colSumMat.T))/np.sqrt(np.dot(colSumMat,colSumMat.T)*(np.dot((cols-colSumMat),(cols-colSumMat).T)/(cols-1)+eps))
        distence = distence - distence*np.eye(distence.shape[0])
        
        prox.append(distence)
        retM = np.zeros(distence.shape)
        retM[distence>=p] = 1
        csn.append(retM)
    return csn, prox

def csnConstruct(
    mat,
    path = './',
    cells = None,
    alpha = 0.01,
    boxsize = 0.1,
    weighted = 0,
    p = None,
) -> np.array :
    """
    Input gene x cell matrix
    Construct matrix of cell-specific networks.
    ----------
    Args:
        mat
            Gene expression matrix, rows = genes, columns = cells.
        cells
            Construct the CSNs for all cells, set c = [] (Default);
            Construct the CSN for cell k, set  c = k
        alpha
            Significant level (eg. 0.001, 0.01, 0.05 ...).
        boxsize
            Size of neighborhood, Default = 0.1.
        weighted
            1  edge is weighted
            0  edge is not weighted (Default)
        p
            Level of significance
    Returns:
        Cell-specific network 
    """
    if not os.path.exists(path):
        os.mkdir(path)
    os.chmod(path, stat.S_IWRITE)
    mat.astype(np.float32)
    if not cells :
        cells = []
        for i in range(np.size(mat, 1)):
            cells.append(i)
    rows =  mat.shape[0] # genes
    cols = mat.shape[1] # cells
    upper, lower = neighborhoodDef(rows, cols, boxsize, mat)
    csn = []
    prox = []
    B = np.zeros((rows, cols))
    
    if not p:
        p = -st.norm.ppf(alpha)
        
    
    for k in tqdm(cells):
        for j in range(cols):
            for g in range(rows):
                if mat[g,j] <= upper[g,k] and mat[g,j] >= lower[g,k]:
                    B[g, j] = 1
                else: B[g, j] = 0
        colSumMat = np.sum(B, axis=1)
        colSumMat = colSumMat.reshape((np.shape(colSumMat)[0],1))
        eps = np.finfo(float).eps

        distence = (np.dot(B,B.T)*cols-np.dot(colSumMat,colSumMat.T))/np.sqrt(np.dot(colSumMat,colSumMat.T)*(np.dot((cols-colSumMat),(cols-colSumMat).T)/(cols-1)+eps))
        distence = distence - distence*np.eye(distence.shape[0])
        
        retM = np.zeros(distence.shape)
        retM[distence>=p] = 1

        retMSp=sparse.csr_matrix(retM)
        sparse.save_npz(path + 'csn' + str(k) + '.npz',retMSp)



