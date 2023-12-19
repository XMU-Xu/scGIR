import os
import sys
import time
import pandas as pd
import numpy as np
import multiprocessing
import ComplexNetM as CNet
import CsnConstruct as Csn
import utils as tool
os.chdir(sys.path[0])

if __name__ =='__main__':
    df = pd.read_csv('./cellcycle1/X.csv', header= None, index_col = None)
    start = time.perf_counter()
    csn, prox = Csn.csnConstructThreading(np.array(df).T)
    end = time.perf_counter()
    print("运行时间为", round(end - start), 'seconds')
    tool.save2Csv(np.array(csn), 'csn')
    tool.save2Csv(np.array(prox), 'prox')

    # csn = np.loadtxt('./csn.csv', delimiter = ',').reshape((-1, df.shape[0], df.shape[1]))
    # prox = np.loadtxt('./prox.csv', delimiter = ',').reshape((-1, df.shape[1], df.shape[1]))

    # m = CNet.martrixCentrality(csn)
    # p = np.array(prox)
    # # tool.drawUmap(m)
    # tool.drawHist(p[1,1,:])
    