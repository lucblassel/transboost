# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 13:45:19 2018

@author: xuebai
"""

import pandas as pd
import numpy as np


def readlog(txtfile, minlines_s, minlines_b):
    
    nb_times_small = np.arange(10, minlines_s+1, 10)
    nb_times_big = np.arange(10, minlines_b+1, 10)
    
    tuples1 = np.concatenate((['small']*(minlines_s/10),['big']*(minlines_b/10)),axis=0)
    tuples2 = np.concatenate((nb_times_small, nb_times_big),axis=0)
    tuples = list(zip(*[tuples1,tuples2]))
    
    index = pd.MultiIndex.from_tuples(tuples,names= ['net','times'])
    
    data = pd.DataFrame(index=index)
    
    with open(txtfile) as f:
        direc = f.readlines()
        
    for path in direc:
        with open(path) as log:
            lines = log.readlines()
        for line in lines:
            if line[0:9] == "threshold":
                colname=line
                data[colname] = []
            if 'bigNet :  False' in line:
                net = 'small'
            elif 'bigNet :  True' in line:
                net ='big'
            if line[0:4] == "time":
                line = line.split() 
                data[net,line[1],colname]=line[4]
    return data

def main():
    data = readlog("resultlog.txt",300,300)
    print(data)
    
if __name__ == "__main__":
    main()
                