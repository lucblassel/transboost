# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 13:45:19 2018

@author: xuebai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def readlog(txtfile, minlines_s, minlines_b):
    
    nb_times_small = np.arange(10, minlines_s+1, 10)
    nb_times_big = np.arange(10, minlines_b+1, 10)
    
    tuples1 = np.concatenate((['small']*int(minlines_s/10),['big']*int(minlines_b/10)),axis=0)
    tuples2 = np.concatenate((nb_times_small, nb_times_big),axis=0)
    tuples = list(zip(*[tuples1,tuples2]))
    
    index = pd.MultiIndex.from_tuples(tuples,names= ['net','times'])
    
    data = pd.DataFrame(index=index)
    
    netdic = {"small":minlines_s,"big":minlines_b}
    
    with open(txtfile) as f:
        direc = f.readlines()
        
    for path in direc:
        with open(path.rstrip()) as log:
            lines = log.readlines()
        for line in lines:
            if len(line)>9 and line[0:9] == "threshold":
                colname=line.rstrip()
                data[colname] = None
                #print(colname)
            if 'bigNet :  False' in line:
                net = 'small'
            elif 'bigNet :  True' in line:
                net ='big'
                #print(net)
            if len(line)>4 and line[0:5] == "time:":
                line = line.split()
                idx = pd.IndexSlice
                data.loc[idx[net,int(line[1])],colname]=float(line[-1])
                if line[1] == netdic[net]:
                    break
    return data

    

def main():
    data = readlog("resultlog.txt",300,300)
    print(data)

    small = data.xs('small',level = 'net')
    #big = data.xs('big',level = 'net')

    plt.figure()
    paints = small.plot()
    
    
    paints.set_ylabel('accuracy')
    plt.ylim(0,1)
    paints.set_title("The accuracy at different thresholds of smallnet and bignet")
    plt.legend(loc ='best')
    
#    big.plot(ax=paints)
    
    plt.show()
    
if __name__ == "__main__":
    main()
                