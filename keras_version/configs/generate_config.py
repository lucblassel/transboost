# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:29:27 2018

@author: XU xuefan

"""
import json

def generate_config(threshold,maxtime):
    f_name='config_threshold%d_maxtime%d'%(int(threshold*100),maxtime)
    d={
       "models_path" : "models",
       "models_weights_path" : "models_weights",
       "path_to_best_model" : "best_top_model.hdf5",
       "threshold" : threshold,
       "proba_threshold" : 0.5,
       "transformation_ratio" : 0.05,
       "originalSize" : 32,
       "resizeFactor" : 5,
       "batch_size_source" : 10,
       "batch_size_target" : 10,
       "epochs_source" : 100,
       "epochs_target" : 100,
       "classes_source" : ["dog","truck"],
       "classes_target" : ["deer","horse"],
       "layerLimit" : 15,
       "times" : maxtime,
       "lr_source" : 0.0001,
       "lr_target" : 0.0001,
       "recompute_transfer_values" : False,
       "train_top_model" : False,
       "reinitialize_bottom_layers" : False,
       "bigNet" : False
       }
    with open('%s.json'%(f_name),'w') as f:
        json.dump(d,f)
    with open('configBatch.conf','a') as g:
        path_name='configs/%s\n'%('%s.json'%(f_name))
        g.write(path_name)
#    with open('1.txt','w') as f:
#        f.write('s')

def main():
    maxtime=500
    for threshold in [0.55,0.6,0.65,0.7]:
        generate_config(threshold,maxtime)
    
if __name__=='__main__':
    main()