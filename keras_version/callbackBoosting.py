"""
-----------------------------------------------
# @Author: Luc Blassel
# @Date:   2018-01-15T11:30:16+01:00
# @Email:  luc.blassel@agroparistech.fr
# @Last modified by:   lucblassel
# @Last modified time: 2018-01-15T11:55:22+01:00

BLASSEL Luc

-----------------------------------------------
"""

import keras
from boosting import saveModelWeigths

class callbackBoosting(keras.callbacks.Callback):
    def __init__(self,threshold,metric,modelPath,verbose):
        super(callbackBoosting,self).__init__()
        self.threshold = threshold
        self.metric = metric
        self.modelPath
        self.verbose = verbose

    def on_train_end(self, logs={}):
        if self.verbose:
            print('training ended')
        return

    def on_epoch_end(self, epoch, logs={}):
        if logs.get(self.metric) >= self.threshold:
            self.model.stop_training = True
            self.model.saveModelWeigths(self.modelPath)
            if self.verbose:
                print('stopping training at accuracy = '+str(logs.get(self.metric))+"on epoch number "+str(epoch+1))
        return
