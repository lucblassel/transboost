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

class callbackBoosting(keras.callbacks.Callback):
    def __init__(self,threshold):
        super(callbackBoosting,self).__init__()
        self.threshold = threshold

    def on_train_end(self, logs={}):
        print('training ended')
        return

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_acc') >= self.threshold:
            self.model.stop_training = True
            print('stopping training at accuracy = '+str(logs.get('val_acc')))
        return
