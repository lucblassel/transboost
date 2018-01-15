"""
-----------------------------------------------
# @Author: Luc Blassel
# @Date:   2018-01-15T11:30:16+01:00
# @Email:  luc.blassel@agroparistech.fr
# @Last modified by:   lucblassel
# @Last modified time: 2018-01-15T11:46:08+01:00

BLASSEL Luc

-----------------------------------------------
"""

import keras

class callbackBoosting(keras.callbacks.Callback):
    def _init_(self,threshold):
        super(callbackBoosting,self)._init_()
        self.threshold = threshold

    def on_train_end(self, logs={}):
        print('training ended')
        retur

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_acc') >= self.threshold:
            self.model.stop_training = True
            print('stopping training at accuracy = '+str(logs.get('val_acc')))
        return
