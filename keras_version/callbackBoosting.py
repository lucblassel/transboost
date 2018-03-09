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
    """Initializer the class and its attributes"""
        super(callbackBoosting,self).__init__()  

        #threshold of weak projectors
        self.threshold = threshold   

        #metric of weak projectors, precision of model
        self.metric = metric     

        self.modelPath = modelPath
        self.verbose = verbose

    def on_train_end(self, logs={}):
	"""Print 'training ended’ at the end of training"""
        if self.verbose:
            print('training ended')
        return

    def on_epoch_end(self, epoch, logs={}):
	"""
	Stop training and save weights and path of model when 
	its precision is equal or greater than the threshold
	"""

        if logs.get(self.metric) >= self.threshold:
            self.model.stop_training = True
            saveModelWeigths(self.model,self.modelPath)
            if self.verbose:
                print('stopping training at accuracy = '+str(logs.get(self.metric))+"on epoch number "+str(epoch+1))
        return
