# -*- coding: utf-8 -*-
# @Author: Romain
# @Date:   2017-11-28 10:51:04
# @Last Modified by:   Romain
# @Last Modified time: 2017-11-28 11:08:18
import os
os.chdir("/Users/Romain/Documents/Cours/APT/IODAA/fil rouge")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta


# Functions and classes for loading and using the Inception model.
import inception

# We use Pretty Tensor to define the new classifier.
import prettytensor as pt



from IPython.display import Image, display
Image('images/08_transfer_learning_flowchart.png')



if __name__ == '__main__':
	main()