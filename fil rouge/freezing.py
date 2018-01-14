# -*- coding: utf-8 -*-
# @Author: Romain
# @Date:   2017-12-07 18:57:26
# @Last Modified by:   Romain
# @Last Modified time: 2017-12-08 11:05:48
import os
os.chdir("/Users/Romain/Documents/Cours/APT/IODAA/transboost/fil rouge")

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

##############################
# LOADING model #
##############################
# Functions and classes for loading and using the Inception model.
import inception

inception.maybe_download()

model = inception.Inception()

##############################
# getting layers names #
##############################
op = model.graph.get_operations()
[m.values() for m in op][0:50]
[m.values() for m in op][len([m.values() for m in op])-1]