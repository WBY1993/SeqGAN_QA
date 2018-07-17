# -*- coding: utf-8 -*-
import configuration
import preprocess
import dataloader
import numpy as np

import tensorflow as tf

t = tf.constant([1,2,3])
r = tf.constant([1,2,3])
sess = tf.Session()
print(sess.run(t * r))
